import json
import os
import sys
from pathlib import Path
from fastapi import FastAPI, Body, APIRouter
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_sql_query_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
# from langchain_community.vectorstores.utils import filter_similar_documents

# ===== Load environment variables =====
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ===== Initialize FastAPI =====
# app = FastAPI()
router = APIRouter()

# ===== Business Rules Path =====
MD_FILE_PATH = Path(__file__).parent / "business_rules.md"

CONFIG_PATH = Path(__file__).parent / "db_config.json"
config_data = {}

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config_data = json.load(f)

# ===== LLM Initialization =====
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY
)

# ===== Read business rules =====
def read_business_rules():
    if not MD_FILE_PATH.exists():
        return ""
    return MD_FILE_PATH.read_text(encoding='utf-8')

business_rules = read_business_rules()

# ===== Vector Store Setup =====
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# You need to prepare the schema_docs as a list of Document objects with schema/sample descriptions
schema_docs = [
    Document(page_content="Table: users, Columns: id, name, email"),
    Document(page_content="Table: orders, Columns: id, user_id, amount"),
    # Load all schema docs...
]

vector_store = FAISS.from_documents(schema_docs, embedding_model)


# Add this helper function to handle document similarity
def filter_similar_documents(docs, similarity_threshold=0.95):
    """
    Filter out similar documents based on content similarity
    Args:
        docs: List of Document objects
        similarity_threshold: Threshold for considering documents similar
    Returns:
        List of filtered Document objects
    """
    if not docs:
        return []
    
    filtered_docs = [docs[0]]
    
    for doc in docs[1:]:
        is_similar = False
        for filtered_doc in filtered_docs:
            # Simple text similarity check
            similarity = len(set(doc.page_content.split()) & set(filtered_doc.page_content.split())) / \
                       len(set(doc.page_content.split()) | set(filtered_doc.page_content.split()))
            if similarity > similarity_threshold:
                is_similar = True
                break
        if not is_similar:
            filtered_docs.append(doc)
    
    return filtered_docs



# ===== Extract relevant business logic =====
def extract_relevant_business_rules(llm, business_rules: str, question: str) -> str:
    prompt = (
        "Business Rules:\n{rules}\n\n"
        "User Question:\n{q}\n\n"
        "Extract only the most relevant parts from the above business rules "
        "that apply to answering the question. If none apply, return 'None'."
    )
    query = prompt.format(rules=business_rules, q=question)
    result = llm.invoke(query)
    return result.content.strip()

# ===== Perform Vector Search =====
def get_combined_vector_search(question: str, extracted_rules: str, vector_store, top_k=5) -> str:
    if extracted_rules.lower() == "none":
        search_query = question
    else:
        search_query = f"{question}\nBusiness Logic: {extracted_rules}"
    results = vector_store.similarity_search(search_query, k=top_k)
    return "\n".join([doc.page_content for doc in results])

# ===== Create SQLAlchemy DB Engine =====
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True
)

# ===== LangChain SQL Database Wrapper =====
from langchain_community.utilities import SQLDatabase

db = SQLDatabase(engine)

# ===== SQL Prompt Template =====
sql_prompt = PromptTemplate.from_template(
    """You are a helpful SQL assistant for MSSQL Server.
Use the schema below and write a syntactically correct query that answers the question.

Schema Info:
{schema}

Question:
{question}

SQL:"""
)

# ===== SQL Chain Builder =====
def generate_sql_from_chain(question: str):
    extracted_rules = extract_relevant_business_rules(llm, business_rules, question)
    relevant_schema = get_combined_vector_search(question, extracted_rules, vector_store)

    sql_chain = create_sql_query_chain(
        llm=llm,
        db=db,
        prompt=sql_prompt,
        k=2
    )

    return sql_chain.invoke({
        "question": question,
        "schema": relevant_schema
    })

# ===== Query Endpoint =====
class QueryInput(BaseModel):
    question: str

@router.post("/query")
async def query_handler(input: QueryInput):
    try:
        sql = generate_sql_from_chain(input.question)
        stmt = text(" ".join(sql.replace("\t", " ").split()))
        with engine.connect() as conn:
            result = conn.execute(stmt).fetchall()
        data = [dict(row._mapping) for row in result]
        return {"status_code": 200, "sql": sql, "data": data}
    except Exception as e:
        return {"status_code": 500, "error": str(e)}

# ===== Business Rule Endpoints =====
@router.get("/business-rules", response_class=PlainTextResponse)
def get_business_rules():
    if not MD_FILE_PATH.exists():
        return {"status_code": 404, "message": "Markdown file not found."}
    return MD_FILE_PATH.read_text(encoding="utf-8")

@router.put("/business-rules")
def update_business_rules(updated_content: str = Body(..., embed=True)):
    try:
        MD_FILE_PATH.write_text(updated_content, encoding="utf-8")
        global business_rules
        business_rules = updated_content
        return {"status_code": 200, "message": "Markdown file updated successfully."}
    except Exception as e:
        return {"status_code": 500, "message": f"Failed to update file: {str(e)}"}

@router.post("/reload-business-rules")
def reload_business_rules():
    global business_rules
    business_rules = read_business_rules()
    return {"status": "success", "message": "Business rules reloaded."}

# ===== Mount Router =====
# app.include_router(router)

# ===== Run with: uvicorn thisfile:app --reload =====
