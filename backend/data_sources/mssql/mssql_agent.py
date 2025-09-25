import os
import sys
import time
from fastapi import Body, FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import create_engine

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from sqlalchemy import text
from fastapi.responses import PlainTextResponse

from typing import Any

from fastapi import APIRouter
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.getcwd())

router = APIRouter()
# import httpx
# from mcp.server.fastmcp import FastMCP


# Create an MCP server
# mcp = FastMCP(
#     name="mssql_server",
#     host="0.0.0.0",  # only used for SSE transport (localhost)
#     port=8001,  # only used for SSE transport (set this to any port)
# )

# 1️⃣ Load environment
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")       # e.g. "mssql+pyodbc://…"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 2️⃣ FastAPI setup
# class QueryRequest(BaseModel):
#     question: str

# 3️⃣ LLM initialization
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY
)
from sqlalchemy.pool import QueuePool
# 4️⃣ Database wrapper
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,  # Number of connections to maintain
    max_overflow=10,  # Maximum number of connections to allow above pool_size
    pool_timeout=30,  # Timeout for getting connection from pool
    pool_pre_ping=True  # Enable connection health checks
)        # uses SQLAlchemy & pyodbc
db = SQLDatabase(engine,
                 include_tables=["all_transactions","AllTables","expenseReport",
                                 "users","project","expenseItems","itemDescription",
                                 "roles","po","attendance","salary","requestMoney",
                                 "ps_RequsetPayment","all_transactions_returned"], 
                 sample_rows_in_table_info=2,  # Reduce sample rows
    max_string_length=100)  # Limit string length in samples)                       # .get_table_info() etc. :contentReference[oaicite:9]{index=9}

import os
from pathlib import Path

MD_FILE_PATH = Path(__file__).parent / "business_rules.md"

def read_business_rules():
    """Read business rules from MD file"""
    rules_path = Path(__file__).parent / "business_rules.md"
    if not rules_path.exists():
        return ""
    
    with open(rules_path, 'r', encoding='utf-8') as f:
        return f.read()



custom_prompt = PromptTemplate(
    input_variables=["input", "table_info", "top_k", "business_rules", "chat_history"],
    template=(
        "You are an expert MSSQL developer .\n"
        "Strictly use MSSQL syntax.\n"
        "Previous conversations:\n{chat_history}\n\n"
        "Here are the top {top_k} rows from each table:\n{table_info}\n\n"
        "Business Rules:\n{business_rules}\n\n"
        "Write only the SQL query (no commentary) to answer this request. "
        "Use Like operator with '%' before and after where necessary for safe filtering. "
        "Apply relevant business rules from above. "
        "Consider context from previous queries when relevant. "
        "Space is must where necessary. "
        "No newlines, no special characters except space, No extra words.\n\n"
        "Question: {input}\n"
        "SQLQuery:"
    )
)



# Modify the chain creation to include business rules
business_rules = read_business_rules()
sql_chain = create_sql_query_chain(
    llm=llm,
    db=db,
    prompt=custom_prompt,
    k=2    # number of sample rows per table
    # default_params={"business_rules": business_rules}
)


@router.post("/update-rules")
async def update_business_rules():
    """Reload business rules from MD file"""
    global business_rules
    business_rules = read_business_rules()
    return {"status": "success", "message": "Business rules updated"}

@router.get("/business-rules", response_class=PlainTextResponse)
def get_business_rules():
    """
    Reads and returns the contents of the markdown file.
    """
    if not MD_FILE_PATH.exists():
        return {"status_code":404, "message":"Markdown file not found."}
    return MD_FILE_PATH.read_text(encoding="utf-8")

@router.put("/business-rules")
def update_business_rules(updated_content: str = Body(..., embed=True)):
    """
    Updates the markdown file with new content provided in the request body.
    """
    try:
        MD_FILE_PATH.write_text(updated_content, encoding="utf-8")
        return {"status_code":200,"message": "Markdown file updated successfully."}
    except Exception as e:
        return {"status_code":500, "message":"Failed to update file: {e}"}

from memory_manager import ConversationMemory

# Initialize memory manager
memory_manager = ConversationMemory(max_messages=5)

@router.post("/query")
async def query_database(question: str, user_id: str = "default") -> dict:
    try:
        start_time = time.time()
        # Get conversation history
        history = memory_manager.get_conversation_history(user_id)
        
        # Add history to context
        context = {
            "chat_history": "\n".join([
                f"Previous Question: {conv['question']}\n"
                f"Generated Query: {conv['query']}\n"
                for conv in history
            ])
        }

        query_gen_time = time.time()
        # Generate SQL with history context
        sql = sql_chain.invoke({
            "question": question,
            "table_info": db.get_table_info(),
            "top_k": 2,
            "business_rules": business_rules,
            "chat_history": context["chat_history"]
        })
        query_gen_end_time = time.time()
        print(f"Query generation took {query_gen_end_time - query_gen_time:.2f} seconds")
        # Clean the SQL query
        sql = " ".join(sql.replace("\t", " ").split())
        stmt = text(sql)

        # Execute against the database
        with engine.connect() as conn:
            result = conn.execute(stmt).fetchall()
        
        # Format results
        data = [dict(row._mapping) for row in result]
        
        # Save to memory
        memory_manager.add_conversation(
            user_id=user_id,
            question=question,
            query=sql,
            results=data
        )
        end_time = time.time()
        print(f"Query executed in {end_time - start_time:.2f} seconds")
        return {
            "status_code": 200,
            "payload": {
                "sql": sql,
                "data": data,
                "history": history  # Include conversation history in response
            }
        }
    except Exception as e:
        return {"status_code": 500, "payload": {"error": str(e)}}

# Add endpoints to manage conversation history
@router.get("/conversation-history/{user_id}")
async def get_history(user_id: str):
    history = memory_manager.get_conversation_history(user_id)
    return {"status_code": 200, "payload": history}

@router.post("/clear-history/{user_id}")
async def clear_history(user_id: str):
    memory_manager.clear_conversation_history(user_id)
    return {"status_code": 200, "message": f"Conversation history cleared for user {user_id}"}



# if __name__ == "__main__":
#     transport = "sse"
#     if transport == "stdio":
#         print("Running server with stdio transport")
#         mcp.run(transport="stdio")
#     elif transport == "sse":
#         print("Running server with SSE transport")
#         mcp.run(transport="sse")
#     else:
#         raise ValueError(f"Unknown transport: {transport}")