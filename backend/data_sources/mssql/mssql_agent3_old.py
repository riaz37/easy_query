import os
import re
import sys
import time
import json
from typing import Dict, List, Tuple
from pathlib import Path
from fastapi import FastAPI, Body, APIRouter
from fastapi.responses import PlainTextResponse
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.getcwd())

# --- Environment Setup --- #
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- App & Router --- #
router = APIRouter()
app = FastAPI()

# --- LLM Initialization --- #
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY
)

# --- Engine Setup --- #
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True
)

# --- Paths --- #
BASE_DIR = Path(__file__).parent
TABLE_INFO_FILE = BASE_DIR / "table_info.txt"
MD_FILE_PATH = BASE_DIR / "business_rules.md"

# --- Memory Management --- #
# class ConversationMemory:
#     def __init__(self, max_messages=5):
#         self.memory = {}
#         self.max_messages = max_messages

#     def get_conversation_history(self, user_id: str):
#         return self.memory.get(user_id, [])

#     def add_conversation(self, user_id: str, question: str, query: str, results):
#         history = self.memory.get(user_id, [])
#         history.append({"question": question, "query": query, "results": results})
#         self.memory[user_id] = history[-self.max_messages:]

#     def clear_conversation_history(self, user_id: str):
#         self.memory[user_id] = []

# memory_manager = ConversationMemory()

from memory_manager import ConversationMemory
memory_manager = ConversationMemory(max_messages=5)

# --- Business Rules --- #
def read_business_rules():
    if not MD_FILE_PATH.exists():
        return ""
    return MD_FILE_PATH.read_text(encoding='utf-8')

business_rules = read_business_rules()

# --- Table Info Parsing --- #
def extract_create_table_blocks(sql: str) -> List[str]:
    pattern = re.compile(r'CREATE TABLE\s+[^\(]+\(', re.IGNORECASE)
    matches = list(pattern.finditer(sql))
    blocks = []
    for match in matches:
        start = match.start()
        open_parens = 0
        in_string = False
        i = match.end() - 1
        while i < len(sql):
            if sql[i] == "'" and (i == 0 or sql[i - 1] != "\\"):
                in_string = not in_string
            elif not in_string:
                if sql[i] == '(':
                    open_parens += 1
                elif sql[i] == ')':
                    open_parens -= 1
                    if open_parens == 0:
                        blocks.append(sql[start:i + 1])
                        break
            i += 1
    return blocks

def parse_custom_table_info_from_file(file_path: Path) -> Dict[str, Tuple[str, List[str]]]:
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found")
    content = file_path.read_text(encoding='utf-8')
    content = re.sub(r'\bGO\b', '', content, flags=re.IGNORECASE)
    table_blocks = extract_create_table_blocks(content)
    table_dict = {}
    for block in table_blocks:
        match = re.search(
            r'CREATE TABLE\s+(?:\[[^\]]+\]|\'[^\']+\'|\w+)',
            block,
            flags=re.IGNORECASE
        )
        if match:
            raw_name = re.search(r'CREATE TABLE\s+(?:\[(.*?)\]|\'(.*?)\'|(\w+))', block)
            name_parts = raw_name.groups() if raw_name else ()
            table_name = next((x for x in name_parts if x), None)
            if not table_name:
                continue
            inner = block[block.find('(') + 1:block.rfind(')')]
            column_names = []
            for line in inner.splitlines():
                line = line.strip()
                if not line or line.upper().startswith("CONSTRAINT"):
                    continue
                col_match = re.match(r'(?:\[(.*?)\]|(\w+))\s+', line)
                if col_match:
                    col_name = col_match.group(1) or col_match.group(2)
                    column_names.append(col_name)
            table_dict[table_name] = (block.strip(), column_names)
    return table_dict

def generate_slim_table_info(custom_table_info: dict) -> str:
    slim_info = ""
    for table_name, (ddl, columns) in custom_table_info.items():
        slim_info += f"Table: {table_name}\nColumns: {', '.join(columns)}\n\n"
    return slim_info.strip()

custom_table_info = parse_custom_table_info_from_file(TABLE_INFO_FILE)
table_info = generate_slim_table_info(custom_table_info)


def get_rules_suggestions(question: str, business_rules: str):
    system_template = """You are a Context Extractor assistant. Use the business rules to select the most relevant Business Rules to the user’s question."""
    user_template = """
    Question: {question}

    Business rules:
    {business_rules}

    Select the most relevant business rule for this Question from the Business rules mark Down File.
    If you got any relevant business rules, rule name, and details will be given there, You must return only that rule.
    return only one business rule with Business Rule, Tables, Outputs, conditions everything of that rule or None if no relevant business rule is found.
    If no business rules are available, just return None.
    """
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])
    rules_chain = LLMChain(llm=llm, prompt=chat_prompt)
    response = rules_chain.invoke({
        "question": question,
        "business_rules": business_rules
    })
    selected_rules = response["text"].strip()
    # cleaned = re.sub(r'^```json\n|```$', '', selected_tables_json.strip())
    return selected_rules


# --- Table Suggestion --- #
def get_table_suggestions(question: str, business_rules: str, tables_str: str):
    system_template = """You are a database assistant. Use the business rules and available table information to select the SQL tables most relevant to the user’s question."""
    user_template = """
    Question: {question}

    Available tables:
    {tables}

    Business rules:
    {business_rules}

    List the most relevant table names for this query as a JSON array of strings (e.g., ["table1","table2"]).
    If you got any relevant business rules, table name will be given there, You must return only those tables.
    If no business rules are available, just return the most 10 relevant tables based on the question and available tables.
    """
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])
    table_chain = LLMChain(llm=llm, prompt=chat_prompt)
    response = table_chain.invoke({
        "question": question,
        "tables": tables_str,
        "business_rules": business_rules
    })
    selected_tables_json = response["text"].strip()
    cleaned = re.sub(r'^```json\n|```$', '', selected_tables_json.strip())
    return json.loads(cleaned)

# --- Query Prompt --- #
query_prompt = PromptTemplate(
    input_variables=["question", "table_info", "business_rules", "chat_history"],
    template=(
        "You are an expert MSSQL developer.\n"
        "Available tables:\n{table_info}\n\n"
        "Business rules:\n{business_rules}\n\n"
        "Conversation context:\n{chat_history}\n\n"
        "You must use the sugeested tables from Business rules if mentioned.\n\n"
        "You must follow the Output format from business rules if mentioned.\n\n"
        "You must follow the conditions from business rules if mentioned.\n\n"
        "Use Like operator with '%' before and after where necessary for safe filtering. \n\n"
        "YOU MUST Use 'ISNULL(TRY_CAST(column AS INT), 0) AS column' instead of 'ISNULL(column, 0) AS column' in everywhere you need, to ensure the execution safety.\n\n"
        "MUST FOLLOW When writing UNION ALL queries, ensure that all fields (especially O_addedby and similar) have consistent data types across all SELECT blocks. If any column can contain non-numeric strings like 'N/A', use CAST(column AS NVARCHAR) for textual fields and ISNULL(TRY_CAST(column AS INT), 0) for integer fields to avoid conversion errors.\n\n"
        "if month or year nothing are given, take current month and current year (YEAR(GETDATE()) (i.e., current year from the system clock)) as month and year accordingly.\n\n"
        "if only month is given, take current year (YEAR(GETDATE()) (i.e., current year from the system clock)) as year. \n\n"
        "if month and year is given take both as is. \n\n"
        "Apply relevant business rules from above. \n\n"
        "Consider context from previous queries when relevant. \n\n"
        "Space is must where necessary. \n\n"
        "No newlines, no special characters except space, No extra words.\n\n"
        "Write ONLY the SQL query (no commentary) to answer:\n"
        "{question}"
    )
)
query_chain = LLMChain(llm=llm, prompt=query_prompt)

# --- FastAPI Routes --- #
@router.post("/query")
async def query_database(question: str, user_id: str = "default"):
    history = memory_manager.get_conversation_history(user_id)
    context = {
        "chat_history": "\n".join([
            f"Previous Question: {conv['question']}\n"
            f"Generated Query: {conv['query']}\n"
            for conv in history
        ])
    }

    business_rules_to_use = get_rules_suggestions(question, business_rules)
    table_names_to_use = get_table_suggestions(question, business_rules_to_use, str(list(custom_table_info.keys())))
    table_info_subset = generate_slim_table_info({k: custom_table_info[k] for k in table_names_to_use if k in custom_table_info})
    response = query_chain.invoke({
        "question": question,
        "table_info": table_info_subset,
        "business_rules": business_rules_to_use,
        "chat_history": context["chat_history"]
    })
    # sql = " ".join(str(response["text"]).replace("\t", " ").split())
    raw_sql = response["text"]
    # Remove markdown-style ```sql ... ```
    # clean_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
    clean_sql = re.sub(r"```(?:sql|text)?", "", raw_sql).strip()
    sql = " ".join(clean_sql.replace("\t", " ").split())
    stmt = text(sql)
    with engine.connect() as conn:
        result = conn.execute(stmt).fetchall()
    data = [dict(row._mapping) for row in result]
    memory_manager.add_conversation(
            user_id=user_id,
            question=question,
            query=sql,
            results=data
        )
    return {"status_code": 200, "payload": {"sql": sql, "data": data, "history": history}}

@router.post("/reload-db")
def reload_db():
    global custom_table_info, table_info
    custom_table_info = parse_custom_table_info_from_file(TABLE_INFO_FILE)
    table_info = generate_slim_table_info(custom_table_info)
    return {"status_code": 200, "message": "reloaded", "table_info_preview": table_info[:500]}

@router.get("/conversation-history/{user_id}")
async def get_history(user_id: str):
    history = memory_manager.get_conversation_history(user_id)
    return {"status_code": 200,"message":"History loaded successfully.", "payload": history}

@router.post("/clear-history/{user_id}")
async def clear_history(user_id: str):
    memory_manager.clear_conversation_history(user_id)
    return {"status_code": 200, "message": f"Conversation history cleared for user {user_id}"}

@router.get("/get_business-rules", response_class=PlainTextResponse)
def get_business_rules():
    return MD_FILE_PATH.read_text(encoding="utf-8") if MD_FILE_PATH.exists() else ""

# @router.put("/business-rules")
# def update_business_rules(updated_content: str = Body(..., embed=True)):
#     try:
#         MD_FILE_PATH.write_text(updated_content, encoding="utf-8")
#         return {"status": "success", "message": "Updated."}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

from fastapi.responses import FileResponse

@router.get("/get_business-rules_file", response_class=FileResponse)
def get_business_rules_file():
    if not MD_FILE_PATH.exists():
        return PlainTextResponse("Markdown file not found.", status_code=404)
    return FileResponse(
        path=MD_FILE_PATH,
        media_type='text/markdown',
        filename=MD_FILE_PATH.name
    )

from fastapi import UploadFile, File

@router.put("/update_business-rules")
async def update_business_rules(file: UploadFile = File(...)):
    if not file.filename.endswith(".md"):
        return {"status": "error", "message": "Only .md files are allowed."}
    try:
        contents = await file.read()
        MD_FILE_PATH.write_bytes(contents)
        return {"status": "success", "message": "Markdown file updated successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# --- Mount router --- #
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mssql_agent3_old:app", host="0.0.0.0", port=8200, reload=False)