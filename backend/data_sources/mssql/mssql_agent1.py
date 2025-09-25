import os
import re
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
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from typing import Any, Tuple

from fastapi import APIRouter
from pathlib import Path

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.getcwd())

router = APIRouter()

# Load environment
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM initialization
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY
)



from sqlalchemy.pool import QueuePool
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True
)

# db = SQLDatabase(engine,
#                  include_tables=["all_transactions", "AllTables", "expenseReport", "users", "project", "expenseItems", "itemDescription", "roles", "po", "attendance", "salary"],
#                  sample_rows_in_table_info=2,
#                  max_string_length=100)

db = None  # Global placeholder for lazy load

# def get_cached_table_info(force_reload=False):
#     global db
#     if force_reload or not TABLE_INFO_FILE.exists():
#         # Lazily load DB and regenerate table_info
#         db = SQLDatabase(
#             engine,
#             include_tables=["all_transactions", "AllTables", "expenseReport", "users", "project", "expenseItems", "itemDescription", "roles", "po", "attendance", "salary"],
#             sample_rows_in_table_info=2,
#             max_string_length=100
#         )
#         table_info = db.get_table_info()
#         TABLE_INFO_FILE.write_text(table_info, encoding='utf-8')
#         return table_info
#     else:
#         # Avoid loading DB
#         return TABLE_INFO_FILE.read_text(encoding='utf-8')


MD_FILE_PATH = Path(__file__).parent / "business_rules.md"
TABLE_INFO_FILE = Path(__file__).parent / "table_info.txt"

def read_business_rules():
    if not MD_FILE_PATH.exists():
        return ""
    return MD_FILE_PATH.read_text(encoding='utf-8')

def load_and_save_table_info():
    global db
    if db is None:
        db = SQLDatabase(
            engine,
            # include_tables=[
            #     "all_transactions", "AllTables", "expenseReport", "users", "project",
            #     "expenseItems", "itemDescription", "roles", "po", "attendance", "salary"
            # ],
            sample_rows_in_table_info=2,
            max_string_length=100
        )
    table_info = db.get_table_info()
    TABLE_INFO_FILE.write_text(table_info, encoding='utf-8')
    return table_info


def get_cached_table_info(force_reload=False):
    if force_reload or not TABLE_INFO_FILE.exists():
        return load_and_save_table_info()
    return TABLE_INFO_FILE.read_text(encoding='utf-8')

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
        # "SQLQuery:"
    )
)

from pathlib import Path
from typing import List

# def extract_table_names_from_sql_file(file_path: Path) -> List[str]:
#     if not file_path.exists():
#         return []

#     content = file_path.read_text(encoding='utf-8')

#     # Match CREATE TABLE table_name (no brackets assumed)
#     table_names = re.findall(r'CREATE TABLE\s+([^\s(]+)', content, flags=re.IGNORECASE)

#     return table_names


import re
from pathlib import Path
from typing import Dict


# def parse_custom_table_info_from_file(file_path: Path) -> Dict[str, str]:
#     """
#     Parses a .txt file with many CREATE TABLE statements (with or without brackets, schema, or quotes)
#     and returns a dictionary mapping table_name -> full CREATE TABLE SQL.
#     """
#     if not file_path.exists():
#         raise FileNotFoundError(f"{file_path} not found")

#     content = file_path.read_text(encoding='utf-8')

#     # Remove GO batch separators
#     content = re.sub(r'\bGO\b', '', content, flags=re.IGNORECASE)

#     # Capture all CREATE TABLE blocks
#     table_blocks = re.findall(
#         r'(CREATE TABLE\s+(?:\[[^\]]+\]|\'[^\']+\'|\w+)(?:\.(?:\[[^\]]+\]|\'[^\']+\'|\w+))?\s*\(.*?\))',
#         content,
#         flags=re.IGNORECASE | re.DOTALL
#     )

#     table_dict = {}

#     for block in table_blocks:
#         # Match and extract final part of table name (after optional schema)
#         match = re.search(
#             r'CREATE TABLE\s+(?:\[[^\]]+\]|\'[^\']+\'|\w+)\.'    # Optional schema (e.g., dbo.)
#             r'(?P<name>\[[^\]]+\]|\'[^\']+\'|\w+)'               # Actual table name
#             r'|CREATE TABLE\s+(?P<name2>\[[^\]]+\]|\'[^\']+\'|\w+)',  # If no schema
#             block,
#             flags=re.IGNORECASE
#         )
#         if match:
#             raw_name = match.group("name") or match.group("name2")
#             # Remove brackets or quotes if any
#             table_name = raw_name.strip("[]'\"")
#             table_dict[table_name] = block.strip()

#     return table_dict


def extract_create_table_blocks(sql: str) -> List[str]:
    """Extract full CREATE TABLE blocks with balanced parentheses."""
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
    """
    Parses a .txt file containing many CREATE TABLE statements and returns:
    {
        table_name: (full_create_sql, [list_of_column_names])
    }
    """
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found")

    content = file_path.read_text(encoding='utf-8')
    content = re.sub(r'\bGO\b', '', content, flags=re.IGNORECASE)

    table_blocks = extract_create_table_blocks(content)
    table_dict = {}

    for block in table_blocks:
        # Extract table name
        match = re.search(
            r'CREATE TABLE\s+(?:\[[^\]]+\]|\'[^\']+\'|\w+)(?:\.(?:\[[^\]]+\]|\'[^\']+\'|\w+))?'
            r'\s*\(',
            block,
            flags=re.IGNORECASE
        )
        if match:
            raw_name = re.search(
                r'CREATE TABLE\s+(?:\[(.*?)\]|\'(.*?)\'|(\w+))(?:\.(?:\[(.*?)\]|\'(.*?)\'|(\w+)))?',
                block
            )
            name_parts = raw_name.groups() if raw_name else ()
            table_name = next((x for x in name_parts if x), None)

            if not table_name:
                continue

            # Extract column names from inside the parentheses
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

def extract_table_names_from_sql_file(file_path: Path) -> list[str]:
    if not file_path.exists():
        return []

    content = file_path.read_text(encoding='utf-8')

    # Match CREATE TABLE [table_name]
    table_names = re.findall(r'CREATE TABLE\s+\[([^\]]+)\]', content)
    return table_names


custom_table_info = parse_custom_table_info_from_file(TABLE_INFO_FILE)
def get_db():
    global db
    if db is None:
        tables = extract_table_names_from_sql_file(TABLE_INFO_FILE)
        if not tables:
            tables = [
                "all_transactions", "AllTables", "expenseReport", "users", "project",
                "expenseItems", "itemDescription", "roles", "po", "attendance", "salary"
            ]
        db = SQLDatabase(
            engine,
            # include_tables=list(custom_table_info.keys()),
            # ignore_tables=['علاء سعيد 205$', 'PO Transactions$'], 
            custom_table_info=custom_table_info,
            sample_rows_in_table_info=0,
            max_string_length=100,
            lazy_table_reflection=False
        )
    return db

# def generate_slim_table_info(custom_table_info: dict) -> str:
#     slim_info = ""
#     for table_name, ddl in custom_table_info.items():
#         lines = ddl.splitlines()
#         columns = []
#         for line in lines:
#             if line.strip().startswith("[") and "]" in line:
#                 col = line.strip().split("]")[0].strip("[")
#                 columns.append(col)
#         slim_info += f"Table: {table_name}\nColumns: {', '.join(columns)}\n\n"
#     return slim_info.strip()

def generate_slim_table_info(custom_table_info: dict) -> str:
    slim_info = ""
    for table_name, (ddl, columns) in custom_table_info.items():
        # ddl is the full CREATE TABLE string
        # columns is the list of column names
        slim_info += f"Table: {table_name}\nColumns: {', '.join(columns)}\n\n"
    return slim_info.strip()


business_rules = read_business_rules()
# table_info = get_cached_table_info()
#improve
table_info = generate_slim_table_info(custom_table_info)

def get_table_suggestions(question: str,business_rules: str, tables_str: str):
    # chain for tables start


    # 1. Prepare the prompt templates.
    system_template = """You are a database assistant.  
    Use the business rules and available table information to select the top 5 SQL tables relevant to the user’s question."""  

    user_template = """
    Question: {question}

    Available tables:
    {tables}

    Business rules:
    {business_rules}

    List the most relevant table names for this query as a JSON array of strings (e.g., ["table1","table2"]).
    Always includes atleast 5 relevant tables.
    If you got any relevant business rules, table name will be given there, collect from there.
    If no business rules are available, just return the most 10 relevant tables based on the question and available tables.
    """

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])

    # 3. Create the LLMChain with our prompt.
    table_chain = LLMChain(llm=llm, prompt=chat_prompt)

    # 4. Format table candidates into a string.
    # table_candidates = [
    #     {"table": "po", "description": "purchase order transactions"},
    #     {"table": "salary", "description": "monthly salaries"}
    # ]
    # tables_str = "\n".join(f"{t['table']}: {t['description']}" for t in table_candidates)

    # 5. Invoke the chain to get relevant tables.
    response = table_chain.invoke({
        "question": question,
        "tables": tables_str,
        "business_rules": business_rules
    })
    selected_tables_json = response["text"].strip()
    # Remove triple backticks and any surrounding markdown
    cleaned = re.sub(r'^```json\n|```$', '', selected_tables_json.strip())


    # Parse JSON array of table names into a Python list.
    import json
    selected_tables = json.loads(cleaned)
    # chain for tables end
    return selected_tables

# current_db= get_db()
# sql_chain = create_sql_query_chain(
#     llm=llm,
#     db=current_db,
#     prompt=custom_prompt,
#     k=2
# )



@router.post("/reload-db")
def reload_db():
    table_info = get_cached_table_info(force_reload=True)
    return {"status_code": 200, "message": "Database reloaded", "table_info_preview": table_info[:500]}


@router.post("/update-rules")
async def update_business_rules():
    global business_rules
    business_rules = read_business_rules()
    return {"status": "success", "message": "Business rules updated"}

@router.get("/business-rules", response_class=PlainTextResponse)
def get_business_rules():
    if not MD_FILE_PATH.exists():
        return {"status_code": 404, "message": "Markdown file not found."}
    return MD_FILE_PATH.read_text(encoding="utf-8")

@router.put("/business-rules")
def update_business_rules(updated_content: str = Body(..., embed=True)):
    try:
        MD_FILE_PATH.write_text(updated_content, encoding="utf-8")
        return {"status_code": 200, "message": "Markdown file updated successfully."}
    except Exception as e:
        return {"status_code": 500, "message": f"Failed to update file: {e}"}

from memory_manager import ConversationMemory
memory_manager = ConversationMemory(max_messages=5)
from sqlalchemy import MetaData
@router.post("/query")
async def query_database(question: str, user_id: str = "default") -> dict:
    
    try:
        start_time = time.time()
        history = memory_manager.get_conversation_history(user_id)
        context = {
            "chat_history": "\n".join([
                f"Previous Question: {conv['question']}\n"
                f"Generated Query: {conv['query']}\n"
                for conv in history
            ])
        }

        tableaget_time = time.time()
        table_names_to_use = get_table_suggestions(
            question=question,
            business_rules=business_rules,
            tables_str=str(list(custom_table_info.keys()))
        )
        tableaget_time_end = time.time()
        print(f"Table suggestion took {tableaget_time_end - tableaget_time:.2f} seconds")


        
        # meta_data_time = time.time()
        # metadata = MetaData()
        # metadata.reflect(bind=engine, only=table_names_to_use)
        # meta_data_end_time = time.time()
        # print(f"Metadata reflection took {meta_data_end_time - meta_data_time:.2f} seconds")

        declare_time = time.time()
        mydb = SQLDatabase(
            engine,
            # include_tables=list(custom_table_info.keys()),
            # ignore_tables=['علاء سعيد 205$', 'PO Transactions$'], 
            # custom_table_info=custom_table_info,
            include_tables=table_names_to_use,
            sample_rows_in_table_info=0,
            max_string_length=100,
            lazy_table_reflection=False
        )
        declare_time_end = time.time()
        print(f"Database declaration took {declare_time_end - declare_time:.2f} seconds")

        chain_time = time.time()
        sql_chain = create_sql_query_chain(
            llm=llm,
            db=mydb,
            prompt=custom_prompt,
            k=2
        )
        chain_time_end = time.time()
        print(f"SQL Chain creation took {chain_time_end - chain_time:.2f} seconds")

        query_gen_time = time.time()
        sql = sql_chain.invoke({
            "question": question,
            "table_info": mydb.get_table_info(table_names=table_names_to_use),
            "top_k": 0,
            "business_rules": business_rules,
            "chat_history": context["chat_history"]
        })
        query_gen_end_time = time.time()
        print(f"Query generation took {query_gen_end_time - query_gen_time:.2f} seconds")

        # sql = sql_chain.invoke({
        #     "question": question,
        #     "table_info": current_db.table_info,
        #     "table_names_to_use":table_names_to_use,
        #     # "top_k": 2,
        #     "business_rules": business_rules,
        #     "chat_history": context["chat_history"]
        # })

        sql = " ".join(str(sql).replace("\t", " ").split())
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
        end_time = time.time()
        print(f"Query executed in {end_time - start_time:.2f} seconds")
        return {
            "status_code": 200,
            "payload": {
                "sql": sql,
                "data": data,
                "history": history
            }
        }
        
    except Exception as e:
        return {"status_code": 500, "payload": {"error": str(e)}}

@router.get("/conversation-history/{user_id}")
async def get_history(user_id: str):
    history = memory_manager.get_conversation_history(user_id)
    return {"status_code": 200, "payload": history}

@router.post("/clear-history/{user_id}")
async def clear_history(user_id: str):
    memory_manager.clear_conversation_history(user_id)
    return {"status_code": 200, "message": f"Conversation history cleared for user {user_id}"}