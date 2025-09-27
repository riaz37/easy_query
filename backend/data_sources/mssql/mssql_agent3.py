import os
import re
import sys
import time
import json
import asyncio
import aiohttp
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from fastapi import FastAPI, Body, APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain_groq import ChatGroq

# Add database manager import for direct access
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'db_manager'))
from mssql_config import DatabaseManager

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.getcwd())

# --- Environment Setup --- #
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Add configuration manager API URL
CONFIG_MANAGER_API_URL = os.getenv("BASE_URL", "https://localhost:8200")

# --- Database Manager Instance --- #
# Initialize database manager for direct access
try:
    db_manager = DatabaseManager(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
        default_db=os.getenv("DB_NAME", "postgres")
    )
except Exception as e:
    print(f"Warning: Could not initialize database manager: {e}")
    db_manager = None

# --- App & Router --- #
router = APIRouter()
app = FastAPI()

# --- Available Models Configuration --- #
AVAILABLE_MODELS = {
    "gemini": "gemini-2.0-flash",
    "llama-3.3-70b-versatile": "llama-3.3-70b-versatile", 
    "openai/gpt-oss-120b": "openai/gpt-oss-120b"
}

# --- Model Factory Function --- #
def create_llm_instance(model_name: str = "gemini"):
    """
    Create an LLM instance based on the specified model name.
    
    Args:
        model_name (str): The model to use. Defaults to "gemini" for backward compatibility.
        
    Returns:
        LLM instance (ChatGoogleGenerativeAI or ChatGroq)
        
    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unsupported model: {model_name}. Available models: {list(AVAILABLE_MODELS.keys())}")
    
    actual_model = AVAILABLE_MODELS[model_name]
    
    if model_name == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini model")
        return ChatGoogleGenerativeAI(
            model=actual_model,
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY
        )
    else:
        # All other models are Groq models
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is required for Groq models")
        return ChatGroq(
            model=actual_model,
            temperature=0.1,
            groq_api_key=GROQ_API_KEY
        )

# --- Default LLM Initialization (for backward compatibility) --- #
llm = create_llm_instance("gemini")

# --- Engine Setup (Per-User) --- #
# We will build engines dynamically per user (per db_id) using the configuration manager

# Caches
user_to_dbid_cache: Dict[str, Any] = {}
engine_cache_by_dbid: Dict[Any, Any] = {}

def convert_to_sqlalchemy_url(db_url: str) -> str:
    """
    Convert a database connection string to SQLAlchemy format compatible with pyodbc.
    Mirrors logic from `db_manager/utilites/semi_structured_To_table_db.py` at a high level.
    """
    if not db_url:
        raise ValueError("Empty database URL")

    # Already SQLAlchemy format
    if db_url.startswith('mssql+pyodbc://'):
        return db_url

    # ODBC style: Server=...;Database=...;User Id=...;Password=...
    if ';' in db_url and ('Server=' in db_url or 'Data Source=' in db_url):
        server_part = None
        if 'Server=' in db_url:
            server_part = db_url.split('Server=')[1].split(';')[0]
        elif 'Data Source=' in db_url:
            server_part = db_url.split('Data Source=')[1].split(';')[0]
        if not server_part:
            raise ValueError('Server/Data Source not found in connection string')

        if 'Database=' in db_url:
            database_part = db_url.split('Database=')[1].split(';')[0]
        elif 'Initial Catalog=' in db_url:
            database_part = db_url.split('Initial Catalog=')[1].split(';')[0]
        else:
            raise ValueError('Database/Initial Catalog not found in connection string')

        if 'User Id=' in db_url:
            username_part = db_url.split('User Id=')[1].split(';')[0]
        elif 'UID=' in db_url:
            username_part = db_url.split('UID=')[1].split(';')[0]
        else:
            raise ValueError('User Id/UID not found in connection string')

        if 'Password=' in db_url:
            password_part = db_url.split('Password=')[1].split(';')[0]
        elif 'PWD=' in db_url:
            password_part = db_url.split('PWD=')[1].split(';')[0]
        else:
            raise ValueError('Password/PWD not found in connection string')

        return (
            f"mssql+pyodbc://{username_part}:{password_part}@{server_part}/{database_part}"
            f"?driver=ODBC+Driver+18+for+SQL+Server&encrypt=no"
        )

    # SQLAlchemy-like but missing driver
    if db_url.startswith('mssql://'):
        sqlalchemy_url = db_url.replace('mssql://', 'mssql+pyodbc://')
        if '?' not in sqlalchemy_url:
            sqlalchemy_url += '?driver=ODBC+Driver+18+for+SQL+Server&encrypt=no'
        elif 'driver=' not in sqlalchemy_url:
            sqlalchemy_url += '&driver=ODBC+Driver+18+for+SQL+Server&encrypt=no'
        return sqlalchemy_url

    # Direct connection form user:pass@host:port/db
    if '@' in db_url and ':' in db_url:
        return f"mssql+pyodbc://{db_url}?driver=ODBC+Driver+18+for+SQL+Server&encrypt=no"

    # Fallback to as-is (may already be valid)
    return db_url

async def fetch_db_config(db_id: Any) -> Optional[Dict[str, Any]]:
    try:
        async with aiohttp.ClientSession() as session:
            # Some routes in the config manager are defined with 'mssql-config' inside the router
            # and the router is also mounted with prefix '/mssql-config'. This yields a final path of
            # '/mssql-config/mssql-config/{db_id}'. Try that first, then fall back to '/mssql-config/{db_id}'.
            primary = f"{CONFIG_MANAGER_API_URL}/mssql-config/mssql-config/{db_id}"
            secondary = f"{CONFIG_MANAGER_API_URL}/mssql-config/{db_id}"

            for url in (primary, secondary):
                async with session.get(url, ssl=False) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "success" and data.get("data"):
                            return data["data"]
                    elif response.status == 404:
                        continue
                    else:
                        print(f"Failed to fetch db config from {url}. Status: {response.status}")
    except Exception as e:
        print(f"Error fetching db config: {e}")
    return None

async def get_engine_for_user(user_id: str):
    """
    Returns a SQLAlchemy engine for the user's current db. Caches per db_id.
    """
    try:
        db_id = user_to_dbid_cache.get(user_id)
        if db_id is None:
            db_details = await fetch_user_current_db_details(user_id)
            if not db_details:
                print(f"DEBUG: No db details for user {user_id}")
                return None
            db_id = db_details.get('db_id')
            user_to_dbid_cache[user_id] = db_id

        if db_id in engine_cache_by_dbid:
            return engine_cache_by_dbid[db_id]

        # Fetch db config to get db_url
        db_config = await fetch_db_config(db_id)
        if not db_config:
            print(f"DEBUG: No db config for db_id {db_id}")
            return None

        db_url = db_config.get('db_url')
        if not db_url:
            print(f"DEBUG: db_url missing in config for db_id {db_id}")
            return None

        sqlalchemy_url = convert_to_sqlalchemy_url(db_url)
        eng = create_engine(
            sqlalchemy_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_pre_ping=True
        )
        # Optionally test the connection once
        try:
            with eng.connect() as _conn:
                pass
        except Exception as e:
            print(f"DEBUG: Engine connection test failed for user {user_id}, db_id {db_id}: {e}")
            return None

        engine_cache_by_dbid[db_id] = eng
        return eng
    except Exception as e:
        print(f"DEBUG: get_engine_for_user failed: {e}")
        return None

def clear_engine_for_user(user_id: str):
    db_id = user_to_dbid_cache.pop(user_id, None)
    if db_id is not None:
        eng = engine_cache_by_dbid.pop(db_id, None)
        if eng is not None:
            try:
                eng.dispose()
            except Exception:
                pass

# --- Database Data Storage --- #
database_business_rules = ""
database_table_info = {}
database_custom_table_info = {}
database_loaded = False  # Flag to track if database data has been loaded

# --- Function to parse SQL DDL to table structure --- #
def parse_sql_ddl_to_tables(ddl_text: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Parse SQL DDL text and extract table names and columns
    
    Args:
        ddl_text (str): Raw SQL DDL text containing CREATE TABLE statements
        
    Returns:
        Dict[str, Dict[str, List[str]]]: Dictionary with table names as keys and column info as values
    """
    tables = {}
    
    # Split by CREATE TABLE statements
    create_table_blocks = ddl_text.split('CREATE TABLE')
    
    for block in create_table_blocks[1:]:  # Skip first empty block
        try:
            # Extract table name (first word after CREATE TABLE)
            lines = block.strip().split('\n')
            first_line = lines[0].strip()
            
            # Find table name (everything before the first parenthesis)
            table_name = first_line.split('(')[0].strip()
            
            # Extract columns from the CREATE TABLE block
            columns = []
            in_columns_section = False
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('/*') or line.startswith('*/'):
                    continue
                
                # Check if we're in the columns section (between parentheses)
                if '(' in line and not in_columns_section:
                    in_columns_section = True
                    # Extract column from this line if it contains one
                    if ')' not in line:  # Column definition continues on next lines
                        continue
                
                if in_columns_section:
                    # Skip constraint lines (they start with CONSTRAINT, PRIMARY KEY, etc.)
                    if any(keyword in line.upper() for keyword in ['CONSTRAINT', 'PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CHECK']):
                        continue
                    
                    # Extract column name (first word before data type)
                    if line and not line.startswith(')') and not line.startswith(','):
                        # Split by whitespace and take first part as column name
                        parts = line.split()
                        if parts:
                            column_name = parts[0].strip()
                            # Remove brackets if present (for SQL Server identifiers)
                            column_name = column_name.strip('[]')
                            if column_name and column_name not in ['CONSTRAINT', 'PRIMARY', 'KEY', 'FOREIGN', 'UNIQUE', 'CHECK']:
                                columns.append(column_name)
                    
                    # Check if we've reached the end of the CREATE TABLE statement
                    if line.endswith(')'):
                        break
            
            if table_name and columns:
                tables[table_name] = {'columns': columns}
                print(f"DEBUG: Parsed table '{table_name}' with {len(columns)} columns: {columns}")
            
        except Exception as e:
            print(f"DEBUG: Error parsing table block: {e}")
            continue
    
    print(f"DEBUG: Successfully parsed {len(tables)} tables from DDL")
    return tables

# --- Function to fetch user's current database details --- #
async def fetch_user_current_db_details(user_id: str) -> Optional[Dict[str, any]]:
    """
    Fetch user's current database details from the configuration manager API
    
    Args:
        user_id (str): User ID to fetch database details for
        
    Returns:
        Optional[Dict[str, any]]: Database details or None if failed
    """
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{CONFIG_MANAGER_API_URL}/mssql-config/user-current-db/{user_id}"
            async with session.get(url,ssl=False) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "success" and data.get("data"):
                        return data["data"]
                    else:
                        print(f"API returned error: {data.get('message', 'Unknown error')}")
                        return None
                else:
                    print(f"Failed to fetch database details. Status: {response.status}")
                    return None
    except Exception as e:
        print(f"Error fetching database details: {e}")
        return None

async def fetch_merged_business_rules(user_id: str) -> str:
    """
    Fetch and merge business rules from both database config and user-specific business rules.
    Uses direct database manager access for better performance.
    
    Args:
        user_id (str): User ID to fetch business rules for
        
    Returns:
        str: Merged business rules (empty string if no rules found)
    """
    try:
        # Get user's current database details
        db_details = await fetch_user_current_db_details(user_id)
        if not db_details:
            print(f"DEBUG: No database details found for user {user_id}")
            return ""
        
        current_db_id = db_details.get('db_id')
        if not current_db_id:
            print(f"DEBUG: No database ID found for user {user_id}")
            return ""
        
        # Get database-level business rules
        db_business_rules = db_details.get('business_rule', '').strip()
        
        # Get user-specific business rules using direct database manager access
        user_business_rules = ""
        if db_manager:
            try:
                user_business_rules = db_manager.get_user_business_rule(user_id, current_db_id) or ""
                user_business_rules = user_business_rules.strip()
            except Exception as e:
                print(f"DEBUG: Error fetching user business rules: {e}")
        else:
            # Fallback to HTTP request if database manager not available
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"{CONFIG_MANAGER_API_URL}/new-table/user-business-rule/{user_id}"
                    async with session.get(url, ssl=False) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("status") == "success" and data.get("data"):
                                user_business_rules = data["data"].get("business_rule", "").strip()
            except Exception as e:
                print(f"DEBUG: Error fetching user business rules via HTTP: {e}")
        
        # Merge business rules
        merged_rules = []
        
        # Add database-level business rules if not empty
        if db_business_rules:
            merged_rules.append(f"# Database Business Rules\n{db_business_rules}")
        
        # Add user-specific business rules if not empty
        if user_business_rules:
            merged_rules.append(f"# User-Specific Business Rules\n{user_business_rules}")
        
        # Join with double newlines for clear separation
        final_rules = "\n\n".join(merged_rules)
        
        print(f"DEBUG: Merged business rules for user {user_id}:")
        print(f"  - Database rules length: {len(db_business_rules)}")
        print(f"  - User rules length: {len(user_business_rules)}")
        print(f"  - Final merged length: {len(final_rules)}")
        
        return final_rules
        
    except Exception as e:
        print(f"Error fetching merged business rules: {e}")
        return ""

async def load_database_data_async(user_id: str = "default"):
    """
    Async version of load_database_data for use in async contexts
    
    Args:
        user_id (str): User ID to load data for
    """
    global database_business_rules, database_table_info, database_custom_table_info, database_loaded
    
    try:
        db_details = await fetch_user_current_db_details(user_id)
        
        if db_details:
            # Fetch merged business rules from both sources
            database_business_rules = await fetch_merged_business_rules(user_id)
            
            # Extract table info
            table_info_data = db_details.get('table_info', {})
            if isinstance(table_info_data, dict):
                database_table_info = table_info_data
                # Convert table_info to custom_table_info format
                database_custom_table_info = {}

                # 1) Prefer structured JSON schema if available
                schema_content = table_info_data.get('schema')
                parsed_schema = None
                if isinstance(schema_content, str) and schema_content.strip():
                    try:
                        parsed_schema = json.loads(schema_content)
                        print("DEBUG: Parsed 'schema' content as JSON")
                    except json.JSONDecodeError as e:
                        print(f"DEBUG: Failed to parse 'schema' as JSON: {e}")
                elif isinstance(schema_content, dict):
                    parsed_schema = schema_content

                if parsed_schema and isinstance(parsed_schema, dict):
                    tables_list = parsed_schema.get('tables') or parsed_schema.get('Tables')
                    if isinstance(tables_list, list):
                        for table_obj in tables_list:
                            if not isinstance(table_obj, dict):
                                continue
                            schema_name = table_obj.get('schema') or table_obj.get('schema_name')
                            tbl_name = table_obj.get('table_name') or table_obj.get('name')
                            full_name = table_obj.get('full_name') or (f"{schema_name}.{tbl_name}" if schema_name and tbl_name else None)
                            cols = table_obj.get('columns') or []
                            # columns could be list[dict] or list[str]
                            if cols and isinstance(cols[0], dict):
                                columns = [c.get('name') for c in cols if isinstance(c, dict) and c.get('name')]
                            else:
                                columns = [c for c in cols if isinstance(c, str)]

                            if full_name:
                                database_custom_table_info[full_name] = (full_name, columns)
                            if tbl_name:
                                # Also add unqualified name for matching with LLM suggestions
                                database_custom_table_info[tbl_name] = (tbl_name, columns)

                    if database_custom_table_info:
                        print(f"DEBUG: Loaded {len(database_custom_table_info)} tables from 'schema' content")

                # 2) If no structured schema, fall back to generated_table_info
                if not database_custom_table_info:
                    generated_table_info = table_info_data.get('generated_table_info', {})
                    print(f"DEBUG: Generated table info type: {type(generated_table_info)}")
                    # If it's a string, try to parse as JSON, else naive text/DDL extraction
                    if isinstance(generated_table_info, str):
                        # Try JSON first
                        try:
                            generated_table_info_parsed = json.loads(generated_table_info)
                            generated_table_info = generated_table_info_parsed
                            print("DEBUG: Successfully parsed generated_table_info as JSON")
                        except json.JSONDecodeError:
                            # Try to parse as SQL DDL
                            print("DEBUG: Attempting to parse generated_table_info as SQL DDL/text")
                            ddl_tables = parse_sql_ddl_to_tables(generated_table_info)
                            if ddl_tables:
                                for table_name, table_data in ddl_tables.items():
                                    cols = table_data.get('columns', []) if isinstance(table_data, dict) else []
                                    database_custom_table_info[table_name] = (table_name, cols)
                            else:
                                # Naive fallback: parse "Table: X" and optional "Columns: ..."
                                try:
                                    parts = [p.strip() for p in generated_table_info.split('Table:') if p.strip()]
                                    for part in parts:
                                        first_line = part.splitlines()[0]
                                        tbl = first_line.split()[0].strip().strip('[]')
                                        cols_line = None
                                        for line in part.splitlines():
                                            if line.lower().startswith('columns'):
                                                cols_line = line
                                                break
                                        cols = []
                                        if cols_line and ':' in cols_line:
                                            cols_text = cols_line.split(':', 1)[1]
                                            cols = [c.strip().strip('[]') for c in cols_text.split(',') if c.strip()]
                                        database_custom_table_info[tbl] = (tbl, cols)
                                except Exception as e:
                                    print(f"DEBUG: Failed naive parsing of generated_table_info: {e}")
                    
                    if isinstance(generated_table_info, dict) and not database_custom_table_info:
                        print(f"DEBUG: Found generated_table_info with {len(generated_table_info)} tables (dict)")
                        for table_name, table_data in generated_table_info.items():
                            if isinstance(table_data, dict):
                                cols = table_data.get('columns', [])
                            else:
                                cols = []
                            database_custom_table_info[table_name] = (table_name, cols)
            
            database_loaded = True
            print(f"Successfully loaded database data for user: {user_id}")
            print(f"DEBUG: Final available tables: {list(database_custom_table_info.keys())}")
            return True
        else:
            print(f"Failed to load database data for user: {user_id}")
            return False
            
    except Exception as e:
        print(f"Error loading database data: {e}")
        return False

def load_database_data(user_id: str = "default"):
    """
    Load business rules and table info from database for a specific user
    (Synchronous wrapper for async function)
    
    Args:
        user_id (str): User ID to load data for
    """
    try:
        # Only try to load from database if we're not in a running event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in a running event loop, defer loading
                print("Deferring database loading to async context")
                return False
        except RuntimeError:
            pass
        
        # We can safely run async code here
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(load_database_data_async(user_id))
            loop.close()
            return result
        except Exception as e:
            print(f"Error in synchronous database loading: {e}")
            return False
            
    except Exception as e:
        print(f"Error loading database data: {e}")
        return False

# --- Initialize data --- #
# Start with empty data structures - data will be loaded on-demand from database
database_business_rules = ""
database_custom_table_info = {}
database_table_info = ""

from memory_manager import ConversationMemory
memory_manager = ConversationMemory(max_messages=5)

# --- Business Rules --- #
def read_business_rules():
    return database_business_rules

business_rules = read_business_rules()

# --- Table Info Generation --- #

def generate_slim_table_info(custom_table_info: dict) -> str:
    slim_info = ""
    for table_name, (ddl, columns) in custom_table_info.items():
        slim_info += f"Table: {table_name}\nColumns: {', '.join(columns)}\n\n"
    return slim_info.strip()

# Initialize with empty data - will be loaded from database on first use
custom_table_info = {}
table_info = ""


def get_rules_suggestions(question: str, business_rules: str, llm_instance=None):
    """
    Get business rules suggestions using the specified LLM instance.
    
    Args:
        question (str): User question
        business_rules (str): Available business rules
        llm_instance: LLM instance to use. If None, uses the default global llm.
    
    Returns:
        str: Selected business rules
    """
    if llm_instance is None:
        llm_instance = llm
        
    system_template = """You are a Context Extractor assistant. Use the business rules to select the most relevant Business Rules to the user's question."""
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
    rules_chain = LLMChain(llm=llm_instance, prompt=chat_prompt)
    response = rules_chain.invoke({
        "question": question,
        "business_rules": business_rules
    })
    selected_rules = response["text"].strip()
    # cleaned = re.sub(r'^```json\n|```$', '', selected_tables_json.strip())
    return selected_rules


# --- Table Suggestion --- #
def get_table_suggestions(question: str, business_rules: str, tables_str: str, llm_instance=None):
    """
    Get table suggestions using the specified LLM instance.
    
    Args:
        question (str): User question
        business_rules (str): Available business rules
        tables_str (str): Available tables as string
        llm_instance: LLM instance to use. If None, uses the default global llm.
    
    Returns:
        List[str]: List of suggested table names
    """
    if llm_instance is None:
        llm_instance = llm
        
    system_template = """You are a database assistant. Use the business rules and available table information to select the SQL tables most relevant to the user's question.
    - Select ONLY from the provided list of table names
    - The provided list contains schema-qualified names; always return and use names EXACTLY as provided
    - Do NOT invent or modify table names
    """
    user_template = """
    Question: {question}

    Available tables:
    {tables}

    Business rules:
    {business_rules}

    Return the most relevant table names for this query as a JSON array of strings, selecting ONLY from the Available tables list. Examples: ["dbo.Table1","hr.Employees"].
    If business rules specify tables, return only those (as listed in Available tables).
    If no business rules are available, return up to the 10 most relevant tables from the Available tables.
    If no tables are available, return an empty array: [].
    """
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])
    table_chain = LLMChain(llm=llm_instance, prompt=chat_prompt)
    response = table_chain.invoke({
        "question": question,
        "tables": tables_str,
        "business_rules": business_rules
    })
    selected_tables_json = response["text"].strip()
    cleaned = re.sub(r'^```json\n|```$', '', selected_tables_json.strip())
    
    # Handle empty or invalid responses
    if not cleaned or cleaned.strip() == "":
        return []
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"Error parsing table suggestions JSON: {e}")
        print(f"Raw response: {selected_tables_json}")
        print(f"Cleaned response: {cleaned}")
        # Return empty list as fallback
        return []

# --- Query Prompt --- #
query_prompt = PromptTemplate(
    input_variables=["question", "table_info", "business_rules", "chat_history"],
    template=(
        "You are an expert MSSQL developer.\n"
        "Available tables:\n{table_info}\n\n"
        "Business rules:\n{business_rules}\n\n"
        "Conversation context:\n{chat_history}\n\n"
        "Use ONLY the tables from Available tables. Always reference tables using their schema-qualified names exactly as shown (e.g., dbo.Table). Do NOT invent table names.\n\n"
        "You must use the sugeested tables from Business rules if mentioned.\n\n"
        "You must follow the Output format from business rules if mentioned.\n\n"
        "You must follow the conditions from business rules if mentioned.\n\n"
        "Use Like operator with '%' before and after where necessary for safe filtering. \n\n"
        "YOU MUST Use 'ISNULL(TRY_CAST(column AS INT), 0) AS column' instead of 'ISNULL(column, 0) AS column' in everywhere you need, to ensure the execution safety.\n\n"
        "MUST FOLLOW When writing UNION ALL queries, ensure that all fields (especially O_addedby and similar) have consistent data types across all SELECT blocks. If any column can contain non-numeric strings like 'N/A', use CAST(column AS NVARCHAR) for textual fields and ISNULL(TRY_CAST(column AS INT), 0) for integer fields to avoid conversion errors.\n\n"
        "if month or year nothing are given, take current month and current year (YEAR(GETDATE()) (i.e., current year from the system clock)) as month and year accordingly.\n\n"
        "if only month is given, take current year (YEAR(GETDATE()) (i.e., current year from the system clock)) as year. \n\n"
        "if month and year is given take both as is. \n\n"
        "Do NOT use T-SQL variables like @Param in the final query. Inline literals or use GETDATE()/DATEFROMPARTS/CONVERT etc.\n\n"
        "If the business rules mention input parameters (e.g., @AbsentDate) and the question doesn't provide them, default to using the current date with CONVERT(date, GETDATE()).\n\n"
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
async def query_database(question: str, user_id: str = "default", model: str = "gemini"):
    # Validate model parameter
    if model not in AVAILABLE_MODELS:
        return {
            "status_code": 400,
            "error": "Invalid model specified",
            "message": f"Model '{model}' is not supported. Available models: {list(AVAILABLE_MODELS.keys())}"
        }
    
    # Create LLM instance for this request
    try:
        request_llm = create_llm_instance(model)
    except ValueError as e:
        return {
            "status_code": 400,
            "error": "Model configuration error",
            "message": str(e)
        }
    
    # Fast-path: if the input looks like raw SQL, execute it directly (bypass LLM and config manager)
    try:
        raw_input = question.strip()
        if re.match(r"(?is)^\s*(SELECT|WITH|UPDATE|DELETE|INSERT|MERGE|EXEC|EXECUTE|DECLARE|CREATE|ALTER|DROP)\b", raw_input) or raw_input.startswith("```"):
            clean_sql_direct = re.sub(r"```(?:sql|text)?", "", raw_input).strip()
            print(f"DEBUG: Direct SQL execution path. SQL: {clean_sql_direct}")
            try:
                stmt_direct = text(clean_sql_direct)
                eng = await get_engine_for_user(user_id)
                if eng is None:
                    raise RuntimeError("No database engine available for user")
                with eng.connect() as conn:
                    result_direct = conn.execute(stmt_direct).fetchall()
                data_direct = [dict(row._mapping) for row in result_direct]
            except Exception as e:
                print(f"ERROR: Direct SQL execution failed: {e}")
                return {"status_code": 500, "payload": {"error": str(e), "sql": clean_sql_direct}}

            history_direct = memory_manager.get_conversation_history(user_id)
            memory_manager.add_conversation(
                user_id=user_id,
                question=question,
                query=clean_sql_direct,
                results=data_direct
            )
            print(f"DEBUG: Direct SQL execution using model: {model} for user: {user_id}")
            return {"status_code": 200, "payload": {"sql": clean_sql_direct, "data": data_direct, "history": history_direct, "model_used": model}}
    except Exception as _e:
        # If detection fails for any reason, fall back to normal flow
        print(f"DEBUG: Direct SQL detection failed: {_e}")

    # Load data for the specific user from database
    success = await load_database_data_async(user_id)
    
    if not success:
        return {
            "status_code": 404,
            "error": "User not found or no database configuration available",
            "message": f"User '{user_id}' does not have a valid database configuration. Please set up a current database for this user."
        }
    
    # Update global variables with fresh data
    global custom_table_info, table_info, business_rules
    custom_table_info = database_custom_table_info
    table_info = generate_slim_table_info(custom_table_info)
    business_rules = database_business_rules
    
    # Check if we have any table information
    if not custom_table_info:
        return {
            "status_code": 404,
            "error": "No table information available",
            "message": f"User '{user_id}' has no table information configured. Please set up table information in the database configuration."
        }
    
    history = memory_manager.get_conversation_history(user_id)
    context = {
        "chat_history": "\n".join([
            f"Previous Question: {conv['question']}\n"
            f"Generated Query: {conv['query']}\n"
            for conv in history
        ])
    }

    business_rules_to_use = get_rules_suggestions(question, business_rules, request_llm)
    print(f"DEBUG: Business rules suggestions: {business_rules_to_use}")
    
    # Only provide schema-qualified table names to reduce ambiguity
    full_table_names = [name for name in custom_table_info.keys() if "." in name]
    try:
        available_tables_str = json.dumps(full_table_names)
    except Exception:
        available_tables_str = str(full_table_names)
    print(f"DEBUG: Available tables (schema-qualified only): {available_tables_str}")
    
    table_names_to_use = get_table_suggestions(question, business_rules_to_use, available_tables_str, request_llm)
    print(f"DEBUG: Table suggestions: {table_names_to_use}")
    
    # Check if any tables were suggested
    if not table_names_to_use:
        return {
            "status_code": 404,
            "error": "No relevant tables found",
            "message": f"No relevant tables found for the question: '{question}'. Available tables: {available_tables_str}. Please check your database configuration and business rules."
        }
    
    # Filter to only include tables that actually exist
    available_tables = {k: custom_table_info[k] for k in table_names_to_use if k in custom_table_info}
    
    if not available_tables:
        return {
            "status_code": 404,
            "error": "No matching tables found",
            "message": f"Suggested tables {table_names_to_use} were not found in the available tables: {list(custom_table_info.keys())}. Please check your database configuration."
        }
    
    table_info_subset = generate_slim_table_info(available_tables)
    
    # Create dynamic query chain with the selected model
    dynamic_query_chain = LLMChain(llm=request_llm, prompt=query_prompt)
    response = dynamic_query_chain.invoke({
        "question": question,
        "table_info": table_info_subset,
        "business_rules": business_rules_to_use,
        "chat_history": context["chat_history"]
    })
    # sql = " ".join(str(response["text"]).replace("\t", " ").split())
    raw_sql = response["text"]
    print(f"DEBUG: Raw LLM SQL: {raw_sql}")
    # Remove markdown-style ```sql ... ```
    # clean_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
    clean_sql = re.sub(r"```(?:sql|text)?", "", raw_sql).strip()
    sql = " ".join(clean_sql.replace("\t", " ").split())

    # Runtime safeguard: replace known T-SQL variables with safe inline defaults
    try:
        # Replace @AbsentDate with today's date, case-insensitive
        sql = re.sub(r"@AbsentDate\b", "CONVERT(date, GETDATE())", sql, flags=re.IGNORECASE)
    except Exception as _e:
        pass

    # Auto-qualify unqualified table names when an unambiguous match exists
    try:
        def qualify_table_names(sql_text: str, table_info: dict) -> str:
            qualified_names = {name for name in table_info.keys() if "." in name}
            if not qualified_names:
                return sql_text

            # Build map from bare table -> set of qualified names
            bare_to_qualified = {}
            for qualified in qualified_names:
                bare = qualified.split(".")[-1].strip("[]")
                key = bare.lower()
                bare_to_qualified.setdefault(key, set()).add(qualified)

            def replace_match(m):
                keyword = m.group(1)
                table_token = m.group(2)
                # Already qualified
                if "." in table_token:
                    return f"{keyword} {table_token}"
                bare_key = table_token.strip("[]").lower()
                candidates = bare_to_qualified.get(bare_key)
                if candidates and len(candidates) == 1:
                    qualified = list(candidates)[0]
                    return f"{keyword} {qualified}"
                return f"{keyword} {table_token}"

            # FROM/JOIN clauses
            sql_updated = re.sub(r"\b(FROM|JOIN)\s+([A-Za-z_\[][^\s\]]*)", replace_match, sql_text, flags=re.IGNORECASE)
            # UPDATE/INTO/DELETE FROM clauses
            sql_updated = re.sub(r"\b(UPDATE|INTO|DELETE\s+FROM)\s+([A-Za-z_\[][^\s\]]*)", replace_match, sql_updated, flags=re.IGNORECASE)
            return sql_updated

        sql = qualify_table_names(sql, custom_table_info)
    except Exception as _e:
        print(f"DEBUG: qualify_table_names failed: {_e}")

    print(f"DEBUG: Final SQL to execute: {sql}")
    print(f"DEBUG: Using model: {model} for user: {user_id}")
    try:
        stmt = text(sql)
        eng = await get_engine_for_user(user_id)
        if eng is None:
            raise RuntimeError("No database engine available for user")
        with eng.connect() as conn:
            result = conn.execute(stmt).fetchall()
        data = [dict(row._mapping) for row in result]
    except Exception as e:
        print(f"ERROR: SQL execution failed: {e}")
        return {"status_code": 500, "payload": {"error": str(e), "sql": sql}}
    memory_manager.add_conversation(
            user_id=user_id,
            question=question,
            query=sql,
            results=data
        )
    return {"status_code": 200, "payload": {"sql": sql, "data": data, "history": history, "model_used": model}}

@router.post("/reload-db")
async def reload_db(user_id: str = "default"):
    global custom_table_info, table_info, business_rules, database_business_rules, database_custom_table_info, database_table_info
    
    # Clear cached engine for user so next query uses fresh config
    try:
        clear_engine_for_user(user_id)
    except Exception as _e:
        print(f"DEBUG: Failed to clear engine cache for user {user_id}: {_e}")

    # Try to reload from database
    success = await load_database_data_async(user_id)
    
    if not success:
        return {
            "status_code": 404,
            "error": "User not found or no database configuration available",
            "message": f"User '{user_id}' does not have a valid database configuration. Please set up a current database for this user."
        }
    
    # Use database data
    custom_table_info = database_custom_table_info
    table_info = generate_slim_table_info(custom_table_info)
    business_rules = database_business_rules
    
    return {
        "status_code": 200, 
        "message": f"reloaded from database for user: {user_id}", 
        "table_info_preview": table_info[:500],
        "source": "database"
    }

@router.get("/conversation-history/{user_id}")
async def get_history(user_id: str):
    history = memory_manager.get_conversation_history(user_id)
    return {"status_code": 200,"message":"History loaded successfully.", "payload": history}

@router.post("/clear-history/{user_id}")
async def clear_history(user_id: str):
    memory_manager.clear_conversation_history(user_id)
    return {"status_code": 200, "message": f"Conversation history cleared for user {user_id}"}

@router.get("/available-models")
async def get_available_models():
    """
    Get list of available models for query processing
    """
    return {
        "status_code": 200,
        "available_models": list(AVAILABLE_MODELS.keys()),
        "default_model": "gemini",
        "model_details": {
            model_name: {
                "provider": "groq" if model_name != "gemini" else "google",
                "model_id": actual_model
            }
            for model_name, actual_model in AVAILABLE_MODELS.items()
        }
    }

@router.get("/get_business-rules", response_class=PlainTextResponse)
async def get_business_rules(user_id: str = "default"):
    # Fetch merged business rules directly for better performance
    merged_rules = await fetch_merged_business_rules(user_id)
    
    if merged_rules is None:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}' does not have a valid database configuration. Please set up a current database for this user."
        )
    
    return merged_rules if merged_rules else ""

from fastapi.responses import FileResponse

from fastapi import UploadFile, File

@router.put("/update_business-rules")
async def update_business_rules(file: UploadFile = File(...), user_id: str = "default"):
    """
    Update business rules for a specific user (this would need to be implemented in the configuration manager)
    """
    return {
        "status": "error", 
        "message": "Business rules can only be updated through the configuration manager API. Use the configuration manager to update business rules for users."
    }

@router.get("/data-source")
async def get_data_source(user_id: str = "default"):
    """
    Get information about the current data source and status
    """
    try:
        # Get user's current database details
        db_details = await fetch_user_current_db_details(user_id)
        
        if not db_details:
            return {
                "status_code": 404,
                "data_source": "none",
                "user_id": user_id,
                "config_manager_url": CONFIG_MANAGER_API_URL,
                "error": f"User '{user_id}' does not have a valid database configuration",
                "has_business_rules": False,
                "has_table_info": False,
                "table_count": 0,
                "business_rules_length": 0,
                "business_rules_sources": {
                    "database_rules": False,
                    "user_rules": False,
                    "merged_rules": False
                }
            }
        
        # Get merged business rules for detailed breakdown
        merged_rules = await fetch_merged_business_rules(user_id)
        
        # Extract individual rule sources for analysis
        db_rules = db_details.get('business_rule', '').strip()
        current_db_id = db_details.get('db_id')
        
        user_rules = ""
        if db_manager and current_db_id:
            try:
                user_rules = db_manager.get_user_business_rule(user_id, current_db_id) or ""
                user_rules = user_rules.strip()
            except Exception:
                pass
        
        # Load table info
        table_info_data = db_details.get('table_info', {})
        custom_table_info = {}
        
        if isinstance(table_info_data, dict):
            # Process table info (simplified version of the full logic)
            schema_content = table_info_data.get('schema')
            if isinstance(schema_content, str) and schema_content.strip():
                try:
                    parsed_schema = json.loads(schema_content)
                    tables_list = parsed_schema.get('tables') or parsed_schema.get('Tables')
                    if isinstance(tables_list, list):
                        for table_obj in tables_list:
                            if isinstance(table_obj, dict):
                                schema_name = table_obj.get('schema') or table_obj.get('schema_name')
                                tbl_name = table_obj.get('table_name') or table_obj.get('name')
                                full_name = table_obj.get('full_name') or (f"{schema_name}.{tbl_name}" if schema_name and tbl_name else None)
                                cols = table_obj.get('columns') or []
                                if cols and isinstance(cols[0], dict):
                                    columns = [c.get('name') for c in cols if isinstance(c, dict) and c.get('name')]
                                else:
                                    columns = [c for c in cols if isinstance(c, str)]
                                if full_name:
                                    custom_table_info[full_name] = (full_name, columns)
                except Exception:
                    pass
        
        return {
            "status_code": 200,
            "data_source": "database",
            "user_id": user_id,
            "config_manager_url": CONFIG_MANAGER_API_URL,
            "has_business_rules": bool(merged_rules),
            "has_table_info": bool(custom_table_info),
            "table_count": len(custom_table_info) if custom_table_info else 0,
            "business_rules_length": len(merged_rules) if merged_rules else 0,
            "business_rules_sources": {
                "database_rules": bool(db_rules),
                "user_rules": bool(user_rules),
                "merged_rules": bool(merged_rules),
                "database_rules_length": len(db_rules),
                "user_rules_length": len(user_rules),
                "merged_rules_length": len(merged_rules)
            }
        }
    except Exception as e:
        return {
            "status_code": 500,
            "error": str(e),
            "data_source": "error"
        }

@router.post("/switch-user")
async def switch_user(user_id: str):
    """
    Switch to a different user and reload their database configuration
    """
    try:
        # Clear cached engine for previous user id; ensure fresh engine on next use
        try:
            clear_engine_for_user(user_id)
        except Exception as _e:
            print(f"DEBUG: Failed to clear engine cache on switch for {user_id}: {_e}")

        success = await load_database_data_async(user_id)
        
        if not success:
            return {
                "status_code": 404,
                "error": "User not found or no database configuration available",
                "message": f"User '{user_id}' does not have a valid database configuration. Please set up a current database for this user."
            }
        
        # Update global variables
        global custom_table_info, table_info, business_rules
        custom_table_info = database_custom_table_info
        table_info = generate_slim_table_info(custom_table_info)
        business_rules = database_business_rules
        
        return {
            "status_code": 200,
            "message": f"Switched to user: {user_id}",
            "success": True,
            "data_source": "database"
        }
    except Exception as e:
        return {
            "status_code": 500,
            "error": str(e),
            "message": f"Failed to switch to user: {user_id}"
        }


# --- Mount router --- #
app.include_router(router)
