import requests
import json
import logging
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def get_matched_tables_json(user_id: str, db_id: int) -> Dict[str, Any]:
    """
    Extract and match table names from business rules and database schema.
    
    Args:
        user_id (str): The user ID
        db_id (int): The database ID
        
    Returns:
        Dict[str, Any]: JSON structure with matched table names and complete metadata
    """
    try:
        # Get base URL from environment
        base_url = os.getenv('BASE_URL', 'http://127.0.0.1:8200').rstrip('/')
        
        # Step 1: Validate user access
        if not _validate_user_access(base_url, user_id, db_id):
            return _create_error_response(f"User {user_id} does not have access to database {db_id}")
        print("i am taking time 1")
        # Step 2: Fetch database configuration
        url = f"{base_url}/mssql-config/mssql-config/{db_id}"
        response = requests.get(url)
        response.raise_for_status()
        print("i am taking time 2")
        data = response.json()
        if data.get("status") != "success":
            return _create_error_response(f"API returned error: {data.get('message')}")
        
        database_config = data.get("data")
        if not database_config:
            return _create_error_response("No database config found")
        
        # Step 3: Extract table names from business rules
        business_rules = database_config.get("business_rule", "")
        business_rules_tables = []
        
        if business_rules:
            business_rules_tables = _extract_with_ai(business_rules)
            logger.info(f"Extracted {len(business_rules_tables)} tables from business rules")
        else:
            logger.warning("No business rules found")
        
        # Step 4: Extract table names and schema content
        schema_tables = _extract_from_schema(database_config)
        schema_content = _get_schema_content(database_config)
        logger.info(f"Extracted {len(schema_tables)} tables from schema")
        
        # Step 5: Find matches and get detailed schema info
        matched_tables, matched_tables_details = _find_matches_with_details(
            business_rules_tables, schema_tables, schema_content
        )
        
        # Step 6: Create result JSON
        result = {
            "status": "success",
            "message": f"Successfully processed database {db_id} for user {user_id}",
            "metadata": {
                "user_id": user_id,
                "db_id": db_id,
                "db_name": database_config.get("db_name", ""),
                "generated_at": datetime.now().isoformat(),
                "total_business_rules_tables": len(business_rules_tables),
                "total_schema_tables": len(schema_tables),
                "total_matches": len(matched_tables)
            },
            "matched_tables": sorted(matched_tables),
            "matched_tables_details": matched_tables_details,
            "business_rules_tables": sorted(business_rules_tables),
            "schema_tables": sorted(schema_tables),
            "unmatched_business_rules": _get_unmatched_tables(business_rules_tables, matched_tables),
            "unmatched_schema": _get_unmatched_tables(schema_tables, matched_tables)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating matched tables JSON: {e}")
        return _create_error_response(f"Error generating matched tables JSON: {e}")


def _validate_user_access(base_url: str, user_id: str, db_id: int) -> bool:
    """Validate if user has access to the database."""
    try:
        url = f"{base_url}/mssql-config/user-access/{user_id}"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        if data.get("status") != "success":
            logger.error(f"User access API returned error: {data.get('message')}")
            return False
        
        access_data = data.get("data", {})
        access_configs = access_data.get("access_configs", [])
        
        if not access_configs:
            logger.warning(f"No access configurations found for user {user_id}")
            return False
        
        # Check if user has access to the specific database
        for config in access_configs:
            database_access = config.get("database_access", {})
            
            # Check parent databases
            parent_databases = database_access.get("parent_databases", [])
            for db in parent_databases:
                if db.get("db_id") == db_id:
                    logger.info(f"User {user_id} has access to database {db_id} (parent)")
                    return True
            
            # Check sub databases
            sub_databases = database_access.get("sub_databases", [])
            for sub_db_group in sub_databases:
                databases = sub_db_group.get("databases", [])
                for db in databases:
                    if db.get("db_id") == db_id:
                        logger.info(f"User {user_id} has access to database {db_id} (sub)")
                        return True
        
        logger.warning(f"User {user_id} does not have access to database {db_id}")
        return False
        
    except Exception as e:
        logger.error(f"Error during user validation: {e}")
        return False


def _extract_with_ai(business_rules: str) -> List[str]:
    """Extract table names using Gemini AI."""
    try:
        gemini_apikey = os.getenv("google_api_key")
        if not gemini_apikey:
            logger.warning("No Google API key found")
            return []
        
        gemini_model_name = os.getenv("google_gemini_name", "gemini-1.5-pro")
        llm = ChatGoogleGenerativeAI(
            model=gemini_model_name, 
            temperature=0,
            google_api_key=gemini_apikey
        )
        
        prompt = f"""
Extract ONLY the table names from the following business rules text.

Business Rules:
{business_rules}

Instructions:
1. Look for any mention of database tables, table names, or entities
2. Extract ONLY the table names
3. Return ONLY a Python list format like: ["table1", "table2", "table3"]
4. If no table names found, return empty list: []
5. Use lowercase table names
6. Remove schema prefixes (like "dbo.")

Example output: ["users", "products", "orders"]

Extract table names:
"""
        
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        return _parse_ai_response(content)
        
    except Exception as e:
        logger.error(f"AI extraction failed: {e}")
        return []


def _parse_ai_response(response: str) -> List[str]:
    """Parse AI response to extract table names list."""
    try:
        response = response.strip()
        
        # Remove markdown code blocks
        if '```python' in response:
            response = response.split('```python')[1]
        if '```' in response:
            response = response.split('```')[0]
        response = response.replace('```', '').strip()
        
        # Try to parse as Python list
        if response.startswith('[') and response.endswith(']'):
            import ast
            try:
                table_names = ast.literal_eval(response)
                if isinstance(table_names, list):
                    return [str(name).strip().lower() for name in table_names if name and str(name).strip()]
            except ValueError:
                pass
        
        # Extract content between brackets
        if '[' in response and ']' in response:
            start = response.find('[')
            end = response.find(']')
            if start != -1 and end != -1:
                content = response[start+1:end]
                parts = content.split(',')
                table_names = []
                for part in parts:
                    part = part.strip().strip('"\'')
                    if part and part.lower() not in ['none', 'null', '']:
                        table_names.append(part.lower())
                return table_names
        
        return []
        
    except Exception as e:
        logger.error(f"Error parsing AI response: {e}")
        return []


def _extract_from_schema(database_config: dict) -> List[str]:
    """Extract table names from schema."""
    try:
        table_info = database_config.get("table_info", {})
        if isinstance(table_info, str):
            table_info = json.loads(table_info)
        
        schema_content = table_info.get("schema")
        if isinstance(schema_content, str):
            schema_content = json.loads(schema_content)
        
        if not schema_content or not isinstance(schema_content, dict):
            return []
        
        tables = schema_content.get("tables", [])
        table_names = []
        
        for table in tables:
            if isinstance(table, dict):
                table_name = table.get("table_name", "")
                if table_name:
                    table_names.append(table_name.lower())
        
        return table_names
        
    except Exception as e:
        logger.error(f"Error extracting from schema: {e}")
        return []


def _get_schema_content(database_config: dict) -> dict:
    """Extract and parse schema content from database config."""
    try:
        table_info = database_config.get("table_info", {})
        if isinstance(table_info, str):
            table_info = json.loads(table_info)
        
        schema_content = table_info.get("schema")
        if isinstance(schema_content, str):
            schema_content = json.loads(schema_content)
        
        if not schema_content or not isinstance(schema_content, dict):
            return {}
        
        return schema_content
        
    except Exception as e:
        logger.error(f"Error extracting schema content: {e}")
        return {}


def _find_matches_with_details(business_rules_tables: List[str], schema_tables: List[str], schema_content: dict) -> tuple:
    """Find matches between business rules and schema tables, and get detailed info."""
    # Convert to lowercase for case-insensitive comparison
    business_rules_lower = [table.lower() for table in business_rules_tables]
    schema_lower = [table.lower() for table in schema_tables]
    
    # Find intersection
    matched_tables_lower = set(business_rules_lower) & set(schema_lower)
    
    # Convert back to original case from schema and get details
    matched_tables = []
    matched_tables_details = []
    
    for table_lower in matched_tables_lower:
        # Find original case from schema tables
        for schema_table in schema_tables:
            if schema_table.lower() == table_lower:
                matched_tables.append(schema_table)
                
                # Get detailed schema information
                table_details = _get_table_details_from_schema(schema_content, schema_table)
                if table_details:
                    matched_tables_details.append(table_details)
                break
    
    return matched_tables, matched_tables_details


def _get_table_details_from_schema(schema_content: dict, table_name: str) -> dict:
    """Get detailed information for a specific table from schema content."""
    try:
        if not schema_content or not isinstance(schema_content, dict):
            return {}
        
        tables = schema_content.get("tables", [])
        
        for table in tables:
            if isinstance(table, dict) and table.get("table_name", "").lower() == table_name.lower():
                return {
                    "schema": table.get("schema", ""),
                    "table_name": table.get("table_name", ""),
                    "full_name": table.get("full_name", ""),
                    "primary_keys": table.get("primary_keys", []),
                    "columns": table.get("columns", []),
                    "relationships": table.get("relationships", []),
                    # Note: sample_data excluded from response as requested
                    "row_count_sample": table.get("row_count_sample", 0)
                }
        
        return {}
        
    except Exception as e:
        logger.error(f"Error getting table details for {table_name}: {e}")
        return {}


def _get_unmatched_tables(all_tables: List[str], matched_tables: List[str]) -> List[str]:
    """Get tables that are not in the matched list."""
    matched_lower = [table.lower() for table in matched_tables]
    return sorted([table for table in all_tables if table.lower() not in matched_lower])


def _create_error_response(message: str) -> Dict[str, Any]:
    """Create a standardized error response."""
    return {
        "status": "error",
        "message": message,
        "matched_tables": [],
        "matched_tables_details": [],
        "business_rules_tables": [],
        "schema_tables": [],
        "metadata": {
            "total_matches": 0,
            "generated_at": datetime.now().isoformat()
        }
    }


# Example usage
if __name__ == "__main__":
    user_id = "nilab"
    db_id = 1
    
    print("🔍 Getting matched tables JSON...")
    result = get_matched_tables_json(user_id, db_id)
    
    if result["status"] == "success":
        print(f"✅ Success! Found {result['metadata']['total_matches']} matched tables")
        print(json.dumps(result, indent=2))
    else:
        print(f"❌ Error: {result['message']}")