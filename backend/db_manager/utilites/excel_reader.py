#!/usr/bin/env python3
"""
Excel Reader with Gemini Column Mapping
Reads Excel files and generates mappings between Excel columns and database table columns using Gemini AI.
"""

import os
import sys
import json
import pandas as pd
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import google.generativeai as genai
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from decimal import Decimal
from datetime import datetime, date, time as dt_time

# Add the path to import the database manager
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mssql_config import db_manager

# Import column details finder functionality
from column_details_finder import get_column_names_with_details

# Load environment variables
load_dotenv()

# Custom JSON encoder for SQL data types
class SQLJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle SQL Server data types"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, dt_time):
            return obj.isoformat()
        elif isinstance(obj, (bytes, bytearray)):
            try:
                return obj.decode('utf-8')
            except UnicodeDecodeError:
                return obj.decode('utf-8', errors='ignore')
        elif hasattr(obj, 'isoformat'):  # Any other datetime-like objects
            return obj.isoformat()
        return super().default(obj)

def convert_to_sqlalchemy_url(db_url: str) -> str:
    """
    Convert a database connection string to SQLAlchemy format.
    
    Args:
        db_url (str): Database connection string (e.g., from mssql_config)
        
    Returns:
        str: SQLAlchemy compatible connection URL
    """
    try:
        print(f"🔄 Converting database URL to SQLAlchemy format...")
        print(f"   Input URL: {db_url}")
        
        # If it's already in SQLAlchemy format, return as is
        if db_url.startswith('mssql+pyodbc://'):
            print("   ✅ Already in SQLAlchemy format")
            return db_url
        
        # Handle different connection string formats
        # Format 1: ODBC connection string "Server=host,port;Database=dbname;User Id=username;Password=password;"
        # Format 2: SQLAlchemy-like but without driver "mssql://username:password@host:port/database"
        # Format 3: Direct connection string
        
        # Try to parse as ODBC connection string first
        if ';' in db_url and ('Server=' in db_url or 'Data Source=' in db_url):
            print("   📋 Parsing as ODBC connection string...")
            
            # Extract server/host
            if 'Server=' in db_url:
                server_part = db_url.split('Server=')[1].split(';')[0]
            elif 'Data Source=' in db_url:
                server_part = db_url.split('Data Source=')[1].split(';')[0]
            else:
                raise ValueError("Could not find Server or Data Source in connection string")
            
            # Extract database name
            if 'Database=' in db_url:
                database_part = db_url.split('Database=')[1].split(';')[0]
            elif 'Initial Catalog=' in db_url:
                database_part = db_url.split('Initial Catalog=')[1].split(';')[0]
            else:
                raise ValueError("Could not find Database or Initial Catalog in connection string")
            
            # Extract username
            if 'User Id=' in db_url:
                username_part = db_url.split('User Id=')[1].split(';')[0]
            elif 'UID=' in db_url:
                username_part = db_url.split('UID=')[1].split(';')[0]
            else:
                raise ValueError("Could not find User Id or UID in connection string")
            
            # Extract password
            if 'Password=' in db_url:
                password_part = db_url.split('Password=')[1].split(';')[0]
            elif 'PWD=' in db_url:
                password_part = db_url.split('PWD=')[1].split(';')[0]
            else:
                raise ValueError("Could not find Password or PWD in connection string")
            
            # Build SQLAlchemy URL
            sqlalchemy_url = f"mssql+pyodbc://{username_part}:{password_part}@{server_part}/{database_part}?driver=ODBC+Driver+18+for+SQL+Server&encrypt=no"
            
        # Try to parse as SQLAlchemy-like URL
        elif db_url.startswith('mssql://'):
            print("   📋 Parsing as SQLAlchemy-like URL...")
            # Remove mssql:// prefix and add pyodbc driver
            sqlalchemy_url = db_url.replace('mssql://', 'mssql+pyodbc://')
            if '?' not in sqlalchemy_url:
                sqlalchemy_url += '?driver=ODBC+Driver+18+for+SQL+Server&encrypt=no'
            elif 'driver=' not in sqlalchemy_url:
                sqlalchemy_url += '&driver=ODBC+Driver+18+for+SQL+Server&encrypt=no'
        
        # Try to parse as direct connection string (like schema_generator format)
        elif '@' in db_url and ':' in db_url:
            print("   📋 Parsing as direct connection string...")
            # Expected format: "username:password@host:port/database"
            if db_url.startswith('mssql+pyodbc://'):
                sqlalchemy_url = db_url
            else:
                sqlalchemy_url = f"mssql+pyodbc://{db_url}?driver=ODBC+Driver+18+for+SQL+Server&encrypt=no"
        
        else:
            raise ValueError(f"Unsupported connection string format: {db_url}")
        
        print(f"   ✅ Converted URL: {sqlalchemy_url}")
        return sqlalchemy_url
        
    except Exception as e:
        print(f"❌ Error converting database URL: {e}")
        print(f"Original URL: {db_url}")
        raise ValueError(f"Invalid database connection string format: {e}")

class ExcelReaderWithMapping:
    """
    Excel reader that generates mappings between Excel columns and database table columns using Gemini AI.
    """
    
    def __init__(self):
        """Initialize the Excel reader with Gemini AI configuration."""
        # Initialize Gemini AI
        self.gemini_api_key = os.getenv("google_api_key")
        if not self.gemini_api_key:
            raise ValueError("Google API key not found in environment variables. Please set 'google_api_key'.")
        
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        print("✅ Excel Reader with Gemini AI initialized successfully")
    
    def generate_specific_table_info(self, user_id: str, table_full_name: str) -> Optional[Dict[str, Any]]:
        """
        Generate detailed table information including CREATE TABLE statement for a specific table.
        
        Args:
            user_id (str): User ID
            table_full_name (str): Full table name (e.g., "dbo.expenseItems")
            
        Returns:
            Optional[Dict[str, Any]]: Detailed table information if found, None otherwise
        """
        try:
            print(f"🔍 Generating detailed table info for: {table_full_name}")
            
            # Step 1: Get user's current database details
            user_data = db_manager.get_user_current_db_details(user_id)
            if not user_data:
                print("❌ No user current database data found")
                return None
            
            # Step 2: Get database ID from user data
            db_id = user_data.get('db_id')
            if not db_id:
                print("❌ No database ID found in user's current database")
                return None
            
            # Step 3: Get database configuration
            db_config = db_manager.get_mssql_config(db_id)
            if not db_config:
                print(f"❌ Database configuration not found for db_id: {db_id}")
                return None
            
            # Step 4: Extract database URL
            db_url = db_config.get('db_url')
            if not db_url:
                print("❌ Database connection URL not found in configuration")
                return None
            
            print(f"✅ Retrieved database URL: {db_url[:50]}...")
            
            # Parse table name
            if '.' in table_full_name:
                schema_name, table_name = table_full_name.split('.', 1)
            else:
                schema_name = 'dbo'
                table_name = table_full_name
            
            # Convert database URL to SQLAlchemy format
            try:
                sqlalchemy_url = convert_to_sqlalchemy_url(db_url)
                print("✅ Database URL converted successfully")
            except Exception as e:
                print(f"⚠️  URL conversion failed: {e}")
                print("🔄 Trying fallback approach...")
                
                # Fallback: Use the exact same format as schema_generator
                try:
                    if not db_url.startswith('mssql'):
                        sqlalchemy_url = f"mssql+pyodbc://{db_url}?driver=ODBC+Driver+18+for+SQL+Server&encrypt=no"
                    else:
                        sqlalchemy_url = db_url
                    print(f"🔄 Fallback URL: {sqlalchemy_url}")
                except Exception as fallback_error:
                    print(f"❌ Fallback also failed: {fallback_error}")
                    return None
            
            # Connect to database
            print("🔌 Connecting to database...")
            try:
                engine = create_engine(sqlalchemy_url)
                conn = engine.connect()
                print("✅ Database connection successful!")
            except SQLAlchemyError as e:
                print(f"❌ Failed to connect to database: {e}")
                return None
            
            try:
                # Get basic column information first
                columns_sql = text("""
                    SELECT 
                        COLUMN_NAME,
                        DATA_TYPE,
                        IS_NULLABLE,
                        CHARACTER_MAXIMUM_LENGTH,
                        NUMERIC_PRECISION,
                        NUMERIC_SCALE,
                        COLUMN_DEFAULT
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
                    ORDER BY ORDINAL_POSITION
                """)
                
                result = conn.execute(columns_sql, {
                    "schema": schema_name,
                    "table": table_name
                })
                
                columns = []
                column_names = []
                
                for row in result.fetchall():
                    col_name, data_type, is_nullable, max_length, precision, scale, default_val = row
                    column_names.append(col_name)
                    
                    # Build column definition
                    col_def = f"[{col_name}] {data_type.upper()}"
                    
                    # Add precision and scale for numeric types
                    if precision and scale:
                        col_def += f"({precision},{scale})"
                    elif max_length and data_type.upper() in ['VARCHAR', 'NVARCHAR', 'CHAR', 'NCHAR']:
                        col_def += f"({max_length})"
                    
                    # Add collation for string types
                    if data_type.upper() in ['VARCHAR', 'NVARCHAR', 'CHAR', 'NCHAR', 'TEXT', 'NTEXT']:
                        col_def += " COLLATE SQL_Latin1_General_CP1_CI_AS"
                    
                    # Add nullable constraint
                    if is_nullable == "NO":
                        col_def += " NOT NULL"
                    else:
                        col_def += " NULL"
                    
                    # Add default value
                    if default_val:
                        col_def += f" DEFAULT {default_val}"
                    
                    columns.append(col_def)
                
                # Get identity columns using sys.columns
                identity_sql = text("""
                    SELECT c.name
                    FROM sys.columns c
                    INNER JOIN sys.tables t ON c.object_id = t.object_id
                    INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
                    WHERE s.name = :schema AND t.name = :table AND c.is_identity = 1
                """)
                
                identity_result = conn.execute(identity_sql, {
                    "schema": schema_name,
                    "table": table_name
                })
                identity_columns = [row[0] for row in identity_result.fetchall()]
                
                # Get primary key columns
                pk_sql = text("""
                    SELECT c.name
                    FROM sys.indexes i
                    INNER JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
                    INNER JOIN sys.columns c ON c.object_id = ic.object_id AND c.column_id = ic.column_id
                    WHERE i.is_primary_key = 1 AND i.object_id = OBJECT_ID(:full_name)
                    ORDER BY ic.key_ordinal
                """)
                
                pk_result = conn.execute(pk_sql, {"full_name": table_full_name})
                primary_keys = [row[0] for row in pk_result.fetchall()]
                
                # Get foreign key information
                fk_sql = text("""
                    SELECT 
                        fk.name as constraint_name,
                        parentCol.name as fk_column,
                        schemaRef.name as referenced_schema,
                        tabRef.name as referenced_table,
                        refCol.name as referenced_column
                    FROM sys.foreign_keys fk
                    INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
                    INNER JOIN sys.tables tabParent ON fk.parent_object_id = tabParent.object_id
                    INNER JOIN sys.schemas schemaParent ON tabParent.schema_id = schemaParent.schema_id
                    INNER JOIN sys.columns parentCol ON fkc.parent_object_id = parentCol.object_id AND fkc.parent_column_id = parentCol.column_id
                    INNER JOIN sys.tables tabRef ON fk.referenced_object_id = tabRef.object_id
                    INNER JOIN sys.schemas schemaRef ON tabRef.schema_id = schemaRef.schema_id
                    INNER JOIN sys.columns refCol ON fkc.referenced_object_id = refCol.object_id AND fkc.referenced_column_id = refCol.column_id
                    WHERE schemaParent.name = :schema AND tabParent.name = :table
                """)
                
                fk_result = conn.execute(fk_sql, {
                    "schema": schema_name,
                    "table": table_name
                })
                foreign_keys = []
                for row in fk_result.fetchall():
                    constraint_name, fk_column, ref_schema, ref_table, ref_column = row
                    foreign_keys.append(f"{fk_column} FOREIGN KEY REFERENCES {ref_schema}.{ref_table}({ref_column})")
                
                # Add identity information to column definitions
                for i, col_name in enumerate(column_names):
                    if col_name in identity_columns:
                        # Find the column definition and add IDENTITY
                        for j, col_def in enumerate(columns):
                            if col_def.startswith(f"[{col_name}]"):
                                columns[j] = col_def + " IDENTITY(1,1)"
                                break
                
                # Build CREATE TABLE statement
                create_table_sql = f"CREATE TABLE [{table_name}] (\n"
                create_table_sql += ",\n".join(f"\t{col}" for col in columns)
                
                # Add constraints
                if primary_keys:
                    pk_constraint = f",\n\tCONSTRAINT [PK_{table_name}] PRIMARY KEY CLUSTERED ([{'], ['.join(primary_keys)}])"
                    create_table_sql += pk_constraint
                
                if foreign_keys:
                    for fk in foreign_keys:
                        create_table_sql += f",\n\tCONSTRAINT [FK_{table_name}_{fk.split()[0]}] FOREIGN KEY([{fk.split()[0]}]) REFERENCES {fk.split('REFERENCES ')[1]}"
                
                create_table_sql += "\n)"
                
                # Get sample data
                sample_sql = text(f"SELECT TOP 3 * FROM {table_full_name}")
                sample_result = conn.execute(sample_sql)
                sample_rows = []
                for row in sample_result.fetchall():
                    sample_rows.append(dict(row._mapping))
                
                table_info = {
                    "table_name": table_name,
                    "schema_name": schema_name,
                    "full_name": table_full_name,
                    "create_table_statement": create_table_sql,
                    "columns": columns,
                    "identity_columns": identity_columns,
                    "primary_keys": primary_keys,
                    "foreign_keys": foreign_keys,
                    "sample_data": sample_rows,
                    "total_columns": len(columns)
                }
                
                print(f"✅ Generated detailed table info for {table_full_name}")
                print(f"   - Total columns: {len(columns)}")
                print(f"   - Identity columns: {len(identity_columns)}")
                print(f"   - Primary keys: {len(primary_keys)}")
                print(f"   - Foreign keys: {len(foreign_keys)}")
                
                return table_info
                
            finally:
                conn.close()
                engine.dispose()
                
        except Exception as e:
            print(f"❌ Error generating table info: {e}")
            return None

    def read_excel_preview(self, excel_file_path: str, num_rows: int = 5) -> Dict[str, Any]:
        """
        Read Excel file and return preview data with column names.
        
        Args:
            excel_file_path (str): Path to the Excel file
            num_rows (int): Number of rows to preview (default: 5)
            
        Returns:
            Dict[str, Any]: Dictionary containing column names and preview data
        """
        try:
            print(f"📖 Reading Excel file: {excel_file_path}")
            
            # Read Excel file
            df = pd.read_excel(excel_file_path)
            
            # Get column names
            column_names = df.columns.tolist()
            
            # Get first n rows as preview
            preview_data = df.head(num_rows).to_dict('records')
            
            result = {
                "excel_file_path": excel_file_path,
                "total_rows": len(df),
                "total_columns": len(column_names),
                "column_names": column_names,
                "preview_data": preview_data,
                "preview_rows_count": num_rows
            }
            
            print(f"✅ Successfully read Excel file")
            print(f"   - Total rows: {result['total_rows']}")
            print(f"   - Total columns: {result['total_columns']}")
            print(f"   - Column names: {column_names}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error reading Excel file: {e}")
            raise
    
    def get_table_columns(self, user_id: str, table_full_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get column details for a specific table from user's database.
        
        Args:
            user_id (str): User ID
            table_full_name (str): Full table name (e.g., "dbo.expenseItems")
            
        Returns:
            Optional[List[Dict[str, Any]]]: List of column details if found, None otherwise
        """
        try:
            print(f"🔍 Getting table columns for user: {user_id}, table: {table_full_name}")
            
            column_details = get_column_names_with_details(user_id, table_full_name)
            
            if column_details:
                print(f"✅ Found {len(column_details)} columns in table")
                return column_details
            else:
                print(f"❌ No columns found for table: {table_full_name}")
                return None
                
        except Exception as e:
            print(f"❌ Error getting table columns: {e}")
            return None
    
    def generate_column_mapping(self, excel_columns: List[str], table_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use Gemini AI to generate comprehensive mappings between Excel columns and database table columns.
        
        Args:
            excel_columns (List[str]): List of Excel column names
            table_info (Dict[str, Any]): Detailed table information including CREATE TABLE statement
            
        Returns:
            Dict[str, Any]: Comprehensive mapping structure with bidirectional mappings
        """
        try:
            print("🤖 Generating comprehensive column mapping using Gemini AI...")
            
            # Extract table information
            identity_columns = table_info.get('identity_columns', [])
            create_table_statement = table_info.get('create_table_statement', '')
            sample_data = table_info.get('sample_data', [])
            
            # Get all table column names (excluding identity columns for mapping)
            all_table_columns = []
            for col_def in table_info.get('columns', []):
                # Extract column name from column definition
                col_name = col_def.split()[0].strip('[]')
                all_table_columns.append(col_name)
            
            # Create prompt for Gemini
            prompt = f"""
You are an expert data analyst tasked with mapping Excel columns to database table columns.

EXCEL COLUMNS:
{json.dumps(excel_columns, indent=2)}

DATABASE TABLE STRUCTURE:
{create_table_statement}

SAMPLE DATA FROM DATABASE TABLE:
{json.dumps(sample_data, indent=2, cls=SQLJSONEncoder)}

IMPORTANT RULES:
1. **DO NOT MAP TO IDENTITY COLUMNS**: The following columns are IDENTITY columns (auto-generated) and should NOT be mapped:
   {identity_columns}

2. **DO NOT MAP TO PRIMARY KEYS**: If a column is a primary key and also identity, it should not be mapped.

3. **Consider data types**: Match Excel columns to database columns with compatible data types.

4. **Consider business context**: Look at sample data to understand what each column represents.

5. **Handle naming variations**: Consider common naming patterns (e.g., "First Name" vs "first_name", "Amount" vs "amount").

6. **If no good match exists**: Map to empty string "".

TASK:
Create a mapping between Excel columns and database table columns. For each Excel column:
- Find the best matching database column
- Consider exact name matches first
- Then consider similar names and business meaning
- Exclude identity columns from mapping
- Return ONLY a JSON object with Excel column names as keys and database column names as values

EXAMPLE OUTPUT FORMAT:
{{
  "Excel Column 1": "database_column_1",
  "Excel Column 2": "database_column_2",
  "Unmatched Column": ""
}}

Return ONLY the JSON mapping object:
"""
            
            # Generate mapping using Gemini
            response = self.model.generate_content(prompt)
            
            # Parse the response
            try:
                # Extract JSON from response
                response_text = response.text.strip()
                
                # Remove any markdown formatting if present
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                excel_to_db_mapping = json.loads(response_text)
                
                # Validate that identity columns are not mapped
                for excel_col, db_col in excel_to_db_mapping.items():
                    if db_col in identity_columns:
                        print(f"⚠️  Warning: Excel column '{excel_col}' was mapped to identity column '{db_col}'. Removing mapping.")
                        excel_to_db_mapping[excel_col] = ""
                
                # Create reverse mapping (Database to Excel)
                db_to_excel_mapping = {}
                for db_col in all_table_columns:
                    db_to_excel_mapping[db_col] = ""
                
                # Fill in the reverse mapping
                for excel_col, db_col in excel_to_db_mapping.items():
                    if db_col and db_col in db_to_excel_mapping:
                        db_to_excel_mapping[db_col] = excel_col
                
                # Create comprehensive mapping structure
                comprehensive_mapping = {
                    "excel_to_db": excel_to_db_mapping,
                    "db_to_excel": db_to_excel_mapping,
                    "all_table_columns": all_table_columns,
                    "all_excel_columns": excel_columns,
                    "identity_columns": identity_columns,
                    "mapping_details": []
                }
                
                # Create detailed mapping information
                for db_col in all_table_columns:
                    excel_col = db_to_excel_mapping.get(db_col, "")
                    is_identity = db_col in identity_columns
                    is_mapped = excel_col != ""
                    
                    mapping_detail = {
                        "table_column": db_col,
                        "excel_column": excel_col,
                        "is_identity": is_identity,
                        "is_mapped": is_mapped,
                        "mapping_status": "IDENTITY" if is_identity else ("MAPPED" if is_mapped else "UNMAPPED")
                    }
                    comprehensive_mapping["mapping_details"].append(mapping_detail)
                
                print("✅ Successfully generated comprehensive column mapping")
                return comprehensive_mapping
                
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing Gemini response as JSON: {e}")
                print(f"Raw response: {response.text}")
                # Return empty comprehensive mapping if parsing fails
                return self._create_empty_comprehensive_mapping(excel_columns, all_table_columns, identity_columns)
                
        except Exception as e:
            print(f"❌ Error generating column mapping: {e}")
            # Return empty comprehensive mapping if generation fails
            return self._create_empty_comprehensive_mapping(excel_columns, all_table_columns, identity_columns)
    
    def _create_empty_comprehensive_mapping(self, excel_columns: List[str], table_columns: List[str], identity_columns: List[str]) -> Dict[str, Any]:
        """
        Create an empty comprehensive mapping structure when mapping generation fails.
        
        Args:
            excel_columns (List[str]): List of Excel column names
            table_columns (List[str]): List of table column names
            identity_columns (List[str]): List of identity column names
            
        Returns:
            Dict[str, Any]: Empty comprehensive mapping structure
        """
        excel_to_db_mapping = {col: "" for col in excel_columns}
        db_to_excel_mapping = {col: "" for col in table_columns}
        
        comprehensive_mapping = {
            "excel_to_db": excel_to_db_mapping,
            "db_to_excel": db_to_excel_mapping,
            "all_table_columns": table_columns,
            "all_excel_columns": excel_columns,
            "identity_columns": identity_columns,
            "mapping_details": []
        }
        
        # Create detailed mapping information
        for db_col in table_columns:
            is_identity = db_col in identity_columns
            mapping_detail = {
                "table_column": db_col,
                "excel_column": "",
                "is_identity": is_identity,
                "is_mapped": False,
                "mapping_status": "IDENTITY" if is_identity else "UNMAPPED"
            }
            comprehensive_mapping["mapping_details"].append(mapping_detail)
        
        return comprehensive_mapping
    
    def process_excel_mapping(self, excel_file_path: str, user_id: str, table_full_name: str, num_preview_rows: int = 5) -> Dict[str, Any]:
        """
        Complete process: Read Excel, get table info, and generate comprehensive mapping.
        
        Args:
            excel_file_path (str): Path to the Excel file
            user_id (str): User ID
            table_full_name (str): Full table name (e.g., "dbo.expenseItems")
            num_preview_rows (int): Number of rows to preview (default: 5)
            
        Returns:
            Dict[str, Any]: Complete result with Excel data, table info, and comprehensive mapping
        """
        try:
            print("🚀 Starting Excel to Database Column Mapping Process")
            print("=" * 60)
            
            # Step 1: Read Excel file
            excel_data = self.read_excel_preview(excel_file_path, num_preview_rows)
            
            # Step 2: Get detailed table information
            table_info = self.generate_specific_table_info(user_id, table_full_name)
            
            if not table_info:
                print("❌ Cannot proceed without table information")
                return {
                    "success": False,
                    "error": "No table information found",
                    "excel_data": excel_data,
                    "table_info": None,
                    "mapping": None
                }
            
            # Step 3: Generate comprehensive mapping
            comprehensive_mapping = self.generate_column_mapping(excel_data["column_names"], table_info)
            
            # Step 4: Calculate mapping statistics
            mapping_details = comprehensive_mapping.get("mapping_details", [])
            total_table_columns = len(comprehensive_mapping.get("all_table_columns", []))
            total_excel_columns = len(comprehensive_mapping.get("all_excel_columns", []))
            identity_columns = len(comprehensive_mapping.get("identity_columns", []))
            mapped_columns = len([m for m in mapping_details if m.get("is_mapped", False)])
            unmapped_columns = len([m for m in mapping_details if not m.get("is_mapped", False) and not m.get("is_identity", False)])
            
            # Step 5: Prepare result
            result = {
                "success": True,
                "excel_data": excel_data,
                "table_info": table_info,
                "mapping": comprehensive_mapping,
                "mapping_summary": {
                    "total_excel_columns": total_excel_columns,
                    "total_table_columns": total_table_columns,
                    "identity_columns": identity_columns,
                    "mapped_columns": mapped_columns,
                    "unmapped_columns": unmapped_columns,
                    "mapping_percentage": round((mapped_columns / (total_table_columns - identity_columns)) * 100, 2) if (total_table_columns - identity_columns) > 0 else 0
                }
            }
            
            # Step 6: Display comprehensive mapping results
            self._display_comprehensive_mapping_results(comprehensive_mapping, result["mapping_summary"])
            
            return result
            
        except Exception as e:
            print(f"❌ Error in complete process: {e}")
            return {
                "success": False,
                "error": str(e),
                "excel_data": None,
                "table_info": None,
                "mapping": None
            }
    
    def _display_comprehensive_mapping_results(self, comprehensive_mapping: Dict[str, Any], summary: Dict[str, Any]):
        """
        Display comprehensive mapping results in a clear format.
        
        Args:
            comprehensive_mapping (Dict[str, Any]): Comprehensive mapping structure
            summary (Dict[str, Any]): Mapping summary statistics
        """
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE MAPPING RESULTS")
        print("=" * 80)
        
        # Display summary
        print(f"\n📈 MAPPING SUMMARY:")
        print(f"   • Total Excel columns: {summary['total_excel_columns']}")
        print(f"   • Total Table columns: {summary['total_table_columns']}")
        print(f"   • Identity columns: {summary['identity_columns']}")
        print(f"   • Mapped columns: {summary['mapped_columns']}")
        print(f"   • Unmapped columns: {summary['unmapped_columns']}")
        print(f"   • Mapping success rate: {summary['mapping_percentage']}%")
        
        # Display detailed mapping table
        print(f"\n🔗 DETAILED COLUMN MAPPING:")
        print("-" * 80)
        print(f"{'TABLE COLUMN':<25} {'EXCEL COLUMN':<25} {'STATUS':<15} {'TYPE':<15}")
        print("-" * 80)
        
        mapping_details = comprehensive_mapping.get("mapping_details", [])
        for detail in mapping_details:
            table_col = detail.get("table_column", "")
            excel_col = detail.get("excel_column", "")
            status = detail.get("mapping_status", "")
            is_identity = detail.get("is_identity", False)
            
            # Format status with emoji
            if status == "IDENTITY":
                status_display = "🔒 IDENTITY"
                type_display = "AUTO-GEN"
            elif status == "MAPPED":
                status_display = "✅ MAPPED"
                type_display = "MANUAL"
            else:  # UNMAPPED
                status_display = "❌ UNMAPPED"
                type_display = "MISSING"
            
            # Truncate long column names for display
            table_col_display = table_col[:24] + "..." if len(table_col) > 24 else table_col
            excel_col_display = excel_col[:24] + "..." if len(excel_col) > 24 else excel_col
            
            print(f"{table_col_display:<25} {excel_col_display:<25} {status_display:<15} {type_display:<15}")
        
        print("-" * 80)
        
        # Display identity columns separately
        identity_columns = comprehensive_mapping.get("identity_columns", [])
        if identity_columns:
            print(f"\n🔒 IDENTITY COLUMNS (Auto-generated, cannot be mapped):")
            for identity_col in identity_columns:
                print(f"   • {identity_col}")
        
        # Display unmapped columns
        unmapped_details = [m for m in mapping_details if not m.get("is_mapped", False) and not m.get("is_identity", False)]
        if unmapped_details:
            print(f"\n❌ UNMAPPED TABLE COLUMNS (No Excel column found):")
            for detail in unmapped_details:
                table_col = detail.get("table_column", "")
                print(f"   • {table_col}")
        
        # Display unmapped Excel columns
        all_excel_columns = comprehensive_mapping.get("all_excel_columns", [])
        mapped_excel_columns = [m.get("excel_column") for m in mapping_details if m.get("is_mapped", False)]
        unmapped_excel_columns = [col for col in all_excel_columns if col not in mapped_excel_columns]
        
        if unmapped_excel_columns:
            print(f"\n📄 UNMAPPED EXCEL COLUMNS (No table column found):")
            for excel_col in unmapped_excel_columns:
                print(f"   • {excel_col}")
        
        print("\n" + "=" * 80)

def main():
    """Main function to demonstrate the Excel reader with mapping functionality."""
    
    # Configuration
    user_id = "nilab"
    table_full_name = "dbo.expenseItems"
    
    # Example Excel file path (you can change this)
    excel_file_path = "/Users/nilab/Desktop/projects/Esap-database_agent/Knowledge_base_backend/Quotation agent.xlsx"
    
    if not excel_file_path:
        print("❌ No Excel file path provided")
        return
    
    if not os.path.exists(excel_file_path):
        print(f"❌ Excel file not found: {excel_file_path}")
        return
    
    try:
        # Initialize the Excel reader
        reader = ExcelReaderWithMapping()
        
        # Process the Excel file
        result = reader.process_excel_mapping(excel_file_path, user_id, table_full_name)
        
        if result["success"]:
            print("\n🎉 Excel to Database Column Mapping completed successfully!")
            
            # Save result to JSON file for reference
            output_file = f"excel_mapping_result_{os.path.basename(excel_file_path)}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, cls=SQLJSONEncoder)
            print(f"📄 Results saved to: {output_file}")
            
        else:
            print(f"❌ Process failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
