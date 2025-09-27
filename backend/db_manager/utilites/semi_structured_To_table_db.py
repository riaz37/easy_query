#!/usr/bin/env python3
"""
Semi-structured Data to Database Table System
Provides endpoints for AI-suggested column mapping and Excel data push to database.
"""

import os
import tempfile
import sys
import json
import pandas as pd
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status, Body
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import google.generativeai as genai
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Add the path to import the database manager
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mssql_config import db_manager, APIResponse

# Import column details finder functionality
from column_details_finder import get_column_names_with_details

# Import Excel reader functionality
from excel_reader import ExcelReaderWithMapping

# Load environment variables
load_dotenv()

# Create router
router = APIRouter(prefix="/excel-to-db", tags=["Excel to Database"])

# Initialize Gemini AI
def initialize_gemini():
    """Initialize Gemini AI configuration."""
    gemini_api_key = os.getenv("google_api_key")
    if not gemini_api_key:
        raise ValueError("Google API key not found in environment variables. Please set 'google_api_key'.")
    
    genai.configure(api_key=gemini_api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

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

# Pydantic Models
class ColumnMappingRequest(BaseModel):
    """Request model for getting AI-suggested column mapping."""
    user_id: str = Field(..., min_length=1, max_length=255, description="User ID")
    table_full_name: str = Field(..., min_length=1, description="Full table name (e.g., 'dbo.expenseItems')")

class ColumnMappingResponse(BaseModel):
    """Response model for AI-suggested column mapping."""
    success: bool
    excel_data: Optional[Dict[str, Any]] = None
    table_info: Optional[Dict[str, Any]] = None
    comprehensive_mapping: Optional[Dict[str, Any]] = None
    mapping_summary: Optional[Dict[str, Any]] = None
    # Legacy fields for backward compatibility
    table_columns: Optional[List[Dict[str, Any]]] = None
    ai_mapping: Optional[Dict[str, str]] = None
    error: Optional[str] = None

class DataPushRequest(BaseModel):
    """Request model for pushing Excel data to database."""
    user_id: str = Field(..., min_length=1, max_length=255, description="User ID")
    table_full_name: str = Field(..., min_length=1, description="Full table name (e.g., 'dbo.expenseItems')")
    column_mapping: Dict[str, str] = Field(
        ..., 
        description="Mapping of Excel columns to database columns. Format: {'excel_column_name': 'database_column_name'}",
        example={
            "Product Name": "product_name",
            "Price": "price",
            "Quantity": "quantity",
            "Created Date": "created_date"
        }
    )
    skip_first_row: bool = Field(default=True, description="If True, treats the first row as data and uses it as column names. If False, treats the first row as headers.")

class DataPushWithFileRequest(BaseModel):
    """Request model for pushing Excel data to database with file upload."""
    user_id: str = Field(..., min_length=1, max_length=255, description="User ID")
    table_full_name: str = Field(..., min_length=1, description="Full table name (e.g., 'dbo.expenseItems')")
    column_mapping: Dict[str, str] = Field(
        ..., 
        description="Mapping of Excel columns to database columns. Format: {'excel_column_name': 'database_column_name'}",
        example={
            "Product Name": "product_name",
            "Price": "price",
            "Quantity": "quantity",
            "Created Date": "created_date"
        }
    )
    skip_first_row: bool = Field(default=True, description="If True, treats the first row as data and uses it as column names. If False, treats the first row as headers.")

class DataPushResponse(BaseModel):
    """Response model for data push operation."""
    success: bool
    rows_processed: Optional[int] = None
    rows_inserted: Optional[int] = None
    errors: Optional[List[str]] = None
    error: Optional[str] = None

class ExcelToDatabaseService:
    """Service class for Excel to database operations."""
    
    def __init__(self):
        """Initialize the service."""
        self.gemini_model = initialize_gemini()
        self.excel_reader = ExcelReaderWithMapping()
        print("✅ Excel to Database Service initialized successfully")
    
    def get_ai_suggested_mapping(self, user_id: str, table_full_name: str, excel_file_path: str) -> ColumnMappingResponse:
        """
        Get AI-suggested column mapping for Excel to database table.
        
        Args:
            user_id (str): User ID
            table_full_name (str): Full table name
            excel_file_path (str): Path to Excel file
            
        Returns:
            ColumnMappingResponse: AI-suggested mapping with comprehensive metadata
        """
        try:
            print(f"🤖 Getting AI-suggested mapping for user: {user_id}, table: {table_full_name}")
            
            # Use the new comprehensive process_excel_mapping method
            result = self.excel_reader.process_excel_mapping(excel_file_path, user_id, table_full_name, num_preview_rows=5)
            
            if not result["success"]:
                return ColumnMappingResponse(
                    success=False,
                    error=result.get("error", "Unknown error occurred")
                )
            
            # Extract comprehensive mapping data
            comprehensive_mapping = result.get("mapping", {})
            mapping_summary = result.get("mapping_summary", {})
            
            # Extract specific mapping information for enhanced response
            mapping_details = comprehensive_mapping.get("mapping_details", [])
            
            # Get mapped columns (Excel columns that have database mappings)
            mapped_columns = []
            for detail in mapping_details:
                if detail.get("is_mapped", False) and not detail.get("is_identity", False):
                    mapped_columns.append({
                        "excel_column": detail.get("excel_column", ""),
                        "table_column": detail.get("table_column", ""),
                        "mapping_status": detail.get("mapping_status", "")
                    })
            
            # Get unmapped table columns (excluding identity columns)
            unmapped_table_columns = []
            for detail in mapping_details:
                if not detail.get("is_mapped", False) and not detail.get("is_identity", False):
                    unmapped_table_columns.append({
                        "table_column": detail.get("table_column", ""),
                        "mapping_status": detail.get("mapping_status", "")
                    })
            
            # Get unmapped Excel columns
            all_excel_columns = comprehensive_mapping.get("all_excel_columns", [])
            mapped_excel_columns = [detail.get("excel_column") for detail in mapping_details if detail.get("is_mapped", False)]
            unmapped_excel_columns = [col for col in all_excel_columns if col not in mapped_excel_columns]
            
            # Get all table columns (including identity columns)
            all_table_columns = comprehensive_mapping.get("all_table_columns", [])
            
            # Get identity columns
            identity_columns = comprehensive_mapping.get("identity_columns", [])
            
            # Convert table columns from strings to proper format for legacy compatibility
            table_columns_raw = result.get("table_info", {}).get("columns", [])
            table_columns_formatted = []
            for col_def in table_columns_raw:
                # Extract column name from definition (e.g., "[report_ID] INT NOT NULL" -> "report_ID")
                col_name = col_def.split()[0].strip('[]')
                table_columns_formatted.append({
                    "column_name": col_name,
                    "column_definition": col_def,
                    "is_identity": col_name in identity_columns
                })
            
            # Create enhanced mapping summary
            enhanced_mapping_summary = {
                "total_excel_columns": len(all_excel_columns),
                "total_table_columns": len(all_table_columns),
                "identity_columns": len(identity_columns),
                "mapped_columns": len(mapped_columns),
                "unmapped_table_columns": len(unmapped_table_columns),
                "unmapped_excel_columns": len(unmapped_excel_columns),
                "mapping_percentage": mapping_summary.get("mapping_percentage", 0),
                "mapped_columns_list": mapped_columns,
                "unmapped_table_columns_list": unmapped_table_columns,
                "unmapped_excel_columns_list": unmapped_excel_columns,
                "all_table_columns_list": all_table_columns,
                "all_excel_columns_list": all_excel_columns,
                "identity_columns_list": identity_columns
            }
            
            return ColumnMappingResponse(
                success=True,
                excel_data=result.get("excel_data"),
                table_info=result.get("table_info"),
                comprehensive_mapping=comprehensive_mapping,
                mapping_summary=enhanced_mapping_summary,
                # Legacy fields for backward compatibility
                table_columns=table_columns_formatted,
                ai_mapping=comprehensive_mapping.get("excel_to_db", {})
            )
            
        except Exception as e:
            print(f"❌ Error getting AI mapping: {e}")
            return ColumnMappingResponse(
                success=False,
                error=str(e)
            )
    
    def push_excel_data_to_database(
        self, 
        user_id: str, 
        table_full_name: str, 
        column_mapping: Dict[str, str], 
        excel_file_path: str,
        skip_first_row: bool = True
    ) -> DataPushResponse:
        """
        Push Excel data to database using provided column mapping.
        
        Args:
            user_id (str): User ID
            table_full_name (str): Full table name
            column_mapping (Dict[str, str]): Mapping of Excel columns to database columns
            excel_file_path (str): Path to Excel file
            skip_first_row (bool): If True, treats the first row as data and uses it as column names. If False, treats the first row as headers.
            
        Returns:
            DataPushResponse: Result of the data push operation
        """
        try:
            print(f"📊 Pushing Excel data to database for user: {user_id}, table: {table_full_name}")
            
            # Step 1: Get user's current database details
            user_db_details = db_manager.get_user_current_db_details(user_id)
            if not user_db_details:
                return DataPushResponse(
                    success=False,
                    error="No current database found for user"
                )
            
            db_id = user_db_details.get('db_id')
            if not db_id:
                return DataPushResponse(
                    success=False,
                    error="No database ID found in user's current database"
                )
            
            # Step 2: Get database connection URL
            db_config = db_manager.get_mssql_config(db_id)
            if not db_config:
                return DataPushResponse(
                    success=False,
                    error=f"Database configuration not found for db_id: {db_id}"
                )
            
            db_url = db_config.get('db_url')
            print(f"🔍 Original database URL: {db_url}")
            if not db_url:
                return DataPushResponse(
                    success=False,
                    error="Database connection URL not found"
                )
            
            # Step 3: Read Excel file
            print(f"📖 Reading Excel file: {excel_file_path}")
            
            # Read Excel file with proper header handling
            if skip_first_row:
                # If skip_first_row is True, read without headers and use first row as data
                print("📋 Reading Excel file without headers (first row will be treated as data)")
                df = pd.read_excel(excel_file_path, header=None)
                print(f"📊 Raw data shape: {df.shape}")
                print(f"📊 First row (will become column names): {df.iloc[0].tolist()}")
                
                # Use the first row as column names
                df.columns = df.iloc[0]
                # Remove the first row since it's now the column names
                df = df.iloc[1:].reset_index(drop=True)
                print(f"📊 After processing - shape: {df.shape}")
                print(f"📊 Column names: {df.columns.tolist()}")
            else:
                # If skip_first_row is False, read with headers (default behavior)
                print("📋 Reading Excel file with headers")
                df = pd.read_excel(excel_file_path)
                print(f"📊 Data shape: {df.shape}")
                print(f"📊 Column names: {df.columns.tolist()}")
            
            # Step 4: Initial mapping validation (before database connection)
            excel_columns = df.columns.tolist()
            initial_mapping = {}
            errors = []
            
            for excel_col, db_col in column_mapping.items():
                if excel_col in excel_columns and db_col:
                    initial_mapping[excel_col] = db_col
                elif excel_col not in excel_columns:
                    errors.append(f"Excel column '{excel_col}' not found in file")
                elif not db_col:
                    errors.append(f"Database column not specified for Excel column '{excel_col}'")
            
            if not initial_mapping:
                return DataPushResponse(
                    success=False,
                    error="No valid column mappings found",
                    errors=errors
                )
            
            print(f"✅ Initial mapping: {initial_mapping}")
            
            # Step 5: Convert database URL to SQLAlchemy format and connect
            print("🔌 Converting database URL to SQLAlchemy format")
            sqlalchemy_url = None
            
            try:
                sqlalchemy_url = convert_to_sqlalchemy_url(db_url)
                print("✅ Database URL converted successfully")
            except Exception as e:
                print(f"⚠️  URL conversion failed: {e}")
                print("🔄 Trying fallback approach...")
                
                # Fallback: Use the exact same format as schema_generator
                # This assumes the db_url might be in a different format
                try:
                    # Try to use the URL as-is with SQLAlchemy
                    if not db_url.startswith('mssql'):
                        # If it's not a SQLAlchemy URL, try to construct one
                        # This is a last resort - we'll try to parse it manually
                        print("🔄 Attempting manual URL construction...")
                        
                        # For now, let's try the schema_generator format directly
                        # This is a temporary fallback - in production, you'd want better parsing
                        sqlalchemy_url = f"mssql+pyodbc://{db_url}?driver=ODBC+Driver+18+for+SQL+Server&encrypt=no"
                    else:
                        sqlalchemy_url = db_url
                        
                    print(f"🔄 Fallback URL: {sqlalchemy_url}")
                    
                except Exception as fallback_error:
                    return DataPushResponse(
                        success=False,
                        error=f"Failed to convert database URL: {str(e)}. Fallback also failed: {str(fallback_error)}"
                    )
            
            print("🔌 Connecting to MSSQL database using SQLAlchemy")
            try:
                engine = create_engine(sqlalchemy_url, fast_executemany=True)
                conn = engine.connect()
                print("✅ Database connection successful!")
            except SQLAlchemyError as e:
                return DataPushResponse(
                    success=False,
                    error=f"Failed to connect to database: {str(e)}"
                )
            
            # Step 6: Get table schema and handle identity columns
            print("🔍 Getting table schema to identify identity columns...")
            valid_mapping = {}
            identity_columns = []
            
            try:
                schema_query = text("""
                    SELECT COLUMN_NAME, IS_IDENTITY
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
                """)
                
                schema_name = table_full_name.split('.')[0] if '.' in table_full_name else 'dbo'
                table_name = table_full_name.split('.')[-1] if '.' in table_full_name else table_full_name
                
                schema_result = conn.execute(schema_query, {"schema": schema_name, "table": table_name})
                column_info = {row[0]: row[1] for row in schema_result.fetchall()}
                
                print(f"🔍 Table schema info: {column_info}")
                
                # Now filter out identity columns from the mapping
                for excel_col, db_col in initial_mapping.items():
                    # Check if this is an identity column
                    if column_info.get(db_col, False):
                        print(f"⚠️  Skipping identity column: {db_col}")
                        identity_columns.append(db_col)
                        continue
                    
                    valid_mapping[excel_col] = db_col
                
                print(f"✅ Valid mapping (excluding identity columns): {valid_mapping}")
                if identity_columns:
                    print(f"⚠️  Identity columns excluded: {identity_columns}")
                
            except Exception as e:
                print(f"⚠️  Could not get table schema: {e}")
                print("🔄 Using initial mapping without identity column detection...")
                valid_mapping = initial_mapping
            
            if not valid_mapping:
                return DataPushResponse(
                    success=False,
                    error="No valid column mappings found after excluding identity columns"
                )
            
            # Step 7: Transform data according to mapping
            print("🔄 Transforming data according to mapping")
            mapped_df = df[list(valid_mapping.keys())].copy()
            mapped_df.columns = list(valid_mapping.values())
            

            
            # Step 8: Prepare insert statement with named parameters
            table_name = table_full_name.split('.')[-1] if '.' in table_full_name else table_full_name
            columns = list(valid_mapping.values())
            
            # Create named parameters for SQLAlchemy
            param_names = [f'param_{i}' for i in range(len(columns))]
            placeholders = ', '.join([f':{param}' for param in param_names])
            
            insert_query = f"INSERT INTO {table_full_name} ({', '.join(columns)}) VALUES ({placeholders})"
            print(f"🔍 Insert query: {insert_query}")
            
            # Step 9: Insert data using pandas to_sql (more reliable for bulk inserts)
            print(f"💾 Inserting {len(mapped_df)} rows into {table_full_name}")
            
            try:
                # Use pandas to_sql for bulk insert (more reliable)
                print("🔄 Using pandas to_sql for bulk insert...")
                
                # Convert NaN to None for database insertion
                mapped_df_clean = mapped_df.where(pd.notnull(mapped_df), None)
                
                # Insert data using pandas to_sql
                mapped_df_clean.to_sql(
                    name=table_name,
                    schema=table_full_name.split('.')[0] if '.' in table_full_name else None,
                    con=engine,
                    if_exists='append',
                    index=False,
                    method='multi',  # Use multi-row insert for better performance
                    chunksize=1000   # Process in chunks for large datasets
                )
                
                rows_inserted = len(mapped_df)
                print(f"✅ Successfully inserted {rows_inserted} rows using pandas to_sql")
                
            except Exception as e:
                print(f"❌ pandas to_sql failed: {e}")
                
                # Check if it's an identity column error
                if "IDENTITY_INSERT" in str(e) and identity_columns:
                    print("🔄 Identity column error detected. Trying with IDENTITY_INSERT...")
                    
                    try:
                        # Enable IDENTITY_INSERT for the table
                        identity_insert_on = text(f"SET IDENTITY_INSERT {table_full_name} ON")
                        conn.execute(identity_insert_on)
                        
                        # Try pandas to_sql again
                        mapped_df_clean.to_sql(
                            name=table_name,
                            schema=table_full_name.split('.')[0] if '.' in table_full_name else None,
                            con=engine,
                            if_exists='append',
                            index=False,
                            method='multi',
                            chunksize=1000
                        )
                        
                        # Disable IDENTITY_INSERT
                        identity_insert_off = text(f"SET IDENTITY_INSERT {table_full_name} OFF")
                        conn.execute(identity_insert_off)
                        
                        rows_inserted = len(mapped_df)
                        print(f"✅ Successfully inserted {rows_inserted} rows with IDENTITY_INSERT")
                        
                    except Exception as identity_error:
                        print(f"❌ IDENTITY_INSERT approach failed: {identity_error}")
                        print("🔄 Falling back to manual insert...")
                        
                        # Disable IDENTITY_INSERT if it was enabled
                        try:
                            identity_insert_off = text(f"SET IDENTITY_INSERT {table_full_name} OFF")
                            conn.execute(identity_insert_off)
                        except:
                            pass
                        
                        # Continue to manual insert fallback
                        raise e  # Re-raise the original error to trigger manual insert
                else:
                    print("🔄 Falling back to manual insert...")
                
                # Fallback to manual insert
                rows_inserted = 0
                insert_errors = []
                
                # Check if we need IDENTITY_INSERT for manual insert
                need_identity_insert = False
                if identity_columns:
                    print(f"🔄 Manual insert: Identity columns detected: {identity_columns}")
                    need_identity_insert = True
                
                try:
                    if need_identity_insert:
                        # Enable IDENTITY_INSERT for manual insert
                        identity_insert_on = text(f"SET IDENTITY_INSERT {table_full_name} ON")
                        conn.execute(identity_insert_on)
                        print("✅ IDENTITY_INSERT enabled for manual insert")
                    
                    for index, row in mapped_df.iterrows():
                        try:
                            # Convert NaN to None for database insertion and create parameter dict
                            row_values = [None if pd.isna(val) else val for val in row.values]
                            param_dict = dict(zip(param_names, row_values))
                            
                            # Use SQLAlchemy text() for parameterized query with named parameters
                            stmt = text(insert_query)
                            conn.execute(stmt, param_dict)
                            rows_inserted += 1
                            
                            # Commit every 100 rows for better performance
                            if rows_inserted % 100 == 0:
                                conn.commit()
                                
                        except Exception as e:
                            error_msg = f"Row {index + 1}: {str(e)}"
                            insert_errors.append(error_msg)
                            print(f"❌ Error inserting row {index + 1}: {e}")
                    
                    # Final commit for remaining rows
                    conn.commit()
                    
                finally:
                    if need_identity_insert:
                        # Disable IDENTITY_INSERT
                        try:
                            identity_insert_off = text(f"SET IDENTITY_INSERT {table_full_name} OFF")
                            conn.execute(identity_insert_off)
                            print("✅ IDENTITY_INSERT disabled")
                        except Exception as e:
                            print(f"⚠️  Warning: Could not disable IDENTITY_INSERT: {e}")
                
            except Exception as e:
                print(f"❌ Error during data insertion: {e}")
                return DataPushResponse(
                    success=False,
                    error=f"Data insertion failed: {str(e)}"
                )
            
            # Step 10: Close connection
            try:
                conn.close()
                engine.dispose()
                print("✅ Database connection closed successfully")
            except Exception as e:
                print(f"⚠️  Warning: Error closing database connection: {e}")
            
            print(f"✅ Successfully inserted {rows_inserted} rows")
            
            return DataPushResponse(
                success=True,
                rows_processed=len(mapped_df),
                rows_inserted=rows_inserted,
                errors=insert_errors if 'insert_errors' in locals() and insert_errors else None
            )
            
        except Exception as e:
            print(f"❌ Error pushing data to database: {e}")
            return DataPushResponse(
                success=False,
                error=str(e)
            )

# Initialize service
excel_service = ExcelToDatabaseService()

# API Endpoints
@router.post("/get-ai-mapping", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def get_ai_suggested_mapping(
    user_id: str = Form(...),
    table_full_name: str = Form(...),
    excel_file: UploadFile = File(...)
):
    """
    Get AI-suggested column mapping for Excel to database table.
    
    Args:
        user_id: User ID
        table_full_name: Full table name (e.g., "dbo.expenseItems")
        excel_file: Excel file to analyze
        
    Returns:
        AI-suggested column mapping with metadata
    """
    try:
        # Save uploaded file to a safe temp location (cross-platform)
        original_name = os.path.basename(excel_file.filename or "uploaded.xlsx")
        _, ext = os.path.splitext(original_name)
        temp_file_path = None
        try:
            content = await excel_file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".xlsx") as tmp:
                tmp.write(content)
                temp_file_path = tmp.name
            
            # Get AI mapping
            result = excel_service.get_ai_suggested_mapping(user_id, table_full_name, temp_file_path)
        finally:
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass
        
        if result.success:
            return APIResponse(
                status="success",
                message="AI-suggested column mapping generated successfully",
                data={
                    "all_table_columns": result.mapping_summary.get("all_table_columns_list", []),
                    "identity_columns": result.mapping_summary.get("identity_columns_list", []),
                    "all_excel_columns": result.mapping_summary.get("all_excel_columns_list", []),
                    "mapping_details": result.comprehensive_mapping.get("mapping_details", [])
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating AI mapping: {str(e)}"
        )

@router.post("/push-data", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def push_excel_data_to_database(
    user_id: str = Form(..., description="User ID"),
    table_full_name: str = Form(..., description="Full table name (e.g., 'dbo.expenseItems')"),
    column_mapping: str = Form(..., description="JSON string of column mapping. Format: {'excel_column_name': 'database_column_name'}"),
    skip_first_row: bool = Form(default=True, description="If True, treats the first row as data and uses it as column names. If False, treats the first row as headers."),
    excel_file: UploadFile = File(..., description="Excel file to upload and process")
):
    """
    Push Excel data to database using provided column mapping.
    
    This endpoint accepts form data with column mapping configuration and an Excel file upload.
    
    **Column Mapping Format:**
    ```json
    {
        "Excel Column Name": "database_column_name",
        "Product Name": "product_name",
        "Price": "price",
        "Quantity": "quantity"
    }
    ```
    
    **Column Mapping Structure:**
    - **Left side (Excel columns)**: Names of columns in your Excel file
    - **Right side (Database columns)**: Names of columns in your database table
    
    **Example:**
    ```json
    {
        "Product Name": "product_name",
        "Unit Price": "price",
        "Qty": "quantity",
        "Total Amount": "amount",
        "Invoice Date": "invoice_date",
        "Supplier": "supplier_name"
    }
    ```
    
    Args:
        user_id: User ID
        table_full_name: Full table name (e.g., "dbo.expenseItems")
        column_mapping: JSON string of column mapping {"excel_col": "db_col"}
        skip_first_row: If True, treats the first row as data and uses it as column names. If False, treats the first row as headers.
        excel_file: Excel file to push to database
        
    Returns:
        Result of the data push operation with rows processed and inserted
    """
    try:
        print(f"🔍 Received request:")
        print(f"   - User ID: {user_id}")
        print(f"   - Table: {table_full_name}")
        print(f"   - Column mapping: {column_mapping}")
        print(f"   - Skip first row: {skip_first_row}")
        print(f"   - Excel file: {excel_file.filename}")
        
        # Parse column mapping JSON
        try:
            # Handle both string and dict inputs
            if isinstance(column_mapping, str):
                mapping_dict = json.loads(column_mapping)
            elif isinstance(column_mapping, dict):
                mapping_dict = column_mapping
            else:
                raise ValueError(f"Invalid column_mapping type: {type(column_mapping)}")
                
            print(f"✅ Parsed mapping: {mapping_dict}")
            print(f"✅ Mapping keys: {list(mapping_dict.keys())}")
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid column mapping JSON format: {str(e)}"
            )
        except Exception as e:
            print(f"❌ Mapping parsing error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error parsing column mapping: {str(e)}"
            )
        
        # Validate mapping
        if not mapping_dict:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Column mapping cannot be empty"
            )
        
        print(f"🔍 Validating mapping with {len(mapping_dict)} entries...")
        for excel_col, db_col in mapping_dict.items():
            if not excel_col or not db_col:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid mapping entry: '{excel_col}' -> '{db_col}'. Both Excel and database column names must be non-empty."
                )
        
        print(f"✅ Mapping validation passed")
        
        # Save uploaded file to a safe temp location (cross-platform)
        original_name = os.path.basename(excel_file.filename or "uploaded.xlsx")
        _, ext = os.path.splitext(original_name)
        temp_file_path = None
        try:
            content = await excel_file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".xlsx") as tmp:
                tmp.write(content)
                temp_file_path = tmp.name
            
            # Push data to database
            result = excel_service.push_excel_data_to_database(
                user_id, 
                table_full_name, 
                mapping_dict, 
                temp_file_path,
                skip_first_row
            )
        finally:
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass
        
        if result.success:
            return APIResponse(
                status="success",
                message=f"Successfully pushed {result.rows_inserted} rows to database",
                data={
                    "rows_processed": result.rows_processed,
                    "rows_inserted": result.rows_inserted,
                    "errors": result.errors
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error pushing data to database: {str(e)}"
        )

# Health check endpoint
@router.get("/health", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def semi_structured_health_check():
    """Health check endpoint for Excel to Database service."""
    return APIResponse(
        status="success",
        message="Excel to Database service is healthy",
        data={
            "service": "excel-to-database",
            "status": "running"
        }
    )

if __name__ == "__main__":
    print("🚀 Excel to Database Service")
    print("=" * 50)
    print("This module provides endpoints for:")
    print("1. AI-suggested column mapping")
    print("2. Excel data push to database")
    print("=" * 50)
