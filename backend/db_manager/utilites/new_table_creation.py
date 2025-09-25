#!/usr/bin/env python3
"""
New Table Creation Utilities

Provides endpoints to:
- Retrieve current `db_id` for a `user_id`
- Create a new table by specifying column names and types
- List supported SQL Server data types for columns
- Track created tables with their schema information
"""

import os
import sys
import json
import tempfile
import asyncio
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Make sure we can import `db_manager` and `APIResponse` defined in `db_manager/mssql_config.py`
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from mssql_config import db_manager, APIResponse, run_update_config_workflow, MSSQLDBConfigUpdate, task_executor, read_schema_file_content  # type: ignore

# Import additional functions needed for the workflow
from table_info_generator import generate_table_info
from schema_generator import generate_schema_and_data, generate_single_table_schema, SQLJSONEncoder

router = APIRouter(tags=["New Table Management"])


def _preferred_odbc_driver() -> str:
    # Allow overriding via env; default to Driver 17 per user's environment
    return os.getenv("PREFERRED_ODBC_DRIVER", "ODBC Driver 17 for SQL Server")


def convert_to_sqlalchemy_url(db_url: str) -> str:
    """
    Convert a database connection string to SQLAlchemy format.
    Mirrors the behavior used in `semi_structured_To_table_db.py`.
    """
    # Already SQLAlchemy style
    if db_url.startswith("mssql+pyodbc://"):
        # Keep existing driver if present; otherwise append preferred driver
        url = db_url
        if "driver=" in url:
            if "encrypt=" not in url:
                url += "&encrypt=no"
        else:
            driver = _preferred_odbc_driver().replace(" ", "+")
            url += ("&" if "?" in url else "?") + f"driver={driver}&encrypt=no"
        return url

    # ODBC connection string style
    if ";" in db_url and ("Server=" in db_url or "Data Source=" in db_url):
        if "Server=" in db_url:
            server_part = db_url.split("Server=")[1].split(";")[0]
        elif "Data Source=" in db_url:
            server_part = db_url.split("Data Source=")[1].split(";")[0]
        else:
            raise ValueError("Could not find Server or Data Source in connection string")

        if "Database=" in db_url:
            database_part = db_url.split("Database=")[1].split(";")[0]
        elif "Initial Catalog=" in db_url:
            database_part = db_url.split("Initial Catalog=")[1].split(";")[0]
        else:
            raise ValueError("Could not find Database or Initial Catalog in connection string")

        if "User Id=" in db_url:
            username_part = db_url.split("User Id=")[1].split(";")[0]
        elif "UID=" in db_url:
            username_part = db_url.split("UID=")[1].split(";")[0]
        else:
            raise ValueError("Could not find User Id or UID in connection string")

        if "Password=" in db_url:
            password_part = db_url.split("Password=")[1].split(";")[0]
        elif "PWD=" in db_url:
            password_part = db_url.split("PWD=")[1].split(";")[0]
        else:
            raise ValueError("Could not find Password or PWD in connection string")

        return (
            f"mssql+pyodbc://{username_part}:{password_part}@{server_part}/{database_part}"
            f"?driver=ODBC+Driver+18+for+SQL+Server&encrypt=no"
        )

    # SQLAlchemy-like URL but missing driver
    if db_url.startswith("mssql://"):
        sqlalchemy_url = db_url.replace("mssql://", "mssql+pyodbc://")
        driver = _preferred_odbc_driver().replace(" ", "+")
        if "?" not in sqlalchemy_url:
            sqlalchemy_url += f"?driver={driver}&encrypt=no"
        elif "driver=" not in sqlalchemy_url:
            sqlalchemy_url += f"&driver={driver}&encrypt=no"
        return sqlalchemy_url

    # Direct connection style: username:password@host:port/database
    if "@" in db_url and ":" in db_url:
        driver = _preferred_odbc_driver().replace(" ", "+")
        return f"mssql+pyodbc://{db_url}?driver={driver}&encrypt=no"

    raise ValueError(f"Unsupported connection string format: {db_url}")


def create_user_created_table_tracking_table():
    """
    Create the usercreatedtable table in PostgreSQL to track user-created tables.
    This table stores information about tables created by users with their schema details.
    """
    create_table_query = """
    CREATE TABLE IF NOT EXISTS usercreatedtable (
        id SERIAL PRIMARY KEY,
        db_id INTEGER NOT NULL,
        user_id VARCHAR(255) NOT NULL,
        table_details JSONB NOT NULL DEFAULT '{}',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (db_id) REFERENCES mssql_config(db_id) ON DELETE CASCADE
    );
    
    -- Create indexes for better query performance
    CREATE INDEX IF NOT EXISTS idx_usercreatedtable_db_id ON usercreatedtable(db_id);
    CREATE INDEX IF NOT EXISTS idx_usercreatedtable_user_id ON usercreatedtable(user_id);
    CREATE INDEX IF NOT EXISTS idx_usercreatedtable_created_at ON usercreatedtable(created_at);
    CREATE INDEX IF NOT EXISTS idx_usercreatedtable_table_details ON usercreatedtable USING GIN(table_details);
    """
    
    try:
        # Use the same connection method as db_manager
        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            cursor.execute(create_table_query)
            conn.commit()
        conn.close()
        print("✅ Table 'usercreatedtable' created successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating usercreatedtable: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create usercreatedtable: {str(e)}"
        )


def store_table_schema_in_tracking_table(db_id: int, user_id: str, table_full_name: str, table_schema: Dict[str, Any]) -> bool:
    """
    Store the table schema information in the usercreatedtable tracking table.
    Updated to use the new structure with one row per user_id and db_id combination.
    Automatically initializes business rule record if it doesn't exist.
    
    Args:
        db_id (int): Database ID from mssql_config
        user_id (str): User ID who created the table
        table_full_name (str): Full table name (schema.table_name)
        table_schema (Dict[str, Any]): Schema information for the table
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Prepare the table details JSON
        table_details = {
            "table_full_name": table_full_name,
            "schema_name": table_full_name.split('.')[0] if '.' in table_full_name else 'dbo',
            "table_name": table_full_name.split('.')[-1] if '.' in table_full_name else table_full_name,
            "table_schema": table_schema,
            "created_by_user": user_id,
            "creation_timestamp": table_schema.get("created_at", None)
        }
        
        # Check if user already has a business rule record
        existing_business_rule = db_manager.get_user_business_rule(user_id, db_id)
        
        # Use the new database manager method to create or update the record
        # If no business rule exists, initialize with an empty string
        result = db_manager.create_or_update_user_table_tracking(
            user_id=user_id,
            db_id=db_id,
            table_details=table_details,
            business_rule=existing_business_rule if existing_business_rule is not None else ""
        )
        
        if result:
            print(f"✅ Table schema stored in tracking table for {table_full_name}")
            
            # If this is the first table and no business rule exists, suggest creating one
            if existing_business_rule is None:
                print(f"💡 Tip: User '{user_id}' can now set business rules for their tables using PUT /new-table/user-business-rule/{user_id}")
            
            return True
        else:
            print(f"❌ Failed to store table schema for {table_full_name}")
            return False
        
    except Exception as e:
        print(f"❌ Error storing table schema: {e}")
        return False


def generate_and_find_table_schema(db_url: str, table_full_name: str) -> Optional[Dict[str, Any]]:
    """
    Generate schema for a single table only - much faster than processing entire database.
    
    Args:
        db_url (str): Database connection URL
        table_full_name (str): Full table name to find (schema.table_name)
        
    Returns:
        Optional[Dict[str, Any]]: Table schema information if found, None otherwise
    """
    try:
        print(f"🔍 Generating schema for single table: {table_full_name}")
        
        # Parse schema and table name from full name
        if '.' in table_full_name:
            schema_name, table_name = table_full_name.split('.', 1)
        else:
            schema_name = 'dbo'  # Default schema
            table_name = table_full_name
        
        # Use the optimized single table schema generator
        table_schema = generate_single_table_schema(
            database_url=db_url,
            schema_name=schema_name,
            table_name=table_name,
            sample_row_count=10  # Reduced sample size for faster processing
        )
        
        if table_schema:
            print(f"✅ Successfully generated schema for {table_full_name}")
            return table_schema
        else:
            print(f"⚠️ Could not generate schema for {table_full_name}")
            return None
                
    except Exception as e:
        print(f"❌ Error generating schema for {table_full_name}: {e}")
        return None


def get_supported_sql_server_types() -> Dict[str, List[str]]:
    """Return categorized supported SQL Server data types."""
    return {
        "numeric": [
            "BIT",
            "TINYINT",
            "SMALLINT",
            "INT",
            "BIGINT",
            "DECIMAL(18,2)",
            "NUMERIC(18,2)",
            "FLOAT",
            "REAL",
            "MONEY",
            "SMALLMONEY",
        ],
        "string": [
            "CHAR(10)",
            "VARCHAR(50)",
            "NVARCHAR(50)",
            "NCHAR(10)",
            "TEXT",
            "NTEXT",
        ],
        "date_time": [
            "DATE",
            "TIME",
            "SMALLDATETIME",
            "DATETIME",
            "DATETIME2",
            "DATETIMEOFFSET",
        ],
        "binary": [
            "BINARY(50)",
            "VARBINARY(50)",
            "IMAGE",
        ],
        "other": [
            "UNIQUEIDENTIFIER",
            "XML",
            "SQL_VARIANT",
        ],
    }


def _extract_base_type(data_type: str) -> str:
    base = data_type.strip().upper()
    if "(" in base:
        base = base.split("(")[0].strip()
    # Normalize synonyms
    synonyms = {
        "INTEGER": "INT",
        "SMALLDATETIME": "SMALLDATETIME",
        "DOUBLE": "FLOAT",
    }
    return synonyms.get(base, base)


def _is_valid_sql_server_type(data_type: str) -> bool:
    supported = get_supported_sql_server_types()
    supported_bases = { _extract_base_type(t) for cat in supported.values() for t in cat }
    base = _extract_base_type(data_type)
    return base in supported_bases


class ColumnDefinition(BaseModel):
    name: str = Field(..., min_length=1, description="Column name")
    data_type: str = Field(..., min_length=1, description="SQL Server data type, e.g., VARCHAR(50), INT")
    nullable: bool = Field(default=True, description="Allow NULL values")
    is_primary: bool = Field(default=False, description="Part of PRIMARY KEY")
    is_identity: bool = Field(default=False, description="Use IDENTITY(1,1) for auto-increment")

    @validator("data_type")
    def validate_data_type(cls, v: str) -> str:
        if not _is_valid_sql_server_type(v):
            raise ValueError(
                "Unsupported or invalid SQL Server data type. Use /new-table/data-types to see options."
            )
        return v


class CreateTableRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="User ID")
    table_name: str = Field(..., min_length=1, description="Table name without schema")
    schema: str = Field(default="dbo", min_length=1, description="Schema name")
    columns: List[ColumnDefinition] = Field(..., min_items=1, description="List of column definitions")

    @validator("table_name")
    def validate_table_name(cls, v: str) -> str:
        name = v.strip()
        if "[" in name or "]" in name or ";" in name:
            raise ValueError("Invalid characters in table name")
        return name

    @validator("schema")
    def validate_schema(cls, v: str) -> str:
        name = v.strip()
        if "[" in name or "]" in name or ";" in name or "." in name:
            raise ValueError("Invalid characters in schema name")
        return name


@router.get("/data-types", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def list_supported_data_types():
    """List supported SQL Server data types for columns."""
    types = get_supported_sql_server_types()
    return APIResponse(
        status="success",
        message="Supported SQL Server data types",
        data=types,
    )


@router.post("/create", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def create_table(payload: CreateTableRequest):
    """
    Create a new table in the user's current MSSQL database and track it with schema information.

    - Validates data types
    - Prevents creating a table that already exists
    - Supports PRIMARY KEY and IDENTITY columns
    - Generates and stores table schema information
    """
    try:
        print(f"🚀 Creating table for user: {payload.user_id}")
        print(f"📋 Table: {payload.schema}.{payload.table_name}")
        print(f"📊 Columns: {len(payload.columns)}")
        
        # Step 1: Resolve user's current db and fetch connection URL
        user_db = db_manager.get_user_current_db_details(payload.user_id)
        if not user_db:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No current database found for user",
            )
        db_id = user_db.get("db_id")
        db_config = db_manager.get_mssql_config(db_id)
        if not db_config:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Database configuration not found for db_id: {db_id}",
            )
        db_url = db_config.get("db_url")
        if not db_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Database connection URL not found",
            )

        # Step 2: Build CREATE TABLE statement
        schema_name = payload.schema
        table_name = payload.table_name
        full_name = f"{schema_name}.{table_name}"

        # Validate at least one non-identity primary key if multiple specified
        primary_cols = [c.name for c in payload.columns if c.is_primary]

        column_lines: List[str] = []
        for col in payload.columns:
            col_parts: List[str] = [f"[{col.name}]", col.data_type]
            if col.is_identity:
                col_parts.append("IDENTITY(1,1)")
            col_parts.append("NOT NULL" if not col.nullable else "NULL")
            column_lines.append(" ".join(col_parts))

        # Table-level primary key if multiple columns specified as primary
        table_constraints: List[str] = []
        if len(primary_cols) == 1:
            # Inline PRIMARY KEY on the single primary column: adjust the line
            for i, col in enumerate(payload.columns):
                if col.name == primary_cols[0]:
                    # Append PRIMARY KEY to that column definition
                    column_lines[i] = column_lines[i] + " PRIMARY KEY"
                    break
        elif len(primary_cols) > 1:
            cols_csv = ", ".join([f"[{c}]" for c in primary_cols])
            table_constraints.append(f"PRIMARY KEY ({cols_csv})")

        all_lines = column_lines + table_constraints
        create_sql = f"CREATE TABLE [{schema_name}].[{table_name}] (\n  " + ",\n  ".join(all_lines) + "\n)"

        # Step 3: Connect and execute
        try:
            sqlalchemy_url = convert_to_sqlalchemy_url(db_url)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid database URL: {str(e)}",
            )

        try:
            engine = create_engine(sqlalchemy_url, fast_executemany=True)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize database engine: {str(e)}",
            )

        # Check if table exists before creating
        exists_query = text(
            """
            SELECT 1
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
            """
        )

        try:
            with engine.begin() as conn:
                result = conn.execute(exists_query, {"schema": schema_name, "table": table_name}).fetchone()
                if result:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Table '{full_name}' already exists",
                    )

                conn.execute(text(create_sql))
                print(f"✅ Table '{full_name}' created successfully in MSSQL")

        except HTTPException:
            raise
        except SQLAlchemyError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create table: {str(e)}",
            )
        finally:
            try:
                engine.dispose()
            except Exception:
                pass

        # Step 4: Generate schema and store table information
        print(f"🔍 Generating schema to capture table information...")
        table_schema = generate_and_find_table_schema(db_url, full_name)
        
        if table_schema:
            print(f"✅ Found table schema, storing in tracking table...")
            # Store the table schema in the tracking table
            store_success = store_table_schema_in_tracking_table(
                db_id=db_id,
                user_id=payload.user_id,
                table_full_name=full_name,
                table_schema=table_schema
            )
            
            if store_success:
                print(f"✅ Table schema stored successfully")
            else:
                print(f"⚠️ Failed to store table schema, but table was created successfully")
        else:
            print(f"⚠️ Could not generate schema for tracking, but table was created successfully")

        return APIResponse(
            status="success",
            message=f"Table '{full_name}' created successfully",
            data={
                "schema": schema_name,
                "table": table_name,
                "columns": [c.dict() for c in payload.columns],
                "table_schema_stored": table_schema is not None,
                "db_id": db_id
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating table: {str(e)}",
        )


@router.get("/user-tables/{user_id}", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def get_user_created_tables(user_id: str):
    """
    Get all tables created by a specific user in their current database.
    Updated to use the new structure with one row per user_id and db_id combination.
    
    Args:
        user_id (str): User ID to get created tables for
        
    Returns:
        List of user-created tables with their schema details for the current database
    """
    try:
        # Step 1: Get user's current database ID
        user_current_db = db_manager.get_user_current_db_details(user_id)
        if not user_current_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No current database found for user '{user_id}'"
            )
        
        current_db_id = user_current_db.get('db_id')
        if not current_db_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No database ID found for user '{user_id}'"
            )
        
        # Step 2: Get the user's table tracking record for the current database
        record = db_manager.get_user_table_tracking_record(user_id, current_db_id)
        
        if not record:
            return APIResponse(
                status="success",
                message=f"No tables found for user '{user_id}' in current database (db_id: {current_db_id})",
                data={
                    "user_id": user_id,
                    "current_db_id": current_db_id,
                    "tables": [],
                    "business_rule": "",
                    "count": 0
                }
            )
        
        # Extract tables from the record
        table_details = record.get('table_details', [])
        if not isinstance(table_details, list):
            table_details = []
        
        business_rule = record.get('business_rule', '')
        
        return APIResponse(
            status="success",
            message=f"Found {len(table_details)} tables created by user {user_id} in current database (db_id: {current_db_id})",
            data={
                "user_id": user_id,
                "current_db_id": current_db_id,
                "tables": table_details,
                "business_rule": business_rule,
                "business_rule_exists": bool(business_rule),
                "count": len(table_details),
                "created_at": record.get('created_at'),
                "updated_at": record.get('updated_at'),
                "business_rule_endpoint": f"/new-table/user-business-rule/{user_id}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user tables: {str(e)}"
        )


@router.get("/user-tables-by-db/{db_id}", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def get_tables_by_database(db_id: int):
    """
    Get all tables created in a specific database by any user.
    Updated to use the new structure with one row per user_id and db_id combination.
    
    Args:
        db_id (int): Database ID to get tables for
        
    Returns:
        List of all tables created in the specified database with their schema details
    """
    try:
        # Step 1: Verify the database exists
        db_config = db_manager.get_mssql_config(db_id)
        if not db_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Database with ID {db_id} not found"
            )
        
        # Step 2: Get all records for the specified database
        select_query = """
        SELECT id, db_id, user_id, table_details, business_rule, created_at, updated_at
        FROM usercreatedtable
        WHERE db_id = %s
        ORDER BY created_at DESC
        """
        
        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            cursor.execute(select_query, (db_id,))
            results = cursor.fetchall()
        conn.close()
        
        # Convert results to list of dictionaries with flattened table structure
        all_tables = []
        user_records = []
        
        for row in results:
            user_record = {
                "id": row[0],
                "db_id": row[1],
                "user_id": row[2],
                "table_details": row[3] if isinstance(row[3], list) else [],
                "business_rule": row[4] if row[4] else "",
                "created_at": row[5].isoformat() if row[5] else None,
                "updated_at": row[6].isoformat() if row[6] else None
            }
            user_records.append(user_record)
            
            # Flatten tables for backward compatibility
            table_details = user_record["table_details"]
            if isinstance(table_details, list):
                for table in table_details:
                    table_with_user = table.copy()
                    table_with_user["created_by_user"] = user_record["user_id"]
                    table_with_user["user_business_rule"] = user_record["business_rule"]
                    all_tables.append(table_with_user)
        
        # Get database name for better response
        db_name = db_config.get('db_name', f'Database {db_id}')
        
        return APIResponse(
            status="success",
            message=f"Found {len(all_tables)} tables in database '{db_name}' (db_id: {db_id}) across {len(user_records)} users",
            data={
                "db_id": db_id,
                "db_name": db_name,
                "tables": all_tables,
                "user_records": user_records,
                "table_count": len(all_tables),
                "user_count": len(user_records)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving tables for database {db_id}: {str(e)}"
        )


@router.get("/setup-tracking-table", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def setup_tracking_table():
    """
    Setup the usercreatedtable tracking table.
    This endpoint creates the table if it doesn't exist.
    """
    try:
        create_user_created_table_tracking_table()
        return APIResponse(
            status="success",
            message="User created table tracking table setup completed",
            data={"table_name": "usercreatedtable"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error setting up tracking table: {str(e)}"
        )


# ============================================================================
# BUSINESS RULE MANAGEMENT ENDPOINTS (USER_ID ONLY)
# ============================================================================

class UserBusinessRuleUpdate(BaseModel):
    business_rule: str = Field(..., description="Business rule for the user's current database")

@router.get("/user-business-rule/{user_id}", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def get_user_business_rule(user_id: str):
    """
    Get the business rule for a user's current database.
    
    - **user_id**: User ID
    
    Returns the business rule associated with the user's current database.
    Automatically uses the user's current database selection.
    """
    try:
        # Get user's current database
        user_current_db = db_manager.get_user_current_db_details(user_id)
        if not user_current_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No current database found for user '{user_id}'. Please set a current database first."
            )
        
        current_db_id = user_current_db.get('db_id')
        if not current_db_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No database ID found for user '{user_id}'"
            )
        
        # Get the business rule
        business_rule = db_manager.get_user_business_rule(user_id, current_db_id)
        
        # Get database name for better response
        db_config = db_manager.get_mssql_config(current_db_id)
        db_name = db_config.get('db_name', f'Database {current_db_id}') if db_config else f'Database {current_db_id}'
        
        if business_rule is None:
            # No record exists, return empty business rule
            return APIResponse(
                status="success",
                message=f"No business rule found for user '{user_id}' in current database '{db_name}'",
                data={
                    "user_id": user_id,
                    "db_id": current_db_id,
                    "db_name": db_name,
                    "business_rule": "",
                    "exists": False
                }
            )
        
        return APIResponse(
            status="success",
            message=f"Business rule retrieved successfully for user '{user_id}' in current database '{db_name}'",
            data={
                "user_id": user_id,
                "db_id": current_db_id,
                "db_name": db_name,
                "business_rule": business_rule,
                "exists": True
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting user business rule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting user business rule: {str(e)}"
        )

@router.put("/user-business-rule/{user_id}", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def update_user_business_rule(user_id: str, update_data: UserBusinessRuleUpdate):
    """
    Update the business rule for a user's current database.
    
    - **user_id**: User ID
    - **business_rule**: New business rule content
    
    Updates the business rule for the user's current database selection.
    Automatically uses the user's current database.
    """
    try:
        # Get user's current database
        user_current_db = db_manager.get_user_current_db_details(user_id)
        if not user_current_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No current database found for user '{user_id}'. Please set a current database first."
            )
        
        current_db_id = user_current_db.get('db_id')
        if not current_db_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No database ID found for user '{user_id}'"
            )
        
        # Verify the database exists
        db_config = db_manager.get_mssql_config(current_db_id)
        if not db_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Database with ID {current_db_id} not found"
            )
        
        # Update the business rule
        result = db_manager.update_user_business_rule(user_id, current_db_id, update_data.business_rule)
        
        # Trigger the update config workflow with empty fields except user_id
        task_id = None
        update_config_triggered = False
        
        try:
            # Create an empty update model (all fields empty except user_id)
            update_model = MSSQLDBConfigUpdate()
            
            # Generate proper task_id using database manager
            task_id = db_manager.create_table_info_task(user_id, current_db_id)
            
            # Start the update config workflow in background
            # This will trigger table info generation, schema generation, and matched tables generation
            asyncio.create_task(
                run_update_config_workflow(
                    task_id=task_id,
                    db_id=current_db_id,
                    update_model=update_model,
                    file_bytes=None,
                    original_filename=None,
                )
            )
            
            update_config_triggered = True
            print(f"✅ Business rule updated and update config workflow triggered for user '{user_id}' in database '{db_config.get('db_name', f'ID {current_db_id}')}' with task_id: {task_id}")
            
        except Exception as workflow_error:
            # Log the error but don't fail the business rule update
            print(f"⚠️ Warning: Failed to trigger update config workflow for user '{user_id}': {workflow_error}")
        
        # Prepare response data
        response_data = {
            "user_id": user_id,
            "db_id": current_db_id,
            "db_name": db_config.get('db_name'),
            "business_rule": update_data.business_rule,
            "updated_at": result.get('updated_at'),
            "update_config_triggered": update_config_triggered
        }
        
        # Add task tracking information if workflow was triggered
        if task_id and update_config_triggered:
            response_data.update({
                "task_id": task_id,
                "task_status_endpoint": f"/mssql-config/tasks/{task_id}"
            })
        
        return APIResponse(
            status="success",
            message=f"Business rule updated successfully for user '{user_id}' in current database '{db_config.get('db_name', f'ID {current_db_id}')}'. Update config workflow triggered." if update_config_triggered else f"Business rule updated successfully for user '{user_id}' in current database '{db_config.get('db_name', f'ID {current_db_id}')}'.",
            data=response_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error updating user business rule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating user business rule: {str(e)}"
        )

# Health check for this router
@router.get("/health", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def new_table_creation_health_check():
    return APIResponse(status="success", message="New Table service is healthy", data={"service": "new-table"})


