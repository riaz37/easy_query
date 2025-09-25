#!/usr/bin/env python3
"""
Test script to get column names for a specific table from user's current database.
"""

import json
import sys
import os
from typing import List, Optional

# Import the database manager
import sys
sys.path.append('db_manager')
from mssql_config import db_manager

def get_column_names(user_id: str, table_full_name: str, base_url: str = None) -> Optional[List[str]]:
    """
    Get column names for a specific table from user's current database.
    
    Args:
        user_id (str): User ID
        table_full_name (str): Full table name (e.g., "dbo.expenseItems")
        base_url (str): Base URL of the API (ignored when using direct function calls)
        
    Returns:
        Optional[List[str]]: List of column names if found, None if not found or error
    """
    try:
        print(f"🔍 Getting user current database details for user: {user_id}")
        
        # Call the database manager function directly
        user_data = db_manager.get_user_current_db_details(user_id)
        
        if user_data:
            print("✅ Successfully retrieved user current database data")
            
            db_schema = user_data.get('db_schema', {})
            
            if db_schema and isinstance(db_schema, dict):
                matched_tables_details = db_schema.get('matched_tables_details', [])
                
                if matched_tables_details:
                    print(f"📋 Found {len(matched_tables_details)} matched tables")
                    
                    # Search for the specified table
                    for table_detail in matched_tables_details:
                        if isinstance(table_detail, dict):
                            current_full_name = table_detail.get('full_name', '')
                            
                            if current_full_name == table_full_name:
                                print(f"✅ Found table: {table_full_name}")
                                
                                # Extract column names
                                columns = table_detail.get('columns', [])
                                if columns:
                                    column_names = [col.get('name', '') for col in columns if col.get('name')]
                                    print(f"📊 Found {len(column_names)} columns")
                                    return column_names
                                else:
                                    print("❌ No columns found for this table")
                                    return None
                    
                    # If we get here, the table was not found
                    print(f"❌ Table '{table_full_name}' not found in matched tables")
                    print("Available tables:")
                    for table_detail in matched_tables_details:
                        if isinstance(table_detail, dict):
                            full_name = table_detail.get('full_name', 'N/A')
                            print(f"  - {full_name}")
                    return None
                else:
                    print("❌ No matched tables found in db_schema")
                    return None
            else:
                print("❌ No db_schema found or it's not a dictionary")
                return None
        else:
            print("❌ No user current database data found")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def get_column_names_with_details(user_id: str, table_full_name: str, base_url: str = None) -> Optional[List[dict]]:
    """
    Get column details (name, type, etc.) for a specific table from user's current database.
    
    Args:
        user_id (str): User ID
        table_full_name (str): Full table name (e.g., "dbo.expenseItems")
        base_url (str): Base URL of the API (ignored when using direct function calls)
        
    Returns:
        Optional[List[dict]]: List of column details if found, None if not found or error
    """
    try:
        print(f"🔍 Getting user current database details for user: {user_id}")
        
        # Call the database manager function directly
        user_data = db_manager.get_user_current_db_details(user_id)
        
        if user_data:
            print("✅ Successfully retrieved user current database data")
            
            db_schema = user_data.get('db_schema', {})
            
            if db_schema and isinstance(db_schema, dict):
                matched_tables_details = db_schema.get('matched_tables_details', [])
                
                if matched_tables_details:
                    print(f"📋 Found {len(matched_tables_details)} matched tables")
                    
                    # Search for the specified table
                    for table_detail in matched_tables_details:
                        if isinstance(table_detail, dict):
                            current_full_name = table_detail.get('full_name', '')
                            
                            if current_full_name == table_full_name:
                                print(f"✅ Found table: {table_full_name}")
                                
                                # Return column details
                                columns = table_detail.get('columns', [])
                                if columns:
                                    print(f"📊 Found {len(columns)} columns")
                                    return columns
                                else:
                                    print("❌ No columns found for this table")
                                    return None
                    
                    # If we get here, the table was not found
                    print(f"❌ Table '{table_full_name}' not found in matched tables")
                    print("Available tables:")
                    for table_detail in matched_tables_details:
                        if isinstance(table_detail, dict):
                            full_name = table_detail.get('full_name', 'N/A')
                            print(f"  - {full_name}")
                    return None
                else:
                    print("❌ No matched tables found in db_schema")
                    return None
            else:
                print("❌ No db_schema found or it's not a dictionary")
                return None
        else:
            print("❌ No user current database data found")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def main():
    """Main function to run the test."""

    
    user_id = "nilab"
    table_full_name = "dbo.expenseItems"
    base_url = "https://127.0.0.1:8200/"
    print("🧪 Getting Column Names for Table")
    print("=" * 50)
    print(f"User ID: {user_id}")
    print(f"Table Full Name: {table_full_name}")
    print("Using direct database manager function calls (no HTTP requests)")
    print()

    # Get column names only
    print("📋 Getting column names only:")
    column_names = get_column_names(user_id, table_full_name, base_url)
    
    if column_names:
        print("✅ Column names:")
        for i, name in enumerate(column_names, 1):
            print(f"  {i}. {name}")
    else:
        print("❌ Failed to get column names")
    
    print()
    
    # Get column details
    print("📊 Getting column details:")
    column_details = get_column_names_with_details(user_id, table_full_name, base_url)
    
    if column_details:
        print("✅ Column details:")
        for i, col in enumerate(column_details, 1):
            name = col.get('name', 'N/A')
            col_type = col.get('type', 'N/A')
            is_primary = col.get('is_primary', False)
            is_foreign = col.get('is_foreign', False)
            is_required = col.get('is_required', False)
            
            print(f"  {i}. {name} ({col_type})")
            print(f"     Primary: {is_primary}, Foreign: {is_foreign}, Required: {is_required}")
    else:
        print("❌ Failed to get column details")

if __name__ == "__main__":
    main()
