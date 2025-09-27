import os
import sys
import json
import pandas as pd
import requests
import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO
from dotenv import load_dotenv
import time
from datetime import datetime, date

# Environment Configuration
# Set BASE_URL environment variable to override the default API URL
# Example: export BASE_URL=https://127.0.0.1:8200
# Default: http://localhost:8000

# Plotly imports - Much more beautiful and modern than Bokeh
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import plotly.io as pio
import kaleido  # For image export

# FastAPI imports
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# LLM imports
from langchain_google_genai import ChatGoogleGenerativeAI

# Database imports
import psycopg2
from psycopg2.extras import RealDictCursor

# Add the project root to the path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)

# Import the query function directly from mssql_agent3
try:
    from data_sources.mssql.mssql_agent3 import query_database
    DIRECT_QUERY_AVAILABLE = True
except ImportError:
    print("Warning: Could not import query_database directly, will use HTTP fallback")
    DIRECT_QUERY_AVAILABLE = False

# Import graph generation functions directly from graph_Generator.py
try:
    from Report_generator.utilites.graph_Generator import generate_beautiful_graph, generate_multiple_graphs
    DIRECT_GRAPH_AVAILABLE = True
    print("✅ Successfully imported direct graph generation functions")
except ImportError:
    print("Warning: Could not import graph generation functions directly, will use HTTP fallback")
    DIRECT_GRAPH_AVAILABLE = False

# Import database manager for direct function calls
try:
    from db_manager.mssql_config import db_manager
    DIRECT_DB_AVAILABLE = True
    print("✅ Successfully imported database manager for direct function calls")
except ImportError:
    print("Warning: Could not import database manager, will use HTTP fallback")
    DIRECT_DB_AVAILABLE = False

# Initialize environment and LLM
load_dotenv(override=True)
gemini_apikey = os.getenv("google_api_key")
if gemini_apikey is None:
    print("Warning: 'google_api_key' not found in environment variables.")

gemini_model_name = os.getenv("google_gemini_name", "gemini-1.5-pro")

def initialize_llm_gemini(api_key: str = gemini_apikey, temperature: int = 0, model: str = gemini_model_name) -> ChatGoogleGenerativeAI:
    """Initialize and return the ChatGoogleGenerativeAI LLM instance."""
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)

llm = initialize_llm_gemini()

# Database connection configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'user': 'postgres',
    'password': '1234',
    'database': 'main_db'
}

def get_database_connection():
    """Create and return a database connection."""
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        raise Exception(f"Database connection failed: {str(e)}")

async def get_db_id_from_user_id(user_id: str) -> int:
    """
    Fetch the current database ID for a user using direct database calls or HTTP endpoint.
    
    Args:
        user_id (str): User ID to get database ID for
        
    Returns:
        int: Database ID for the user
        
    Raises:
        Exception: If user has no current database or API call fails
    """
    try:
        print(f"🔍 Fetching database ID for user: {user_id}")
        
        # Try direct database manager call first
        if DIRECT_DB_AVAILABLE:
            print("🎯 Using direct database manager call (no HTTP overhead)")
            try:
                user_data = db_manager.get_user_current_db_details(user_id)
                if user_data and user_data.get('db_id'):
                    db_id = user_data['db_id']
                    print(f"✅ Found database ID {db_id} for user {user_id} via direct call")
                    return db_id
                else:
                    raise Exception(f"No current database found for user {user_id}")
            except Exception as e:
                print(f"❌ Direct database call failed: {e}")
                print("   Falling back to HTTP call...")
        
        # Fallback to HTTP call if direct call is not available or failed
        print("🌐 Using HTTP API call")
        import aiohttp
        
        # Load base URL from environment variable
        base_url = os.getenv("BASE_URL", "http://localhost:8000")
        print(f"🌐 Using base URL: {base_url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/mssql-config/user-current-db/{user_id}",
                timeout=aiohttp.ClientTimeout(total=30),
                ssl=False  # Disable SSL verification for self-signed certificates
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "success" and data.get("data"):
                        db_id = data["data"].get("db_id")
                        if db_id:
                            print(f"✅ Found database ID {db_id} for user {user_id} via HTTP")
                            return db_id
                        else:
                            raise Exception(f"No database ID found in response for user {user_id}")
                    else:
                        error_msg = data.get("message", "Unknown error")
                        raise Exception(f"API returned error: {error_msg}")
                elif response.status == 404:
                    raise Exception(f"No current database found for user {user_id}")
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
    except Exception as e:
        print(f"❌ Error fetching database ID for user {user_id}: {e}")
        raise Exception(f"Failed to fetch database ID for user {user_id}: {str(e)}")

def get_db_id_from_user_id_sync(user_id: str) -> int:
    """
    Synchronous version of get_db_id_from_user_id.
    
    Args:
        user_id (str): User ID to get database ID for
        
    Returns:
        int: Database ID for the user
        
    Raises:
        Exception: If user has no current database or API call fails
    """
    return asyncio.run(get_db_id_from_user_id(user_id))

def get_report_structure_from_db(db_id: int = 1) -> str:
    """
    Fetch report_structure from the database for a given db_id.
    
    Args:
        db_id (int): Database ID (default: 1)
        
    Returns:
        str: Report structure content from database
        
    Raises:
        Exception: If database connection fails or record not found
    """
    try:
        print(f"\n🔍 DEBUG: Fetching report_structure from database ID {db_id}")
        
        # Try direct database manager call first
        if DIRECT_DB_AVAILABLE:
            print("🎯 Using direct database manager call (no SQL overhead)")
            try:
                db_config = db_manager.get_mssql_config(db_id)
                if db_config and db_config.get('report_structure'):
                    content = db_config['report_structure']
                    print(f"✅ Content retrieved successfully via direct call")
                    print(f"   Content length: {len(content)} characters")
                    print(f"   Content type: {type(content)}")
                    
                    # Show first 500 characters as preview
                    preview = content[:500] + "..." if len(content) > 500 else content
                    print(f"   Content preview:")
                    print(f"   {'-'*50}")
                    print(f"   {preview}")
                    print(f"   {'-'*50}")
                    
                    return content
                else:
                    print(f"❌ No report_structure found for database ID {db_id} via direct call")
                    raise Exception(f"Report structure not found for database ID {db_id}")
            except Exception as e:
                print(f"❌ Direct database call failed: {e}")
                print("   Falling back to SQL query...")
        
        # Fallback to direct SQL query
        print("🗄️ Using direct SQL query")
        conn = get_database_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT report_structure FROM mssql_config WHERE db_id = %s",
                (db_id,)
            )
            result = cursor.fetchone()
        conn.close()
        
        print(f"✅ Database query completed")
        print(f"   Result found: {result is not None}")
        
        if result and result['report_structure']:
            content = result['report_structure']
            print(f"   Content retrieved successfully")
            print(f"   Content length: {len(content)} characters")
            print(f"   Content type: {type(content)}")
            
            # Show first 500 characters as preview
            preview = content[:500] + "..." if len(content) > 500 else content
            print(f"   Content preview:")
            print(f"   {'-'*50}")
            print(f"   {preview}")
            print(f"   {'-'*50}")
            
            return content
        else:
            print(f"❌ No report_structure found for database ID {db_id}")
            raise Exception(f"Report structure not found for database ID {db_id}")
            
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        raise Exception(f"Database error: {str(e)}")
    except Exception as e:
        print(f"❌ Error fetching report structure: {e}")
        raise Exception(f"Failed to fetch report structure: {str(e)}")

def get_report_structure_from_user(user_id: str) -> str:
    """
    Fetch report_structure from the database for a given user_id.
    
    Args:
        user_id (str): User ID
        
    Returns:
        str: Report structure content from database
        
    Raises:
        Exception: If user has no current database, database connection fails, or record not found
    """
    try:
        # Get the database ID for the user
        db_id = get_db_id_from_user_id_sync(user_id)
        
        # Fetch the report structure using the database ID
        return get_report_structure_from_db(db_id)
        
    except Exception as e:
        print(f"❌ Error fetching report structure for user {user_id}: {e}")
        raise Exception(f"Failed to fetch report structure for user {user_id}: {str(e)}")

async def get_report_structure_from_user_async(user_id: str) -> str:
    """
    Async version of get_report_structure_from_user.
    
    Args:
        user_id (str): User ID
        
    Returns:
        str: Report structure content from database
        
    Raises:
        Exception: If user has no current database, database connection fails, or record not found
    """
    try:
        # Get the database ID for the user
        db_id = await get_db_id_from_user_id(user_id)
        
        # Fetch the report structure using the database ID
        return get_report_structure_from_db(db_id)
        
    except Exception as e:
        print(f"❌ Error fetching report structure for user {user_id}: {e}")
        raise Exception(f"Failed to fetch report structure for user {user_id}: {str(e)}")

def parse_report_structure_with_llm(report_structure_content: str, user_query: str = None) -> Dict[str, Any]:
    """
    Parse the report structure content using LLM to extract sections and queries.
    If user_query is provided, extract ALL queries but customize them based on the user's request.
    
    Args:
        report_structure_content (str): Report structure content from database
        user_query (str, optional): User's specific request (e.g., "financial report", "attendance summary")
        
    Returns:
        Dict[str, Any]: Structured data containing sections and their queries
        Format:
        {
            "sections": [
                {
                    "section_number": 1,
                    "section_name": "Employee Daily Attendance Summary for any Month",
                    "queries": [
                        {
                            "query_number": 1,
                            "query": "Get employee-wise punch count for March. 2024."
                        },
                        {
                            "query_number": 2,
                            "query": "Who are the users absent today on August 17, 2024..."
                        }
                    ]
                }
            ]
        }
    """
    
    # DEBUG: Print what is being sent to LLM
    print("\n" + "="*80)
    print("🔍 DEBUG: Content being sent to LLM from database:")
    print("="*80)
    if user_query:
        print(f"🎯 USER QUERY: {user_query}")
        print(f"📋 MODE: Extract ALL queries with customization based on user request")
    else:
        print(f"📋 MODE: Extract all queries (backward compatibility)")
    print(f"Content length: {len(report_structure_content)} characters")
    print(f"Content type: {type(report_structure_content)}")
    print("\n📄 Full content:")
    print("-"*80)
    print(report_structure_content)
    print("-"*80)
    print("="*80)
    
    # Create a context-aware prompt for the LLM
    if user_query:
        # User has provided a specific query - extract ALL queries but customize them based on user request
        prompt = f"""
        USER REQUEST: {user_query}
        
        Please analyze the following report structure content and extract ALL sections and queries, but customize the queries based on the user's request.
        
        Report Structure Content:
        {report_structure_content}
        
        Based on the user's request "{user_query}", please:
        1. Extract ALL sections and queries from the content
        2. If the user mentions specific names, dates, or values, incorporate them into the queries where appropriate
        3. If user mentions specific details, update the queries to include those details
        4. Keep all original queries but modify them to match the user's specific request
        
        Please return a JSON object with the following structure:
        {{
            "sections": [
                {{
                    "section_number": <number>,
                    "section_name": "<section name>",
                    "queries": [
                        {{
                            "query_number": <number>,
                            "query": "<query text - customized based on user request>"
                        }}
                    ]
                }}
            ]
        }}
        
        Rules:
        1. Extract section numbers from "### Section X:" or "### Section X:"
        2. Extract section names (the text after the colon)
        3. Extract ALL queries marked with "**Query:**" or "** Query:**"
        4. If user mentions specific details (names, dates, etc.), incorporate them into the queries
        5. DO NOT filter out any queries - include ALL queries from ALL sections
        6. If user asks for "financial report", customize queries to focus on financial data but keep all queries
        7. If user asks for "attendance report", customize queries to focus on attendance data but keep all queries
        8. If user asks for "general report", keep all queries as they are
        9. Number queries sequentially within each section
        10. Clean up any extra whitespace or formatting
        11. Return only valid JSON, no additional text
        
        IMPORTANT: Extract ALL queries from ALL sections, do not filter based on relevance.
        
        Please return the JSON structure:
        """
    else:
        # No user query - extract all content (backward compatibility)
        prompt = f"""
        Please parse the following report structure content and extract sections and queries in a structured format.
        
        Content:
        {report_structure_content}
        
        Please analyze this content and return a JSON object with the following structure:
        {{
            "sections": [
                {{
                    "section_number": <number>,
                    "section_name": "<section name>",
                    "queries": [
                        {{
                            "query_number": <number>,
                            "query": "<query text>"
                        }}
                    ]
                }}
            ]
        }}
        
        Rules:
        1. Extract section numbers from "### Section X:" or "### Section X:"
        2. Extract section names (the text after the colon)
        3. Extract all queries marked with "**Query:**" or "** Query:**"
        4. Number queries sequentially within each section
        5. Clean up any extra whitespace or formatting
        6. Return only valid JSON, no additional text
        
        Please return the JSON structure:
        """
    
    # DEBUG: Print the prompt being sent to LLM
    print("\n" + "="*80)
    print("🤖 DEBUG: Prompt being sent to LLM:")
    print("="*80)
    print(prompt)
    print("="*80)
    
    try:
        # Get response from LLM
        response = llm.invoke(prompt)
        response_text = response.content
        
        # DEBUG: Print LLM response
        print("\n" + "="*80)
        print("🤖 DEBUG: LLM Response:")
        print("="*80)
        print(response_text)
        print("="*80)
        
        # Extract JSON from response (in case LLM adds extra text)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed_data = json.loads(json_str)
            
            # DEBUG: Print parsed data
            print("\n" + "="*80)
            print("✅ DEBUG: Parsed JSON Data:")
            print("="*80)
            print(json.dumps(parsed_data, indent=2))
            print("="*80)
            
            return parsed_data
        else:
            # If no JSON found, try to parse the entire response
            parsed_data = json.loads(response_text)
            
            # DEBUG: Print parsed data
            print("\n" + "="*80)
            print("✅ DEBUG: Parsed JSON Data (from full response):")
            print("="*80)
            print(json.dumps(parsed_data, indent=2))
            print("="*80)
            
            return parsed_data
            
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing LLM response as JSON: {e}")
        print(f"Raw response: {response_text}")
        raise Exception("Failed to parse LLM response as valid JSON")
    except Exception as e:
        print(f"❌ Error getting response from LLM: {e}")
        raise Exception(f"LLM processing failed: {str(e)}")

# Keep the old function for backward compatibility
def parse_testing_file_with_llm(file_path: str = "Report_generator/utilites/testing.txt") -> Dict[str, Any]:
    """
    Parse the testing.txt file using LLM to extract sections and queries.
    This function is kept for backward compatibility.
    
    Args:
        file_path (str): Path to the testing.txt file
        
    Returns:
        Dict[str, Any]: Structured data containing sections and their queries
    """
    
    # Read the file content
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")
    
    # Use the new parsing function
    return parse_report_structure_with_llm(content)

def get_sections_and_queries_from_db(db_id: int = 1) -> str:
    """
    Get sections and queries from the database and return formatted string.
    
    Args:
        db_id (int): Database ID (default: 1)
        
    Returns:
        str: Formatted string with sections and queries
    """
    try:
        # Fetch report structure from database
        report_structure_content = get_report_structure_from_db(db_id)
        
        # Parse the content
        parsed_data = parse_report_structure_with_llm(report_structure_content)
        
        # Format the output as requested
        output_lines = []
        
        for section in parsed_data.get("sections", []):
            section_num = section.get("section_number", "Unknown")
            section_name = section.get("section_name", "Unknown Section")
            
            output_lines.append(f"Section {section_num}: {section_name}")
            
            for query in section.get("queries", []):
                query_num = query.get("query_number", "Unknown")
                query_text = query.get("query", "No query text")
                output_lines.append(f"  Query {query_num}: {query_text}")
            
            output_lines.append("")  # Add empty line between sections
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"Error processing database content: {str(e)}"

def get_sections_and_queries_dict_from_db(db_id: int = 1, user_query: str = None) -> Dict[str, Any]:
    """
    Get sections and queries from the database and return as dictionary.
    If user_query is provided, return ALL queries but customize them based on the user's request.
    
    Args:
        db_id (int): Database ID (default: 1)
        user_query (str, optional): User's specific request (e.g., "financial report", "attendance summary")
        
    Returns:
        Dict[str, Any]: Structured data with sections and queries
    """
    try:
        # Fetch report structure from database
        report_structure_content = get_report_structure_from_db(db_id)
        
        # Parse the content with user query if provided
        return parse_report_structure_with_llm(report_structure_content, user_query)
        
    except Exception as e:
        print(f"Error getting sections and queries from database: {e}")
        return {"sections": [], "error": str(e)}

def get_sections_and_queries_from_user(user_id: str) -> str:
    """
    Get sections and queries from the database for a user and return formatted string.
    
    Args:
        user_id (str): User ID
        
    Returns:
        str: Formatted string with sections and queries
    """
    try:
        # Fetch report structure from database for the user
        report_structure_content = get_report_structure_from_user(user_id)
        
        # Parse the content
        parsed_data = parse_report_structure_with_llm(report_structure_content)
        
        # Format the output as requested
        output_lines = []
        
        for section in parsed_data.get("sections", []):
            section_num = section.get("section_number", "Unknown")
            section_name = section.get("section_name", "Unknown Section")
            
            output_lines.append(f"Section {section_num}: {section_name}")
            
            for query in section.get("queries", []):
                query_num = query.get("query_number", "Unknown")
                query_text = query.get("query", "No query text")
                output_lines.append(f"  Query {query_num}: {query_text}")
            
            output_lines.append("")  # Add empty line between sections
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"Error processing database content for user {user_id}: {str(e)}"

def get_sections_and_queries_dict_from_user(user_id: str, user_query: str = None) -> Dict[str, Any]:
    """
    Get sections and queries from the database for a user and return as dictionary.
    If user_query is provided, return ALL queries but customize them based on the user's request.
    
    Args:
        user_id (str): User ID
        user_query (str, optional): User's specific request (e.g., "financial report", "attendance summary")
        
    Returns:
        Dict[str, Any]: Structured data with sections and queries
    """
    try:
        # Fetch report structure from database for the user
        report_structure_content = get_report_structure_from_user(user_id)
        
        # Parse the content with user query if provided
        return parse_report_structure_with_llm(report_structure_content, user_query)
        
    except Exception as e:
        print(f"Error getting sections and queries from database for user {user_id}: {e}")
        return {"sections": [], "error": str(e)}

async def get_sections_and_queries_dict_from_user_async(user_id: str, user_query: str = None) -> Dict[str, Any]:
    """
    Async version of get_sections_and_queries_dict_from_user.
    
    Args:
        user_id (str): User ID
        user_query (str, optional): User's specific request (e.g., "financial report", "attendance summary")
        
    Returns:
        Dict[str, Any]: Structured data with sections and queries (ALL queries, customized based on user request)
    """
    try:
        # Fetch report structure from database for the user
        report_structure_content = await get_report_structure_from_user_async(user_id)
        
        # Parse the content with user query if provided
        return parse_report_structure_with_llm(report_structure_content, user_query)
        
    except Exception as e:
        print(f"Error getting sections and queries from database for user {user_id}: {e}")
        return {"sections": [], "error": str(e)}

async def process_all_queries_with_graph_generation(
    db_id: int = 1,
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string"
) -> Dict[str, Any]:
    """
    Process all queries from the database by calling the generate-graph function directly.
    
    Args:
        db_id (int): Database ID to fetch report structure from (default: 1)
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis
        
    Returns:
        Dict[str, Any]: Results for all processed queries
    """
    
    try:
        print(f"🚀 Starting batch processing of all queries from database ID {db_id}")
        
        # Step 1: Parse the database content
        print("📖 Parsing database content...")
        sections_data = get_sections_and_queries_dict_from_db(db_id)
        
        if not sections_data or "sections" not in sections_data:
            raise Exception("Failed to parse database content or no sections found")
        
        # Step 2: Extract all queries
        all_queries = []
        for section in sections_data["sections"]:
            section_num = section.get("section_number", 0)
            section_name = section.get("section_name", "Unknown Section")
            
            for query in section.get("queries", []):
                query_num = query.get("query_number", 0)
                query_text = query.get("query", "")
                
                all_queries.append({
                    "section_number": section_num,
                    "section_name": section_name,
                    "query_number": query_num,
                    "query": query_text
                })
        
        print(f"📊 Found {len(all_queries)} queries to process")
        
        # Step 3: Process each query using direct function calls
        results = []
        successful_count = 0
        failed_count = 0
        
        if DIRECT_GRAPH_AVAILABLE:
            print("🎯 Using direct graph generation functions (no HTTP overhead)")
            
            async def process_single_query_direct(query_info: Dict[str, Any]) -> Dict[str, Any]:
                """Process a single query using direct function call."""
                query_start_time = time.time()
                
                try:
                    # Call the direct graph generation function
                    graph_result = await generate_beautiful_graph(
                        query=query_info["query"],
                        user_id=user_id,
                        export_format=export_format,
                        theme=theme
                    )
                    
                    query_processing_time = time.time() - query_start_time
                    
                    if graph_result.get("success"):
                        # Serialize the graph_result to handle date objects
                        serialized_graph_result = serialize_for_json(graph_result)
                        
                        # Get the data for pagination and analysis
                        data = serialized_graph_result.get("data", {}).get("sample", [])
                        
                        # Create paginated table
                        table = create_paginated_table(data, page_size=10)
                        
                        # Create graph analysis with LLM data analysis
                        graph_analysis = create_graph_analysis(graph_result, data, query_info["query"])
                        
                        return {
                            "section_number": query_info["section_number"],
                            "section_name": query_info["section_name"],
                            "query_number": query_info["query_number"],
                            "query": query_info["query"],
                            "success": True,
                            "table": table,
                            "graph_and_analysis": graph_analysis
                        }
                    else:
                        return {
                            "section_number": query_info["section_number"],
                            "section_name": query_info["section_name"],
                            "query_number": query_info["query_number"],
                            "query": query_info["query"],
                            "success": False,
                            "error": str(graph_result.get("error", "Unknown error"))
                        }
                        
                except Exception as e:
                    query_processing_time = time.time() - query_start_time
                    return {
                        "section_number": query_info["section_number"],
                        "section_name": query_info["section_name"],
                        "query_number": query_info["query_number"],
                        "query": query_info["query"],
                        "success": False,
                        "error": str(e)
                    }
            
            # Process queries with limited concurrency to avoid overwhelming the system
            semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent requests
            
            async def process_with_semaphore(query_info):
                async with semaphore:
                    return await process_single_query_direct(query_info)
            
            # Execute all queries
            tasks = [process_with_semaphore(query_info) for query_info in all_queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        else:
            # Fallback to HTTP API calls if direct functions are not available
            print("⚠️ Using HTTP API calls (direct functions not available)")
            import aiohttp
            
            # Load base URL from environment variable
            base_url = os.getenv("BASE_URL", "http://localhost:8000")
            print(f"🌐 Using base URL: {base_url}")
            
            async def process_single_query_http(query_info: Dict[str, Any]) -> Dict[str, Any]:
                """Process a single query by calling the generate-graph endpoint."""
                query_start_time = time.time()
                
                try:
                    # Prepare request payload
                    payload = {
                        "query": query_info["query"],
                        "user_id": user_id,
                        "export_format": export_format,
                        "theme": theme,
                        "analysis_subject": analysis_subject
                    }
                    
                    # Make async HTTP request
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{base_url}/graph/generate-graph",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
                        ) as response:
                            
                            if response.status == 200:
                                result_data = await response.json()
                                query_processing_time = time.time() - query_start_time
                                
                                # Serialize the result_data to handle date objects
                                serialized_result_data = serialize_for_json(result_data)
                                
                                return {
                                    "section_number": query_info["section_number"],
                                    "section_name": query_info["section_name"],
                                    "query_number": query_info["query_number"],
                                    "query": query_info["query"],
                                    "success": True,
                                    "files": serialized_result_data.get("files", {}),
                                    "data": serialized_result_data.get("data", {})
                                }
                            else:
                                error_text = await response.text()
                                query_processing_time = time.time() - query_start_time
                                
                                return {
                                    "section_number": query_info["section_number"],
                                    "section_name": query_info["section_name"],
                                    "query_number": query_info["query_number"],
                                    "query": query_info["query"],
                                    "success": False,
                                    "error": f"HTTP {response.status}: {error_text}"
                                }
                                
                except Exception as e:
                    query_processing_time = time.time() - query_start_time
                    return {
                        "section_number": query_info["section_number"],
                        "section_name": query_info["section_name"],
                        "query_number": query_info["query_number"],
                        "query": query_info["query"],
                        "success": False,
                        "error": str(e)
                    }
            
            # Process queries concurrently (limit to 5 concurrent requests to avoid overwhelming the server)
            semaphore = asyncio.Semaphore(5)
            
            async def process_with_semaphore(query_info):
                async with semaphore:
                    return await process_single_query_http(query_info)
            
            # Execute all queries
            tasks = [process_with_semaphore(query_info) for query_info in all_queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_count = sum(1 for r in results if isinstance(r, dict) and r.get("success", False))
        failed_count = len(results) - successful_count
        
        # Generate combined HTML report
        final_html_path = generate_combined_html_report(results, user_id)
        
        # Generate summary
        total_processing_time = sum(r.get("processing_time", 0) for r in results if isinstance(r, dict))
        
        summary = {
            "total_sections": len(sections_data["sections"]),
            "total_queries": len(all_queries),
            "successful_queries": successful_count,
            "failed_queries": failed_count,
            "success_rate": (successful_count / len(all_queries)) * 100 if all_queries else 0,
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(all_queries) if all_queries else 0,
            "sections_processed": len(set(r.get("section_number", 0) for r in results if isinstance(r, dict))),
            "processing_method": "direct_function" if DIRECT_GRAPH_AVAILABLE else "http_api",
            "database_id": db_id,
            "errors_summary": {}
        }
        
        # Group errors by type
        for result in results:
            if isinstance(result, dict) and result.get("error"):
                error_type = result["error"].split(":")[0] if ":" in result["error"] else "Unknown"
                summary["errors_summary"][error_type] = summary["errors_summary"].get(error_type, 0) + 1
        
        print(f"🎉 Batch processing completed!")
        print(f"   Database ID: {db_id}")
        print(f"   Total queries: {len(all_queries)}")
        print(f"   Successful: {successful_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Total time: {total_processing_time:.2f}s")
        print(f"   Method: {'Direct function calls' if DIRECT_GRAPH_AVAILABLE else 'HTTP API calls'}")
        print(f"   Final HTML: {final_html_path}")
        
        return {
            "success": True,
            "database_id": db_id,
            "total_queries": len(all_queries),
            "successful_queries": successful_count,
            "failed_queries": failed_count,
            "results": results,
            "total_processing_time": total_processing_time,
            "summary": summary,
            "final_html": final_html_path
        }
        
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        return {
            "success": False,
            "error": str(e),
            "database_id": db_id,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "total_processing_time": 0,
            "summary": {}
        }

# Keep the old function for backward compatibility
async def process_all_queries_with_graph_generation_from_file(
    file_path: str = "Report_generator/utilites/testing.txt",
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string"
) -> Dict[str, Any]:
    """
    Process all queries from the testing file by calling the generate-graph function directly.
    This function is kept for backward compatibility.
    
    Args:
        file_path (str): Path to the testing.txt file
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis
        
    Returns:
        Dict[str, Any]: Results for all processed queries
    """
    
    try:
        print(f"🚀 Starting batch processing of all queries from {file_path}")
        
        # Step 1: Parse the testing file
        print("📖 Parsing testing file...")
        sections_data = get_sections_and_queries_dict_from_db(file_path)
        
        if not sections_data or "sections" not in sections_data:
            raise Exception("Failed to parse testing file or no sections found")
        
        # Step 2: Extract all queries
        all_queries = []
        for section in sections_data["sections"]:
            section_num = section.get("section_number", 0)
            section_name = section.get("section_name", "Unknown Section")
            
            for query in section.get("queries", []):
                query_num = query.get("query_number", 0)
                query_text = query.get("query", "")
                
                all_queries.append({
                    "section_number": section_num,
                    "section_name": section_name,
                    "query_number": query_num,
                    "query": query_text
                })
        
        print(f"📊 Found {len(all_queries)} queries to process")
        
        # Step 3: Process each query using direct function calls
        results = []
        successful_count = 0
        failed_count = 0
        
        if DIRECT_GRAPH_AVAILABLE:
            print("🎯 Using direct graph generation functions (no HTTP overhead)")
            
            async def process_single_query_direct(query_info: Dict[str, Any]) -> Dict[str, Any]:
                """Process a single query using direct function call."""
                query_start_time = time.time()
                
                try:
                    # Call the direct graph generation function
                    graph_result = await generate_beautiful_graph(
                        query=query_info["query"],
                        user_id=user_id,
                        export_format=export_format,
                        theme=theme
                    )
                    
                    query_processing_time = time.time() - query_start_time
                    
                    if graph_result.get("success"):
                        # Serialize the graph_result to handle date objects
                        serialized_graph_result = serialize_for_json(graph_result)
                        
                        # Get the data for pagination and analysis
                        data = serialized_graph_result.get("data", {}).get("sample", [])
                        
                        # Create paginated table
                        table = create_paginated_table(data, page_size=10)
                        
                        # Create graph analysis with LLM data analysis
                        graph_analysis = create_graph_analysis(graph_result, data, query_info["query"])
                        
                        return {
                            "section_number": query_info["section_number"],
                            "section_name": query_info["section_name"],
                            "query_number": query_info["query_number"],
                            "query": query_info["query"],
                            "success": True,
                            "table": table,
                            "graph_and_analysis": graph_analysis
                        }
                    else:
                        return {
                            "section_number": query_info["section_number"],
                            "section_name": query_info["section_name"],
                            "query_number": query_info["query_number"],
                            "query": query_info["query"],
                            "success": False,
                            "error": str(graph_result.get("error", "Unknown error"))
                        }
                        
                except Exception as e:
                    query_processing_time = time.time() - query_start_time
                    return {
                        "section_number": query_info["section_number"],
                        "section_name": query_info["section_name"],
                        "query_number": query_info["query_number"],
                        "query": query_info["query"],
                        "success": False,
                        "error": str(e)
                    }
            
            # Process queries with limited concurrency to avoid overwhelming the system
            semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent requests
            
            async def process_with_semaphore(query_info):
                async with semaphore:
                    return await process_single_query_direct(query_info)
            
            # Execute all queries
            tasks = [process_with_semaphore(query_info) for query_info in all_queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        else:
            # Fallback to HTTP API calls if direct functions are not available
            print("⚠️ Using HTTP API calls (direct functions not available)")
            import aiohttp
            
            # Load base URL from environment variable
            base_url = os.getenv("BASE_URL", "http://localhost:8000")
            print(f"🌐 Using base URL: {base_url}")
            
            async def process_single_query_http(query_info: Dict[str, Any]) -> Dict[str, Any]:
                """Process a single query by calling the generate-graph endpoint."""
                query_start_time = time.time()
                
                try:
                    # Prepare request payload
                    payload = {
                        "query": query_info["query"],
                        "user_id": user_id,
                        "export_format": export_format,
                        "theme": theme,
                        "analysis_subject": analysis_subject
                    }
                    
                    # Make async HTTP request
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{base_url}/graph/generate-graph",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
                        ) as response:
                            
                            if response.status == 200:
                                result_data = await response.json()
                                query_processing_time = time.time() - query_start_time
                                
                                # Serialize the result_data to handle date objects
                                serialized_result_data = serialize_for_json(result_data)
                                
                                return {
                                    "section_number": query_info["section_number"],
                                    "section_name": query_info["section_name"],
                                    "query_number": query_info["query_number"],
                                    "query": query_info["query"],
                                    "success": True,
                                    "files": serialized_result_data.get("files", {}),
                                    "data": serialized_result_data.get("data", {})
                                }
                            else:
                                error_text = await response.text()
                                query_processing_time = time.time() - query_start_time
                                
                                return {
                                    "section_number": query_info["section_number"],
                                    "section_name": query_info["section_name"],
                                    "query_number": query_info["query_number"],
                                    "query": query_info["query"],
                                    "success": False,
                                    "error": f"HTTP {response.status}: {error_text}"
                                }
                                
                except Exception as e:
                    query_processing_time = time.time() - query_start_time
                    return {
                        "section_number": query_info["section_number"],
                        "section_name": query_info["section_name"],
                        "query_number": query_info["query_number"],
                        "query": query_info["query"],
                        "success": False,
                        "error": str(e)
                    }
            
            # Process queries concurrently (limit to 5 concurrent requests to avoid overwhelming the server)
            semaphore = asyncio.Semaphore(5)
            
            async def process_with_semaphore(query_info):
                async with semaphore:
                    return await process_single_query_http(query_info)
            
            # Execute all queries
            tasks = [process_with_semaphore(query_info) for query_info in all_queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_count = sum(1 for r in results if isinstance(r, dict) and r.get("success", False))
        failed_count = len(results) - successful_count
        
        # Generate combined HTML report
        final_html_path = generate_combined_html_report(results, user_id)
        
        # Generate summary
        total_processing_time = sum(r.get("processing_time", 0) for r in results if isinstance(r, dict))
        
        summary = {
            "total_sections": len(sections_data["sections"]),
            "total_queries": len(all_queries),
            "successful_queries": successful_count,
            "failed_queries": failed_count,
            "success_rate": (successful_count / len(all_queries)) * 100 if all_queries else 0,
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(all_queries) if all_queries else 0,
            "sections_processed": len(set(r.get("section_number", 0) for r in results if isinstance(r, dict))),
            "processing_method": "direct_function" if DIRECT_GRAPH_AVAILABLE else "http_api",
            "errors_summary": {}
        }
        
        # Group errors by type
        for result in results:
            if isinstance(result, dict) and result.get("error"):
                error_type = result["error"].split(":")[0] if ":" in result["error"] else "Unknown"
                summary["errors_summary"][error_type] = summary["errors_summary"].get(error_type, 0) + 1
        
        print(f"🎉 Batch processing completed!")
        print(f"   Total queries: {len(all_queries)}")
        print(f"   Successful: {successful_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Total time: {total_processing_time:.2f}s")
        print(f"   Method: {'Direct function calls' if DIRECT_GRAPH_AVAILABLE else 'HTTP API calls'}")
        print(f"   Final HTML: {final_html_path}")
        
        return {
            "success": True,
            "total_queries": len(all_queries),
            "successful_queries": successful_count,
            "failed_queries": failed_count,
            "results": results,
            "total_processing_time": total_processing_time,
            "summary": summary,
            "final_html": final_html_path
        }
        
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        return {
            "success": False,
            "error": str(e),
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "total_processing_time": 0,
            "summary": {}
        }

async def process_all_queries_with_graph_generation_optimized(
    db_id: int = 1,
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string",
    user_query: str = None
) -> Dict[str, Any]:
    """
    OPTIMIZED VERSION: Process all queries using the generate_multiple_graphs function.
    This is the most efficient method as it's specifically designed for batch processing.
    
    Args:
        db_id (int): Database ID to fetch report structure from (default: 1)
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis
        user_query (str, optional): User's specific request for filtering queries
        
    Returns:
        Dict[str, Any]: Results for all processed queries
    """
    
    try:
        if user_query:
            print(f"🚀 Starting OPTIMIZED batch processing of ALL queries with customization from database ID {db_id}")
            print(f"🎯 User Query: {user_query}")
        else:
            print(f"🚀 Starting OPTIMIZED batch processing of all queries from database ID {db_id}")
        
        # Step 1: Parse the database content
        if user_query:
            print(f"📖 Parsing database content with user query: {user_query}")
        else:
            print("📖 Parsing database content...")
        sections_data = get_sections_and_queries_dict_from_db(db_id, user_query)
        
        if not sections_data or "sections" not in sections_data:
            raise Exception("Failed to parse database content or no sections found")
        
        # Step 2: Extract all queries
        all_queries = []
        query_mapping = {}  # Map query text to section info
        
        for section in sections_data["sections"]:
            section_num = section.get("section_number", 0)
            section_name = section.get("section_name", "Unknown Section")
            
            for query in section.get("queries", []):
                query_num = query.get("query_number", 0)
                query_text = query.get("query", "")
                
                all_queries.append(query_text)
                query_mapping[query_text] = {
                    "section_number": section_num,
                    "section_name": section_name,
                    "query_number": query_num,
                    "query": query_text
                }
        
        print(f"📊 Found {len(all_queries)} queries to process")
        
        # Step 3: Use the optimized batch processing function
        if DIRECT_GRAPH_AVAILABLE:
            print("🎯 Using optimized generate_multiple_graphs function")
            start_time = time.time()
            
            # Call the optimized batch function
            graph_results = await generate_multiple_graphs(
                queries=all_queries,
                user_id=user_id,
                theme=theme,
                export_format=export_format
            )
            
            total_processing_time = time.time() - start_time
            
            # Step 4: Process results and map back to section information
            results = []
            successful_count = 0
            failed_count = 0
            
            for i, graph_result in enumerate(graph_results):
                query_text = all_queries[i]
                section_info = query_mapping[query_text]
                
                if graph_result.get("success"):
                    successful_count += 1
                    
                    # Serialize the graph_result to handle date objects
                    serialized_graph_result = serialize_for_json(graph_result)
                    
                    # Get the data for pagination and analysis
                    data = serialized_graph_result.get("data", {}).get("sample", [])
                    
                    # Create paginated table
                    table = create_paginated_table(data, page_size=10)
                    
                    # Create graph analysis with LLM data analysis
                    graph_analysis = create_graph_analysis(graph_result, data, section_info["query"])
                    
                    results.append({
                        "section_number": section_info["section_number"],
                        "section_name": section_info["section_name"],
                        "query_number": section_info["query_number"],
                        "query": section_info["query"],
                        "success": True,
                        "table": table,
                        "graph_and_analysis": graph_analysis
                    })
                else:
                    failed_count += 1
                    results.append({
                        "section_number": section_info["section_number"],
                        "section_name": section_info["section_name"],
                        "query_number": section_info["query_number"],
                        "query": section_info["query"],
                        "success": False,
                        "error": str(graph_result.get("error", "Unknown error"))
                    })
            
        else:
            # Fallback to the regular function if direct functions are not available
            print("⚠️ Direct functions not available, falling back to regular processing")
            return await process_all_queries_with_graph_generation(
                db_id=db_id,
                user_id=user_id,
                export_format=export_format,
                theme=theme,
                analysis_subject=analysis_subject
            )
        
        # Generate combined HTML report
        final_html_path = generate_combined_html_report(results, user_id)
        
        # Generate summary
        summary = {
            "total_sections": len(sections_data["sections"]),
            "total_queries": len(all_queries),
            "successful_queries": successful_count,
            "failed_queries": failed_count,
            "success_rate": (successful_count / len(all_queries)) * 100 if all_queries else 0,
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(all_queries) if all_queries else 0,
            "sections_processed": len(set(r.get("section_number", 0) for r in results)),
            "processing_method": "optimized_batch_function",
            "database_id": db_id,
            "errors_summary": {}
        }
        
        # Group errors by type
        for result in results:
            if result.get("error"):
                error_type = result["error"].split(":")[0] if ":" in result["error"] else "Unknown"
                summary["errors_summary"][error_type] = summary["errors_summary"].get(error_type, 0) + 1
        
        print(f"🎉 OPTIMIZED batch processing completed!")
        print(f"   Database ID: {db_id}")
        print(f"   Total queries: {len(all_queries)}")
        print(f"   Successful: {successful_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Total time: {total_processing_time:.2f}s")
        print(f"   Method: Optimized batch function")
        print(f"   Final HTML: {final_html_path}")
        
        return {
            "success": True,
            "database_id": db_id,
            "total_queries": len(all_queries),
            "successful_queries": successful_count,
            "failed_queries": failed_count,
            "results": results,
            "total_processing_time": total_processing_time,
            "summary": summary,
            "final_html": final_html_path
        }
        
    except Exception as e:
        print(f"❌ Error in optimized batch processing: {e}")
        return {
            "success": False,
            "error": str(e),
            "database_id": db_id,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "total_processing_time": 0,
            "summary": {}
        }

def process_all_queries_with_graph_generation_sync(
    db_id: int = 1,
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string"
) -> Dict[str, Any]:
    """
    Synchronous wrapper for process_all_queries_with_graph_generation.
    
    Args:
        db_id (int): Database ID to fetch report structure from (default: 1)
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis
        
    Returns:
        Dict[str, Any]: Results for all processed queries
    """
    return asyncio.run(process_all_queries_with_graph_generation(
        db_id=db_id,
        user_id=user_id,
        export_format=export_format,
        theme=theme,
        analysis_subject=analysis_subject
    ))

def process_all_queries_with_graph_generation_optimized_sync(
    db_id: int = 1,
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string",
    user_query: str = None
) -> Dict[str, Any]:
    """
    Synchronous wrapper for process_all_queries_with_graph_generation_optimized.
    
    Args:
        db_id (int): Database ID to fetch report structure from (default: 1)
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis
        user_query (str, optional): User's specific request for filtering queries
        
    Returns:
        Dict[str, Any]: Results for all processed queries
    """
    return asyncio.run(process_all_queries_with_graph_generation_optimized(
        db_id=db_id,
        user_id=user_id,
        export_format=export_format,
        theme=theme,
        analysis_subject=analysis_subject,
        user_query=user_query
    ))

# Keep the old functions for backward compatibility
async def process_all_queries_with_graph_generation_optimized_from_file(
    file_path: str = "Report_generator/utilites/testing.txt",
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string"
) -> Dict[str, Any]:
    """
    OPTIMIZED VERSION: Process all queries using the generate_multiple_graphs function.
    This function is kept for backward compatibility.
    
    Args:
        file_path (str): Path to the testing.txt file
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis
        
    Returns:
        Dict[str, Any]: Results for all processed queries
    """
    
    try:
        print(f"🚀 Starting OPTIMIZED batch processing of all queries from {file_path}")
        
        # Step 1: Parse the testing file
        print("📖 Parsing testing file...")
        sections_data = get_sections_and_queries_dict_from_db(file_path)
        
        if not sections_data or "sections" not in sections_data:
            raise Exception("Failed to parse testing file or no sections found")
        
        # Step 2: Extract all queries
        all_queries = []
        query_mapping = {}  # Map query text to section info
        
        for section in sections_data["sections"]:
            section_num = section.get("section_number", 0)
            section_name = section.get("section_name", "Unknown Section")
            
            for query in section.get("queries", []):
                query_num = query.get("query_number", 0)
                query_text = query.get("query", "")
                
                all_queries.append(query_text)
                query_mapping[query_text] = {
                    "section_number": section_num,
                    "section_name": section_name,
                    "query_number": query_num,
                    "query": query_text
                }
        
        print(f"📊 Found {len(all_queries)} queries to process")
        
        # Step 3: Use the optimized batch processing function
        if DIRECT_GRAPH_AVAILABLE:
            print("🎯 Using optimized generate_multiple_graphs function")
            start_time = time.time()
            
            # Call the optimized batch function
            graph_results = await generate_multiple_graphs(
                queries=all_queries,
                user_id=user_id,
                theme=theme,
                export_format=export_format
            )
            
            total_processing_time = time.time() - start_time
            
            # Step 4: Process results and map back to section information
            results = []
            successful_count = 0
            failed_count = 0
            
            for i, graph_result in enumerate(graph_results):
                query_text = all_queries[i]
                section_info = query_mapping[query_text]
                
                if graph_result.get("success"):
                    successful_count += 1
                    
                    # Serialize the graph_result to handle date objects
                    serialized_graph_result = serialize_for_json(graph_result)
                    
                    # Get the data for pagination and analysis
                    data = serialized_graph_result.get("data", {}).get("sample", [])
                    
                    # Create paginated table
                    table = create_paginated_table(data, page_size=10)
                    
                    # Create graph analysis with LLM data analysis
                    graph_analysis = create_graph_analysis(graph_result, data, section_info["query"])
                    
                    results.append({
                        "section_number": section_info["section_number"],
                        "section_name": section_info["section_name"],
                        "query_number": section_info["query_number"],
                        "query": section_info["query"],
                        "success": True,
                        "table": table,
                        "graph_and_analysis": graph_analysis
                    })
                else:
                    failed_count += 1
                    results.append({
                        "section_number": section_info["section_number"],
                        "section_name": section_info["section_name"],
                        "query_number": section_info["query_number"],
                        "query": section_info["query"],
                        "success": False,
                        "error": str(graph_result.get("error", "Unknown error"))
                    })
            
        else:
            # Fallback to the regular function if direct functions are not available
            print("⚠️ Direct functions not available, falling back to regular processing")
            return await process_all_queries_with_graph_generation_from_file(
                file_path=file_path,
                user_id=user_id,
                export_format=export_format,
                theme=theme,
                analysis_subject=analysis_subject
            )
        
        # Generate combined HTML report
        final_html_path = generate_combined_html_report(results, user_id)
        
        # Generate summary
        summary = {
            "total_sections": len(sections_data["sections"]),
            "total_queries": len(all_queries),
            "successful_queries": successful_count,
            "failed_queries": failed_count,
            "success_rate": (successful_count / len(all_queries)) * 100 if all_queries else 0,
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(all_queries) if all_queries else 0,
            "sections_processed": len(set(r.get("section_number", 0) for r in results)),
            "processing_method": "optimized_batch_function",
            "database_id": db_id,
            "errors_summary": {}
        }
        
        # Group errors by type
        for result in results:
            if result.get("error"):
                error_type = result["error"].split(":")[0] if ":" in result["error"] else "Unknown"
                summary["errors_summary"][error_type] = summary["errors_summary"].get(error_type, 0) + 1
        
        print(f"🎉 OPTIMIZED batch processing completed!")
        print(f"   Total queries: {len(all_queries)}")
        print(f"   Successful: {successful_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Total time: {total_processing_time:.2f}s")
        print(f"   Method: Optimized batch function")
        print(f"   Final HTML: {final_html_path}")
        
        return {
            "success": True,
            "total_queries": len(all_queries),
            "successful_queries": successful_count,
            "failed_queries": failed_count,
            "results": results,
            "total_processing_time": total_processing_time,
            "summary": summary,
            "final_html": final_html_path
        }
        
    except Exception as e:
        print(f"❌ Error in optimized batch processing: {e}")
        return {
            "success": False,
            "error": str(e),
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "total_processing_time": 0,
            "summary": {}
        }

def serialize_for_json(obj, exclude_html=True):
    """
    Enhanced JSON serializer to handle date objects and other non-serializable types.
    Excludes HTML content from graph results to reduce response size.
    
    Args:
        obj: Object to serialize
        exclude_html: Whether to exclude HTML content from graph results
    """
    if obj is None:
        return None
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif hasattr(obj, 'isoformat'):  # For any object with isoformat method
        return obj.isoformat()
    elif hasattr(obj, 'strftime'):  # For any object with strftime method
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif hasattr(obj, 'tolist'):  # For numpy arrays
        return obj.tolist()
    elif isinstance(obj, dict):
        # Special handling for graph results to exclude HTML content
        if exclude_html and 'graph' in obj and isinstance(obj['graph'], dict):
            # Create a clean version of graph data without HTML
            clean_graph = {}
            for k, v in obj['graph'].items():
                if k != 'html':  # Exclude HTML content
                    clean_graph[k] = serialize_for_json(v, exclude_html)
            # Add a flag indicating HTML was removed
            clean_graph['html_removed'] = True
            
            # Create new dict with clean graph
            result = {}
            for k, v in obj.items():
                if k == 'graph':
                    result[k] = clean_graph
                else:
                    result[k] = serialize_for_json(v, exclude_html)
            return result
        else:
            return {str(k): serialize_for_json(v, exclude_html) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item, exclude_html) for item in obj]
    elif hasattr(obj, '__dict__'):
        return {str(k): serialize_for_json(v, exclude_html) for k, v in obj.__dict__.items()}
    else:
        return str(obj)

def create_paginated_table(data: List[Dict[str, Any]], page_size: int = 10) -> Dict[str, Any]:
    """
    Create a paginated table structure from data.
    
    Args:
        data: List of dictionaries containing the data
        page_size: Number of rows per page
        
    Returns:
        Dict containing paginated table structure
    """
    if not data:
        return {
            "total_rows": 0,
            "page_size": page_size,
            "total_pages": 0,
            "current_page": 1,
            "columns": [],
            "pages": {}
        }
    
    total_rows = len(data)
    total_pages = (total_rows + page_size - 1) // page_size  # Ceiling division
    columns = list(data[0].keys()) if data else []
    
    pages = {}
    for page_num in range(1, total_pages + 1):
        start_idx = (page_num - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        page_data = data[start_idx:end_idx]
        
        pages[str(page_num)] = {
            "page_number": page_num,
            "start_row": start_idx + 1,
            "end_row": end_idx,
            "data": serialize_for_json(page_data)
        }
    
    return {
        "total_rows": total_rows,
        "page_size": page_size,
        "total_pages": total_pages,
        "current_page": 1,
        "columns": columns,
        "pages": pages
    }

def create_graph_analysis(graph_result: Dict[str, Any], data: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """
    Create analysis and metadata for the graph.
    
    Args:
        graph_result: Result from graph generation
        data: The data used for the graph
        query: The original query
        
    Returns:
        Dict containing graph analysis
    """
    if not graph_result.get("success"):
        return {
            "error": graph_result.get("error", "Graph generation failed"),
            "status": "failed"
        }
    
    # Extract graph information
    graph_type = graph_result.get("graph", {}).get("type", "unknown")
    image_path = graph_result.get("files", {}).get("image_path", "")
    
    # Basic data analysis
    total_records = len(data)
    columns_count = len(data[0].keys()) if data else 0
    sql_query = graph_result.get("data", {}).get("sql", "")
    generation_time = graph_result.get("metadata", {}).get("generation_time", datetime.now().isoformat())
    
    # Create data summary
    data_summary = f"Generated {graph_type.title()} chart with {total_records} records across {columns_count} columns"
    
    # Extract LLM data analysis if available
    llm_analysis = graph_result.get("data_analysis", {})
    
    return {
        "graph_type": graph_type,
        "theme": graph_result.get("graph", {}).get("theme", "modern"),
        "image_path": image_path,
        "analysis": {
            "total_records": total_records,
            "columns_count": columns_count,
            "sql_query": sql_query,
            "generation_time": generation_time,
            "data_summary": data_summary,
            "query": query
        },
        "llm_analysis": llm_analysis  # Include LLM analysis from graph_Generator
    }

def generate_combined_html_report(results: List[Dict[str, Any]], user_id: str = "nabil") -> str:
    """
    Generate a beautiful combined HTML report from all results.
    
    Args:
        results: List of processed results
        user_id: User ID for the report
        
    Returns:
        Path to the generated HTML file
    """
    try:
        # Create storage directory if it doesn't exist
        storage_dir = os.path.join(BASE_DIR, "storage", "reports")
        os.makedirs(storage_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = int(time.time())
        html_filename = f"combined_report_{user_id}_{timestamp}.html"
        html_path = os.path.join(storage_dir, html_filename)
        
        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Analysis Report - {user_id}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .header p {{
            color: #7f8c8d;
            font-size: 1.1em;
        }}
        
        .section {{
            background: white;
            margin-bottom: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .section-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
        }}
        
        .section-header h2 {{
            font-size: 1.8em;
            margin-bottom: 5px;
        }}
        
        .section-header p {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        
        .query-block {{
            border-bottom: 1px solid #ecf0f1;
            padding: 25px 30px;
        }}
        
        .query-block:last-child {{
            border-bottom: none;
        }}
        
        .query-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #ecf0f1;
        }}
        
        .query-info h3 {{
            color: #2c3e50;
            font-size: 1.4em;
            margin-bottom: 5px;
        }}
        
        .query-info p {{
            color: #7f8c8d;
            font-style: italic;
        }}
        
        .status-badge {{
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        
        .status-success {{
            background: #27ae60;
            color: white;
        }}
        
        .status-error {{
            background: #e74c3c;
            color: white;
        }}
        
        .data-section {{
            margin: 20px 0;
        }}
        
        .data-summary {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        
        .data-summary h4 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .data-summary p {{
            color: #7f8c8d;
            margin: 5px 0;
        }}
        
                 .pagination {{
             display: flex;
             justify-content: center;
             align-items: center;
             margin: 20px 0;
             gap: 10px;
             flex-wrap: wrap;
         }}
         
         .page-btn {{
             padding: 8px 12px;
             border: 1px solid #ddd;
             background: white;
             cursor: pointer;
             border-radius: 5px;
             transition: all 0.3s ease;
             min-width: 40px;
             text-align: center;
         }}
         
         .page-btn:hover {{
             background: #667eea;
             color: white;
         }}
         
         .page-btn.active {{
             background: #667eea;
             color: white;
         }}
         
         .page-btn:disabled {{
             opacity: 0.5;
             cursor: not-allowed;
         }}
         
         .pagination-info {{
             margin: 0 15px;
             color: #7f8c8d;
             font-weight: 500;
         }}
        
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .data-table th {{
            background: #34495e;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        .data-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .data-table tr:hover {{
            background: #f8f9fa;
        }}
        
                 .analysis-graph-container {{
             display: grid;
             grid-template-columns: 1fr 1fr;
             gap: 30px;
             margin: 30px 0;
         }}
         
         .analysis-column {{
             background: #f8f9fa;
             padding: 25px;
             border-radius: 15px;
             box-shadow: 0 5px 15px rgba(0,0,0,0.1);
         }}
         
         .graph-column {{
             text-align: center;
         }}
         
         .analysis-column h4 {{
             color: #2c3e50;
             margin-bottom: 20px;
             font-size: 1.3em;
             border-bottom: 2px solid #667eea;
             padding-bottom: 10px;
         }}
         
         .analysis-grid {{
             display: grid;
             grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
             gap: 15px;
             margin-top: 15px;
         }}
         
         .analysis-item {{
             background: white;
             padding: 15px;
             border-radius: 8px;
             box-shadow: 0 2px 10px rgba(0,0,0,0.1);
             text-align: center;
         }}
         
         .analysis-item h5 {{
             color: #667eea;
             margin-bottom: 8px;
             font-size: 0.9em;
             text-transform: uppercase;
             letter-spacing: 0.5px;
         }}
         
         .analysis-item p {{
             color: #2c3e50;
             font-weight: 600;
             font-size: 1.1em;
         }}
         
         .analysis-item .value {{
             font-size: 1.3em;
             color: #667eea;
             font-weight: bold;
         }}
         
         .analysis-item .label {{
             font-size: 0.8em;
             color: #7f8c8d;
             margin-top: 5px;
         }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            color: white;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            margin-top: 30px;
        }}
        
                 @media (max-width: 768px) {{
             .container {{
                 padding: 10px;
             }}
             
             .header h1 {{
                 font-size: 2em;
             }}
             
             .query-header {{
                 flex-direction: column;
                 align-items: flex-start;
                 gap: 10px;
             }}
             
             .analysis-graph-container {{
                 grid-template-columns: 1fr;
                 gap: 20px;
             }}
             
             .pagination {{
                 flex-direction: column;
                 gap: 15px;
             }}
             
             .page-btn {{
                 min-width: 50px;
                 padding: 10px 15px;
             }}
         }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Data Analysis Report</h1>
            <p>Generated for {user_id} on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
"""
        
        # Group results by section
        sections = {}
        for result in results:
            section_num = result.get("section_number", 0)
            if section_num not in sections:
                sections[section_num] = {
                    "section_name": result.get("section_name", "Unknown Section"),
                    "queries": []
                }
            sections[section_num]["queries"].append(result)
        
        # Generate HTML for each section
        for section_num in sorted(sections.keys()):
            section = sections[section_num]
            html_content += f"""
        <div class="section">
            <div class="section-header">
                <h2>Section {section_num}: {section['section_name']}</h2>
                <p>{len(section['queries'])} queries processed</p>
            </div>
"""
            
            for i, query_result in enumerate(section["queries"], 1):
                success = query_result.get("success", False)
                status_class = "status-success" if success else "status-error"
                status_text = "Success" if success else "Failed"
                
                html_content += f"""
            <div class="query-block">
                <div class="query-header">
                    <div class="query-info">
                        <h3>Query {i}: {query_result.get('query', 'No query')}</h3>
                        <p>Section {section_num} • Query {i}</p>
                    </div>
                    <span class="status-badge {status_class}">{status_text}</span>
                </div>
"""
                
                if success:
                    # Add table data
                    table_data = query_result.get("table", {})
                    if table_data and table_data.get("total_rows", 0) > 0:
                        html_content += f"""
                <div class="data-section">
                    <div class="data-summary">
                        <h4>Data Summary</h4>
                        <p><strong>Total Rows:</strong> {table_data.get('total_rows', 0)}</p>
                        <p><strong>Columns:</strong> {', '.join(table_data.get('columns', []))}</p>
                        <p><strong>Pages:</strong> {table_data.get('total_pages', 0)} pages of {table_data.get('page_size', 10)} rows each</p>
                    </div>
                    
                                        <div class="pagination" id="pagination-{i}">
                        <span class="page-btn active">1</span>"""
                        
                        # Add pagination buttons for all pages (static for now)
                        total_pages = table_data.get('total_pages', 1)
                        for page_num in range(2, min(total_pages + 1, 11)):  # Show max 10 pages
                            html_content += f"""
                        <span class="page-btn">{page_num}</span>"""
                        
                        if total_pages > 10:
                            html_content += f"""
                        <span class="pagination-info">...</span>
                        <span class="page-btn">{total_pages}</span>"""
                        
                        html_content += f"""
                        <span class="pagination-info">Page 1 of {total_pages}</span>
                    </div>
                    
                    <table class="data-table" id="data-table-{i}">
                        <thead>
                            <tr>"""
                        for column in table_data.get("columns", []):
                            html_content += f"<th>{column}</th>"
                        html_content += """
                            </tr>
                        </thead>
                        <tbody id="table-body-{i}">"""
                        
                        # Add first page data
                        first_page = table_data.get("pages", {}).get("1", {}).get("data", [])
                        for row in first_page:
                            html_content += "<tr>"
                            for value in row.values():
                                html_content += f"<td>{value}</td>"
                            html_content += "</tr>"
                        
                        html_content += """
                        </tbody>
                    </table>
                </div>
"""
                    
                                        # Add graph and analysis in two-column layout
                    graph_analysis = query_result.get("graph_and_analysis", {})
                    if graph_analysis and graph_analysis.get("image_path"):
                        # Get analysis values
                        total_records = graph_analysis.get('analysis', {}).get('total_records', 0)
                        columns_count = graph_analysis.get('analysis', {}).get('columns_count', 0)
                        graph_type = graph_analysis.get('graph_type', 'Unknown').title()
                        theme = graph_analysis.get('theme', 'Modern').title()
                        data_summary = graph_analysis.get('analysis', {}).get('data_summary', 'No analysis available')
                        generation_time = graph_analysis.get('analysis', {}).get('generation_time', 'Unknown')
                        image_path = graph_analysis.get('image_path', '')
                        
                        # Start the analysis-graph-container
                        html_content += f"""
                <div class="analysis-graph-container">
                    <div class="analysis-column">
                        <h4>Analysis Results</h4>
                        <div class="analysis-grid">
                            <div class="analysis-item">
                                <h5>Total Records</h5>
                                <p class="value">{total_records}</p>
                                <p class="label">Data Points</p>
                            </div>
                            <div class="analysis-item">
                                <h5>Columns</h5>
                                <p class="value">{columns_count}</p>
                                <p class="label">Fields</p>
                            </div>
                            <div class="analysis-item">
                                <h5>Graph Type</h5>
                                <p class="value">{graph_type}</p>
                                <p class="label">Visualization</p>
                            </div>
                            <div class="analysis-item">
                                <h5>Theme</h5>
                                <p class="value">{theme}</p>
                                <p class="label">Style</p>
                            </div>
                        </div>
                        <div style="margin-top: 20px; padding: 15px; background: white; border-radius: 8px; border-left: 4px solid #667eea;">
                            <h5 style="color: #667eea; margin-bottom: 10px;">Data Summary</h5>
                            <p style="color: #2c3e50; line-height: 1.6;">
                                {data_summary}
                            </p>
                        </div>
                        <div style="margin-top: 15px; padding: 15px; background: white; border-radius: 8px; border-left: 4px solid #27ae60;">
                            <h5 style="color: #27ae60; margin-bottom: 10px;">Generated</h5>
                            <p style="color: #2c3e50;">
                                {generation_time}
                            </p>
                        </div>"""
                        
                        # Add LLM Analysis section if available
                        llm_analysis = graph_analysis.get("llm_analysis", {})
                        if llm_analysis and llm_analysis.get("analysis"):
                            analysis_content = llm_analysis.get("analysis", "")
                            analysis_subject = llm_analysis.get("analysis_subject", "Data Analysis")
                            data_coverage = llm_analysis.get("data_coverage", "")
                            
                            html_content += f"""
                        <div style="margin-top: 15px; padding: 20px; background: white; border-radius: 8px; border-left: 4px solid #e67e22;">
                            <h5 style="color: #e67e22; margin-bottom: 15px;">AI Data Analysis</h5>
                            <div style="margin-bottom: 10px;">
                                <strong style="color: #2c3e50;">Subject:</strong> {analysis_subject}
                            </div>
                            <div style="margin-bottom: 10px;">
                                <strong style="color: #2c3e50;">Coverage:</strong> {data_coverage}
                            </div>
                            <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;">
                                <pre style="color: #2c3e50; line-height: 1.6; white-space: pre-wrap; font-family: inherit; margin: 0;">{analysis_content}</pre>
                            </div>
                        </div>"""
                        
                        # Close analysis-column and add graph-column
                        html_content += f"""
                    </div>
                    
                    <div class="graph-column">
                        <h4>Data Visualization</h4>
                        <img src="{image_path}" alt="Data Visualization" style="max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);" />
                    </div>
                </div>
"""
                else:
                    # Show error
                    error_msg = query_result.get("error", "Unknown error")
                    html_content += f"""
                <div class="data-section">
                    <div class="data-summary" style="background: #fee; border-left: 4px solid #e74c3c;">
                        <h4 style="color: #e74c3c;">Error</h4>
                        <p style="color: #c0392b;">{error_msg}</p>
                    </div>
                </div>
"""
                
                html_content += """
            </div>
"""
            
            html_content += """
        </div>
"""
        
        # Add footer
        html_content += f"""
        <div class="footer">
            <p>Report generated by Data Analysis System</p>
            <p>Total queries processed: {len(results)}</p>
            <p>Successful: {sum(1 for r in results if r.get('success', False))} | Failed: {sum(1 for r in results if not r.get('success', False))}</p>
                </div>
    </div>
</body>
</html>
"""
        
        # Write HTML file
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Combined HTML report generated: {html_path}")
        return html_path
        
    except Exception as e:
        print(f"❌ Error generating combined HTML report: {e}")
        return ""

def get_all_results_as_json_dict(
    db_id: int = 1,
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string",
    use_optimized: bool = True
) -> Dict[str, Any]:
    """
    Get all results as a JSON-serializable dictionary.
    
    Args:
        db_id (int): Database ID to fetch report structure from (default: 1)
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis
        use_optimized (bool): Whether to use optimized processing
        
    Returns:
        Dict[str, Any]: JSON-serializable dictionary with all results
    """
    try:
        print(f"🚀 Getting all results as JSON dictionary from database ID {db_id}")
        
        # Choose the appropriate function based on optimization preference
        if use_optimized:
            print("🎯 Using optimized processing...")
            result = process_all_queries_with_graph_generation_optimized_sync(
                db_id=db_id,
                user_id=user_id,
                export_format=export_format,
                theme=theme,
                analysis_subject=analysis_subject
            )
        else:
            print("🔄 Using regular processing...")
            result = process_all_queries_with_graph_generation_sync(
                db_id=db_id,
                user_id=user_id,
                export_format=export_format,
                theme=theme,
                analysis_subject=analysis_subject
            )
        
        # Convert the result to a JSON-serializable format
        json_result = serialize_for_json(result)
        
        print(f"✅ Successfully converted results to JSON format")
        print(f"   Database ID: {db_id}")
        print(f"   Total queries processed: {json_result.get('total_queries', 0)}")
        print(f"   Successful: {json_result.get('successful_queries', 0)}")
        print(f"   Failed: {json_result.get('failed_queries', 0)}")
        
        return json_result
        
    except Exception as e:
        print(f"❌ Error getting results as JSON: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "database_id": db_id,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "summary": {}
        }

def get_results_as_json_simple(
    db_id: int = 1,
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string"
) -> Dict[str, Any]:
    """
    Simple function to get results as JSON immediately.
    This function handles JSON serialization issues directly.
    
    Args:
        db_id (int): Database ID to fetch report structure from (default: 1)
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis
        
    Returns:
        Dict[str, Any]: Clean JSON-serializable dictionary with all results
    """
    try:
        print(f"🚀 Getting results as JSON for user: {user_id} from database ID: {db_id}")
        
        # Get the results using optimized processing
        result = process_all_queries_with_graph_generation_optimized_sync(
            db_id=db_id,
            user_id=user_id,
            export_format=export_format,
            theme=theme,
            analysis_subject=analysis_subject
        )
        
        print("✅ Results obtained successfully")
        print(f"   Database ID: {db_id}")
        print(f"   Total queries: {result.get('total_queries', 0)}")
        print(f"   Successful: {result.get('successful_queries', 0)}")
        print(f"   Failed: {result.get('failed_queries', 0)}")
        
        # Use our enhanced serializer to handle all types properly
        clean_result = serialize_for_json(result)
        
        print(f"✅ Successfully converted to JSON format")
        print(f"   Database ID: {db_id}")
        print(f"   Total queries processed: {clean_result.get('total_queries', 0)}")
        print(f"   Successful: {clean_result.get('successful_queries', 0)}")
        print(f"   Failed: {clean_result.get('failed_queries', 0)}")
        
        return clean_result
        
    except Exception as e:
        print(f"❌ Error getting results as JSON: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "database_id": db_id,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "summary": {}
        }

# Keep the old functions for backward compatibility
def get_all_results_as_json_dict_from_file(
    file_path: str = "Report_generator/utilites/testing.txt",
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string",
    use_optimized: bool = True
) -> Dict[str, Any]:
    """
    Get all results as a JSON-serializable dictionary from file.
    This function is kept for backward compatibility.
    
    Args:
        file_path (str): Path to the testing.txt file
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis
        use_optimized (bool): Whether to use optimized processing
        
    Returns:
        Dict[str, Any]: JSON-serializable dictionary with all results
    """
    try:
        print(f"🚀 Getting all results as JSON dictionary from {file_path}")
        
        # Choose the appropriate function based on optimization preference
        if use_optimized:
            print("🎯 Using optimized processing...")
            result = asyncio.run(process_all_queries_with_graph_generation_optimized_from_file(
                file_path=file_path,
                user_id=user_id,
                export_format=export_format,
                theme=theme,
                analysis_subject=analysis_subject
            ))
        else:
            print("🔄 Using regular processing...")
            result = asyncio.run(process_all_queries_with_graph_generation_from_file(
                file_path=file_path,
                user_id=user_id,
                export_format=export_format,
                theme=theme,
                analysis_subject=analysis_subject
            ))
        
        # Convert the result to a JSON-serializable format
        json_result = serialize_for_json(result)
        
        print(f"✅ Successfully converted results to JSON format")
        print(f"   File: {file_path}")
        print(f"   Total queries processed: {json_result.get('total_queries', 0)}")
        print(f"   Successful: {json_result.get('successful_queries', 0)}")
        print(f"   Failed: {json_result.get('failed_queries', 0)}")
        
        return json_result
        
    except Exception as e:
        print(f"❌ Error getting results as JSON: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "file_path": file_path,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "summary": {}
        }

def get_results_as_json_simple_from_file(
    file_path: str = "Report_generator/utilites/testing.txt",
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string"
) -> Dict[str, Any]:
    """
    Simple function to get results as JSON immediately from file.
    This function is kept for backward compatibility.
    
    Args:
        file_path (str): Path to the testing.txt file
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis
        
    Returns:
        Dict[str, Any]: Clean JSON-serializable dictionary with all results
    """
    try:
        print(f"🚀 Getting results as JSON for user: {user_id} from file: {file_path}")
        
        # Get the results using optimized processing
        result = asyncio.run(process_all_queries_with_graph_generation_optimized_from_file(
            file_path=file_path,
            user_id=user_id,
            export_format=export_format,
            theme=theme,
            analysis_subject=analysis_subject
        ))
        
        print("✅ Results obtained successfully")
        print(f"   File: {file_path}")
        print(f"   Total queries: {result.get('total_queries', 0)}")
        print(f"   Successful: {result.get('successful_queries', 0)}")
        print(f"   Failed: {result.get('failed_queries', 0)}")
        
        # Use our enhanced serializer to handle all types properly
        clean_result = serialize_for_json(result)
        
        print(f"✅ Successfully converted to JSON format")
        print(f"   File: {file_path}")
        print(f"   Total queries processed: {clean_result.get('total_queries', 0)}")
        print(f"   Successful: {clean_result.get('successful_queries', 0)}")
        print(f"   Failed: {clean_result.get('failed_queries', 0)}")
        
        return clean_result
        
    except Exception as e:
        print(f"❌ Error getting results as JSON: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "file_path": file_path,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "summary": {}
        }

def convert_result_to_json_serializable(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a result dictionary to be JSON serializable by handling date objects and other non-serializable types.
    """
    try:
        # Use our enhanced serializer to handle all types properly
        return serialize_for_json(result)
    except Exception as e:
        print(f"Error: Could not serialize result: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a basic error response
        return {
            "success": False,
            "error": f"Serialization failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "total_queries": result.get("total_queries", 0),
            "successful_queries": 0,
            "failed_queries": result.get("total_queries", 0),
            "results": [],
            "summary": {"error": "Failed to serialize results"}
        }

def generate_complete_report(user_id: str, user_query: str = None) -> Dict[str, Any]:
    """
    MAIN FUNCTION: Generate a complete report for a user with user_id and optional user query.
    
    This function:
    1. Gets the database ID from user_id automatically
    2. Fetches and parses the report structure
    3. If user_query is provided, customizes queries based on user's specific request (extracts ALL queries)
    4. Generates graphs as images using optimized batch processing
    5. Returns the complete report with all data in the current format
    
    Args:
        user_id (str): User ID to generate report for
        user_query (str, optional): User's specific request (e.g., "financial report", "attendance summary", "salary analysis for July 2024")
        
    Returns:
        Dict[str, Any]: Complete report with success status, results, graphs as images, and summary
    """
    try:
        print(f"🚀 Generating complete report for user: {user_id}")
        start_time = time.time()
        
        # Step 1: Get the database ID for the user using direct database manager call
        print(f"📊 Getting database ID for user: {user_id}")
        
        if DIRECT_DB_AVAILABLE:
            print("🎯 Using direct database manager call")
            try:
                user_data = db_manager.get_user_current_db_details(user_id)
                if not user_data or not user_data.get('db_id'):
                    return {
                        "success": False,
                        "error": f"No current database found for user {user_id}. Please set a current database first.",
                        "user_id": user_id,
                        "timestamp": datetime.now().isoformat(),
                        "total_queries": 0,
                        "successful_queries": 0,
                        "failed_queries": 0,
                        "results": [],
                        "summary": {}
                    }
                
                db_id = user_data['db_id']
                print(f"✅ Found database ID {db_id} for user {user_id}")
                
            except Exception as e:
                print(f"❌ Error getting database ID: {e}")
                return {
                    "success": False,
                    "error": f"Failed to get database ID for user {user_id}: {str(e)}",
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "total_queries": 0,
                    "successful_queries": 0,
                    "failed_queries": 0,
                    "results": [],
                    "summary": {}
                }
        else:
            # Fallback to sync version if direct database is not available
            print("🔄 Using sync database ID lookup")
            try:
                db_id = get_db_id_from_user_id_sync(user_id)
                print(f"✅ Found database ID {db_id} for user {user_id}")
            except Exception as e:
                print(f"❌ Error getting database ID: {e}")
                return {
                    "success": False,
                    "error": f"Failed to get database ID for user {user_id}: {str(e)}",
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "total_queries": 0,
                    "successful_queries": 0,
                    "failed_queries": 0,
                    "results": [],
                    "summary": {}
                }
        
        # Step 2: Generate the complete report using optimized batch processing
        if user_query:
            print(f"📈 Generating targeted report for database ID {db_id}")
            print(f"🎯 User Query: {user_query}")
        else:
            print(f"📈 Generating complete report for database ID {db_id}")
        
        # Use the optimized processing function with PNG export for images
        result = process_all_queries_with_graph_generation_optimized_sync(
            db_id=db_id,
            user_id=user_id,
            export_format="png",  # Always use PNG for images
            theme="modern",       # Use modern theme
            analysis_subject="data analysis",
            user_query=user_query  # Pass user query for filtering
        )
        
        total_time = time.time() - start_time
        
        # Step 3: Enhance the result with additional metadata
        if result.get("success"):
            result["user_id"] = user_id
            result["generation_time"] = total_time
            result["timestamp"] = datetime.now().isoformat()
            
            # Update summary with user info
            if "summary" in result:
                result["summary"]["user_id"] = user_id
                result["summary"]["generation_time"] = total_time
                result["summary"]["timestamp"] = datetime.now().isoformat()
            
            print(f"🎉 Report generation completed successfully!")
            print(f"   User ID: {user_id}")
            print(f"   Database ID: {db_id}")
            print(f"   Total queries: {result.get('total_queries', 0)}")
            print(f"   Successful: {result.get('successful_queries', 0)}")
            print(f"   Failed: {result.get('failed_queries', 0)}")
            print(f"   Total generation time: {total_time:.2f}s")
            print(f"   HTML report: {result.get('final_html', 'Not generated')}")
        else:
            result["user_id"] = user_id
            result["generation_time"] = total_time
            result["timestamp"] = datetime.now().isoformat()
            print(f"❌ Report generation failed for user {user_id}")
        
        # Convert to JSON-serializable format
        return serialize_for_json(result)
        
    except Exception as e:
        total_time = time.time() - start_time if 'start_time' in locals() else 0
        print(f"❌ Critical error in report generation for user {user_id}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": f"Critical error in report generation: {str(e)}",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "generation_time": total_time,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "summary": {"error": "Critical failure in report generation"}
        }

async def generate_complete_report_async(user_id: str, user_query: str = None) -> Dict[str, Any]:
    """
    Async version of generate_complete_report.
    
    Args:
        user_id (str): User ID to generate report for
        user_query (str, optional): User's specific request (e.g., "financial report", "attendance summary", "salary analysis for July 2024")
        
    Returns:
        Dict[str, Any]: Complete report with success status, results, graphs as images, and summary
    """
    try:
        print(f"🚀 Generating complete report (async) for user: {user_id}")
        start_time = time.time()
        
        # Step 1: Get the database ID for the user
        print(f"📊 Getting database ID for user: {user_id}")
        
        try:
            db_id = await get_db_id_from_user_id(user_id)
            print(f"✅ Found database ID {db_id} for user {user_id}")
        except Exception as e:
            print(f"❌ Error getting database ID: {e}")
            return {
                "success": False,
                "error": f"Failed to get database ID for user {user_id}: {str(e)}",
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "results": [],
                "summary": {}
            }
        
        # Step 2: Generate the complete report using optimized batch processing
        if user_query:
            print(f"📈 Generating targeted report for database ID {db_id}")
            print(f"🎯 User Query: {user_query}")
        else:
            print(f"📈 Generating complete report for database ID {db_id}")
        
        # Use the optimized async processing function with PNG export for images
        result = await process_all_queries_with_graph_generation_optimized(
            db_id=db_id,
            user_id=user_id,
            export_format="png",  # Always use PNG for images
            theme="modern",       # Use modern theme
            analysis_subject="data analysis",
            user_query=user_query  # Pass user query for filtering
        )
        
        total_time = time.time() - start_time
        
        # Step 3: Enhance the result with additional metadata
        if result.get("success"):
            result["user_id"] = user_id
            result["generation_time"] = total_time
            result["timestamp"] = datetime.now().isoformat()
            
            # Update summary with user info
            if "summary" in result:
                result["summary"]["user_id"] = user_id
                result["summary"]["generation_time"] = total_time
                result["summary"]["timestamp"] = datetime.now().isoformat()
            
            print(f"🎉 Async report generation completed successfully!")
            print(f"   User ID: {user_id}")
            print(f"   Database ID: {db_id}")
            print(f"   Total queries: {result.get('total_queries', 0)}")
            print(f"   Successful: {result.get('successful_queries', 0)}")
            print(f"   Failed: {result.get('failed_queries', 0)}")
            print(f"   Total generation time: {total_time:.2f}s")
            print(f"   HTML report: {result.get('final_html', 'Not generated')}")
        else:
            result["user_id"] = user_id
            result["generation_time"] = total_time
            result["timestamp"] = datetime.now().isoformat()
            print(f"❌ Async report generation failed for user {user_id}")
        
        # Convert to JSON-serializable format
        return serialize_for_json(result)
        
    except Exception as e:
        total_time = time.time() - start_time if 'start_time' in locals() else 0
        print(f"❌ Critical error in async report generation for user {user_id}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": f"Critical error in async report generation: {str(e)}",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "generation_time": total_time,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "summary": {"error": "Critical failure in async report generation"}
        }

def save_results_to_json_file(
    results: Dict[str, Any],
    output_file: str = "report_results.json"
) -> str:
    """
    Save results to a JSON file.
    
    Args:
        results: The results dictionary
        output_file: Output file path
        
    Returns:
        str: Path to the saved file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, default=serialize_for_json, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"❌ Error saving results to file: {e}")
        return ""

async def process_all_queries_with_graph_generation_for_user(
    user_id: str,
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string"
) -> Dict[str, Any]:
    """
    Process all queries from the database for a specific user by calling the generate-graph function directly.
    
    Args:
        user_id (str): User ID to get database and process queries for
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis
        
    Returns:
        Dict[str, Any]: Results for all processed queries
    """
    
    try:
        print(f"🚀 Starting batch processing of all queries for user: {user_id}")
        
        # Step 1: Get the database ID for the user
        db_id = await get_db_id_from_user_id(user_id)
        print(f"📊 Using database ID {db_id} for user {user_id}")
        
        # Step 2: Parse the database content
        print("📖 Parsing database content...")
        sections_data = await get_sections_and_queries_dict_from_user_async(user_id)
        
        if not sections_data or "sections" not in sections_data:
            raise Exception("Failed to parse database content or no sections found")
        
        # Step 3: Extract all queries
        all_queries = []
        for section in sections_data["sections"]:
            section_num = section.get("section_number", 0)
            section_name = section.get("section_name", "Unknown Section")
            
            for query in section.get("queries", []):
                query_num = query.get("query_number", 0)
                query_text = query.get("query", "")
                
                all_queries.append({
                    "section_number": section_num,
                    "section_name": section_name,
                    "query_number": query_num,
                    "query": query_text
                })
        
        print(f"📊 Found {len(all_queries)} queries to process")
        
        # Step 4: Process each query using direct function calls
        results = []
        successful_count = 0
        failed_count = 0
        
        if DIRECT_GRAPH_AVAILABLE:
            print("🎯 Using direct graph generation functions (no HTTP overhead)")
            
            async def process_single_query_direct(query_info: Dict[str, Any]) -> Dict[str, Any]:
                """Process a single query using direct function call."""
                query_start_time = time.time()
                
                try:
                    # Call the direct graph generation function
                    graph_result = await generate_beautiful_graph(
                        query=query_info["query"],
                        user_id=user_id,
                        export_format=export_format,
                        theme=theme
                    )
                    
                    query_processing_time = time.time() - query_start_time
                    
                    if graph_result.get("success"):
                        # Serialize the graph_result to handle date objects
                        serialized_graph_result = serialize_for_json(graph_result)
                        
                        # Get the data for pagination and analysis
                        data = serialized_graph_result.get("data", {}).get("sample", [])
                        
                        # Create paginated table
                        table = create_paginated_table(data, page_size=10)
                        
                        # Create graph analysis with LLM data analysis
                        graph_analysis = create_graph_analysis(graph_result, data, query_info["query"])
                        
                        return {
                            "section_number": query_info["section_number"],
                            "section_name": query_info["section_name"],
                            "query_number": query_info["query_number"],
                            "query": query_info["query"],
                            "success": True,
                            "table": table,
                            "graph_and_analysis": graph_analysis
                        }
                    else:
                        return {
                            "section_number": query_info["section_number"],
                            "section_name": query_info["section_name"],
                            "query_number": query_info["query_number"],
                            "query": query_info["query"],
                            "success": False,
                            "error": str(graph_result.get("error", "Unknown error"))
                        }
                        
                except Exception as e:
                    query_processing_time = time.time() - query_start_time
                    return {
                        "section_number": query_info["section_number"],
                        "section_name": query_info["section_name"],
                        "query_number": query_info["query_number"],
                        "query": query_info["query"],
                        "success": False,
                        "error": str(e)
                    }
            
            # Process queries with limited concurrency to avoid overwhelming the system
            semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent requests
            
            async def process_with_semaphore(query_info):
                async with semaphore:
                    return await process_single_query_direct(query_info)
            
            # Execute all queries
            tasks = [process_with_semaphore(query_info) for query_info in all_queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        else:
            # Fallback to HTTP API calls if direct functions are not available
            print("⚠️ Using HTTP API calls (direct functions not available)")
            import aiohttp
            
            # Load base URL from environment variable
            base_url = os.getenv("BASE_URL", "http://localhost:8000")
            print(f"🌐 Using base URL: {base_url}")
            
            async def process_single_query_http(query_info: Dict[str, Any]) -> Dict[str, Any]:
                """Process a single query by calling the generate-graph endpoint."""
                query_start_time = time.time()
                
                try:
                    # Prepare request payload
                    payload = {
                        "query": query_info["query"],
                        "user_id": user_id,
                        "export_format": export_format,
                        "theme": theme,
                        "analysis_subject": analysis_subject
                    }
                    
                    # Make async HTTP request
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{base_url}/graph/generate-graph",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=300),  # 5 minutes timeout
                            ssl=False  # Disable SSL verification for self-signed certificates
                        ) as response:
                            
                            if response.status == 200:
                                result_data = await response.json()
                                query_processing_time = time.time() - query_start_time
                                
                                # Serialize the result_data to handle date objects
                                serialized_result_data = serialize_for_json(result_data)
                                
                                return {
                                    "section_number": query_info["section_number"],
                                    "section_name": query_info["section_name"],
                                    "query_number": query_info["query_number"],
                                    "query": query_info["query"],
                                    "success": True,
                                    "files": serialized_result_data.get("files", {}),
                                    "data": serialized_result_data.get("data", {})
                                }
                            else:
                                error_text = await response.text()
                                query_processing_time = time.time() - query_start_time
                                
                                return {
                                    "section_number": query_info["section_number"],
                                    "section_name": query_info["section_name"],
                                    "query_number": query_info["query_number"],
                                    "query": query_info["query"],
                                    "success": False,
                                    "error": f"HTTP {response.status}: {error_text}"
                                }
                                
                except Exception as e:
                    query_processing_time = time.time() - query_start_time
                    return {
                        "section_number": query_info["section_number"],
                        "section_name": query_info["section_name"],
                        "query_number": query_info["query_number"],
                        "query": query_info["query"],
                        "success": False,
                        "error": str(e)
                    }
            
            # Process queries concurrently (limit to 5 concurrent requests to avoid overwhelming the server)
            semaphore = asyncio.Semaphore(5)
            
            async def process_with_semaphore(query_info):
                async with semaphore:
                    return await process_single_query_http(query_info)
            
            # Execute all queries
            tasks = [process_with_semaphore(query_info) for query_info in all_queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_count = sum(1 for r in results if isinstance(r, dict) and r.get("success", False))
        failed_count = len(results) - successful_count
        
        # Generate combined HTML report
        final_html_path = generate_combined_html_report(results, user_id)
        
        # Generate summary
        total_processing_time = sum(r.get("processing_time", 0) for r in results if isinstance(r, dict))
        
        summary = {
            "total_sections": len(sections_data["sections"]),
            "total_queries": len(all_queries),
            "successful_queries": successful_count,
            "failed_queries": failed_count,
            "success_rate": (successful_count / len(all_queries)) * 100 if all_queries else 0,
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(all_queries) if all_queries else 0,
            "sections_processed": len(set(r.get("section_number", 0) for r in results if isinstance(r, dict))),
            "processing_method": "direct_function" if DIRECT_GRAPH_AVAILABLE else "http_api",
            "database_id": db_id,
            "user_id": user_id,
            "errors_summary": {}
        }
        
        # Group errors by type
        for result in results:
            if isinstance(result, dict) and result.get("error"):
                error_type = result["error"].split(":")[0] if ":" in result["error"] else "Unknown"
                summary["errors_summary"][error_type] = summary["errors_summary"].get(error_type, 0) + 1
        
        print(f"🎉 Batch processing completed!")
        print(f"   User ID: {user_id}")
        print(f"   Database ID: {db_id}")
        print(f"   Total queries: {len(all_queries)}")
        print(f"   Successful: {successful_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Total time: {total_processing_time:.2f}s")
        print(f"   Method: {'Direct function calls' if DIRECT_GRAPH_AVAILABLE else 'HTTP API calls'}")
        print(f"   Final HTML: {final_html_path}")
        
        return {
            "success": True,
            "user_id": user_id,
            "database_id": db_id,
            "total_queries": len(all_queries),
            "successful_queries": successful_count,
            "failed_queries": failed_count,
            "results": results,
            "total_processing_time": total_processing_time,
            "summary": summary,
            "final_html": final_html_path
        }
        
    except Exception as e:
        print(f"❌ Error in batch processing for user {user_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "user_id": user_id,
            "database_id": None,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "total_processing_time": 0,
            "summary": {}
        }
if __name__ == "__main__":
    try:
        print("="*80)
        print("🚀 STREAMLINED REPORT GENERATION SYSTEM")
        print("="*80)
        print("📝 This system now requires only user_id as input!")
        print("📊 Database ID is automatically fetched from the user's current database setting")
        print("🎯 Graphs are generated as PNG images with modern theme")
        print("📈 Uses optimized batch processing for maximum performance")
        print("="*80)

        # ✅ Set your inputs here (no need to type in terminal)
        test_user_id = "nabil"              # Default user_id
        user_query = "financial report 2023"  # Or None if you want the full report

        print(f"📋 Using user_id: {test_user_id}")
        if user_query:
            print(f"🎯 Targeted query: {user_query}")
        else:
            print("💡 No specific query provided - generating complete report")

        print("-"*80)

        # Generate the report
        start_time = time.time()
        report_result = generate_complete_report(test_user_id, user_query)
        total_time = time.time() - start_time

        print("-"*80)
        print("📊 FINAL REPORT SUMMARY:")
        print("-"*80)

        if report_result.get("success"):
            print(f"✅ SUCCESS: Report generated successfully!")
            print(f"👤 User ID: {report_result.get('user_id', 'Unknown')}")
            print(f"🗄️ Database ID: {report_result.get('database_id', 'Unknown')}")
            print(f"📝 Total Queries: {report_result.get('total_queries', 0)}")
            print(f"✅ Successful: {report_result.get('successful_queries', 0)}")
            print(f"❌ Failed: {report_result.get('failed_queries', 0)}")
            print(f"⏱️ Generation Time: {total_time:.2f} seconds")
            print(f"📈 Success Rate: {report_result.get('summary', {}).get('success_rate', 0):.1f}%")
            print(f"🌐 HTML Report: {report_result.get('final_html', 'Not generated')}")

            sections_processed = report_result.get('summary', {}).get('sections_processed', 0)
            total_sections = report_result.get('summary', {}).get('total_sections', 0)
            print(f"📂 Sections Processed: {sections_processed}/{total_sections}")

            method = report_result.get('summary', {}).get('processing_method', 'Unknown')
            print(f"⚙️ Processing Method: {method}")

            errors_summary = report_result.get('summary', {}).get('errors_summary', {})
            if errors_summary:
                print(f"⚠️ Errors Summary:")
                for error_type, count in errors_summary.items():
                    print(f"   - {error_type}: {count} occurrences")

            print(f"\n🎉 Report generation completed successfully!")
            print(f"🌐 Open the HTML file to view the complete visual report")

            # Save results to file
            print(f"\n💾 Saving results to JSON file...")
            timestamp = int(time.time())
            output_file = f"report_results_{test_user_id}_{timestamp}.json"
            saved_file = save_results_to_json_file(report_result, output_file)
            if saved_file:
                print(f"✅ Results saved to: {saved_file}")
            else:
                print(f"❌ Failed to save results to file")

        else:
            print(f"❌ FAILED: Report generation failed")
            print(f"👤 User ID: {report_result.get('user_id', 'Unknown')}")
            print(f"🚨 Error: {report_result.get('error', 'Unknown error')}")
            print(f"⏱️ Time Before Failure: {total_time:.2f} seconds")

        print("="*80)
        print("🔚 Report generation process completed")
        print("="*80)

    except KeyboardInterrupt:
        print(f"\n🛑 Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Critical error in main function: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
