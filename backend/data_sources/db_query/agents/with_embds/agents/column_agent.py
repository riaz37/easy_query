import os
import json
import time
import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from utilities import initialize_llm_gemini_light
from with_embds.agents.table_agent import table_agent

import os
import json
import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# Define Pydantic models for structured data
class RequiredTable(BaseModel):
    table_name: str = Field(..., description="Name of the table")
    relevance: str = Field(..., description="Explanation of why this table is relevant")

class SystemPromptOutput(BaseModel):
    query: str = Field(..., description="The original user query")
    required_tables: List[RequiredTable] = Field(
        ..., description="List of tables identified as required for answering the query"
    )
    overall_reasoning: str = Field(
        ..., description="A summary explanation of the overall reasoning behind the table selections"
    )

class RelevantColumn(BaseModel):
    column_name: str = Field(..., description="Name of the column")
    data_type: str = Field(..., description="Data type of the column")

class TableAnalysis(BaseModel):
    table_name: str = Field(..., description="Name of the table")
    relevant_columns: List[RelevantColumn] = Field(
        ..., description="List of columns identified as relevant for the query"
    )
    table_reasoning: str = Field(..., description="Overall reasoning for this table's relevance to the query")

class ConciseDetailedTable(BaseModel):
    query: str = Field(..., description="The original user query that led to selection of this table")
    table_name: str = Field(..., description="Name of the table")
    columns_concise: List[str] = Field(default_factory=list, description="Concise, human-readable column summaries with all information")
    relationships_concise: List[str] = Field(default_factory=list, description="Concise, human-readable relationship summaries with all information")

def format_columns_concisely(columns: List[Dict[str, Any]]) -> List[str]:
    """
    Create a concise string representation for each column, including all key-value pairs.
    
    Args:
        columns: List of dictionaries with column details.
    
    Returns:
        List of formatted strings, one per column.
    """
    formatted_columns = []
    for col in columns:
        # Build a list of key-value pairs for this column.
        parts = []
        for key, value in col.items():
            parts.append(f"{key}: {value}")
        formatted_columns.append("; ".join(parts))
    return formatted_columns

def format_relationship_concisely(relationship: Dict[str, Any]) -> str:
    """
    Create a concise string representation for a relationship dictionary, including all key-value pairs.
    
    Args:
        relationship: Dictionary containing relationship details.
    
    Returns:
        A formatted string with all key-value pairs.
    """
    parts = []
    for key, value in relationship.items():
        parts.append(f"{key}: {value}")
    return "; ".join(parts)

def get_detailed_tables_concise_output(user_query: str, table_names: List[str], json_data: Dict[str, Any]) -> List[ConciseDetailedTable]:
    """
    For each table in table_names, fetch detailed table info from the JSON data and then produce 
    both full and concise representations for columns and relationships, excluding empty relationships.
    
    Process:
      1. Retrieve detailed table info from the JSON data.
      2. Format the column details into a concise list using format_columns_concisely.
      3. For each candidate table (other than the current one), get relationship details 
         from the JSON data and collect both full and concise representations.
         If the relationship data is empty or null, it is skipped.
    
    Args:
        user_query: The original query provided by the user.
        table_names: A list of table names (e.g. from previous LLM output).
        json_data: The loaded JSON data containing table information.
    
    Returns:
        List of ConciseDetailedTable objects.
    """
    concise_tables = []
    
    # Check if tables exist in the JSON data
    if "tables" not in json_data:
        print("No tables found in the JSON data")
        return []
    
    # Create a lookup dictionary for quick access to table data
    table_lookup = {table_data.get("table_name"): table_data for table_data in json_data["tables"]}
    
    for table_name in table_names:
        try:
            # Get the table data from the lookup dictionary
            table_data = table_lookup.get(table_name)
            
            if not table_data:
                print(f"Table '{table_name}' not found in the JSON data")
                continue
            
            # Retrieve full column details
            full_columns = table_data.get("columns", [])
            # Build the concise version while retaining all information
            formatted_columns = format_columns_concisely(full_columns)
            
            # Initialize relationships info
            relationships_concise = []
            
            # Process relationships from the JSON data
            if "relationships" in table_data:
                for rel in table_data["relationships"]:
                    related_table = rel.get("related_table", "")
                    # Only include relationships to tables that are in our table_names list
                    if related_table in table_names:
                        relationships_concise.append(format_relationship_concisely(rel))
            
            # Build the detailed info dictionary including both full and concise representations
            detailed_info = {
                "query": user_query,
                "table_name": table_data.get("table_name", table_name),
                "columns_concise": formatted_columns,
                "relationships_concise": relationships_concise
            }
            
            # Create a Pydantic instance
            concise_table = ConciseDetailedTable(**detailed_info)
            concise_tables.append(concise_table)
        
        except Exception as e:
            print(f"Error processing table '{table_name}': {e}")
            continue

    return concise_tables

def format_for_llm(tables: List[ConciseDetailedTable]) -> str:
    """
    Format the table information in a structured way for an LLM to easily understand
    
    Args:
        tables: List of ConciseDetailedTable objects
        
    Returns:
        A formatted string representation optimized for LLM consumption
    """
    output = "# DATABASE SCHEMA INFORMATION\n\n"
    
    # Add query context
    if tables:
        output += f"## QUERY CONTEXT\n"
        output += f"User Query: {tables[0].query}\n\n"
    
    # Add detailed table information
    output += f"## RELEVANT TABLES\n\n"
    
    for table in tables:
        # Table header with name
        output += f"### TABLE: {table.table_name}\n\n"
        
        # Table columns
        output += "#### Columns:\n"
        for col in table.columns_concise:
            output += f"- {col}\n"
        
        output += "\n"
        
        # Table relationships (only if they exist)
        if table.relationships_concise:
            output += "#### Relationships:\n"
            for rel in table.relationships_concise:
                output += f"- {rel}\n"
        
        output += "\n---\n\n"
    
    return output

async def analyze_table_with_llm_async(llm, query: str, table_info: ConciseDetailedTable) -> TableAnalysis:
    """
    Use Gemini to analyze which columns in a table are relevant to the query (async version).
    
    Args:
        llm: Initialized Gemini model
        query: The user's original query
        table_info: ConciseDetailedTable object containing table name, columns, and relationships
        
    Returns:
        TableAnalysis object with table name and relevant columns
    """
    # Create prompt for the LLM
    prompt = f"""
    # Database Table Analysis for SQL Query Generation

    ## Context and Task
    You are a senior database architect tasked with identifying exactly which columns from this table are necessary to answer the user's query. This analysis will be used to construct an SQL query.

    ## User Query
    {query}

    ## Table Information
    Table Name: {table_info.table_name}

    ### Available Columns
    {json.dumps(table_info.columns_concise, indent=4)}

    ### Relationships with Other Tables
    {json.dumps(table_info.relationships_concise, indent=4)}

    ## Selection Guidelines
    1. Include columns needed for SELECT, WHERE, JOIN, GROUP BY, ORDER BY clauses
    2. Include columns required for calculations or conditions mentioned in the query
    3. Include primary/foreign keys needed for table relationships if they'll be used in JOINs
    4. Exclude columns that provide no functional value for this specific query
    5. Provide one overall reasoning for why this table is relevant to the query

    ## Response Requirements
    Provide a JSON object that MUST follow this exact structure:
    ```json
    {{
        "table_name": "name_of_the_table",
        "relevant_columns": [
            {{
                "column_name": "name_of_the_column",
                "data_type": "string"
            }},
            {{
                "column_name": "name_of_the_column",
                "data_type": "string"
            }}
        ],
        "table_reasoning": "Single comprehensive explanation of why this table is relevant to the query and how the selected columns contribute to answering it"
    }}
    ```

    Only include columns genuinely needed to answer the query. If this table contributes no relevant columns to the query, return an empty "relevant_columns" array with appropriate explanation in table_reasoning.
    """
    
    # Run the LLM call in a thread pool to make it async
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, llm.invoke, prompt)
    response_text = response.content    
    print(f"LLM response for {table_info.table_name}: {response_text}")
    
    # Extract JSON from response
    try:
        # First try to find JSON content within code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
        else:
            # If no code blocks, try to parse the entire response as JSON
            data = json.loads(response_text)
        
        # Clean and validate the data before parsing into TableAnalysis
        if "relevant_columns" in data:
            # Remove any None entries or fix entries with missing fields
            cleaned_columns = []
            for col in data["relevant_columns"]:
                if col is None:
                    continue
                
                # Ensure all required fields exist with default values if missing
                valid_col = {
                    "column_name": col.get("column_name", "unknown_column"),
                    "data_type": col.get("data_type", "string")
                }
                cleaned_columns.append(valid_col)
            
            # Replace the original list with the cleaned list
            data["relevant_columns"] = cleaned_columns
        
        # Ensure table_reasoning exists
        if "table_reasoning" not in data:
            data["table_reasoning"] = "No reasoning provided"
        
        # Parse into TableAnalysis model
        return TableAnalysis(**data)
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing LLM response for {table_info.table_name}: {e}")
        print(f"Raw response text: {response_text[:500]}...")  # Print part of the raw response for debugging
        
        # Create a fallback TableAnalysis with empty columns
        return TableAnalysis(
            table_name=table_info.table_name,
            relevant_columns=[],
            table_reasoning="Error occurred during analysis"
        )

def get_column_descriptions(table_name: str, json_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Get column descriptions from the JSON data for a given table and enhance them with
    additional information about constraints (primary, required, etc.)
    
    Args:
        table_name: Name of the table to fetch column information for
        json_data: The loaded JSON data containing table information
        
    Returns:
        Dictionary mapping column names to their enhanced descriptions
    """
    # Check if tables exist in the JSON data
    if "tables" not in json_data:
        print("No tables found in the JSON data")
        return {}
    
    # Find the table in the JSON data
    table_data = None
    for table in json_data["tables"]:
        if table.get("table_name") == table_name:
            table_data = table
            break
    
    if not table_data:
        print(f"Table '{table_name}' not found in the JSON data")
        return {}
    
    # Create a dictionary mapping column names to enhanced descriptions
    enhanced_descriptions = {}
    for column in table_data.get("columns", []):
        # Start with the original description
        description = column.get("description", "")
        enhanced_descriptions[column.get("name")] = description
    
    return enhanced_descriptions

def generate_table_analysis_summary(query: str, table_analyses: List[TableAnalysis], json_data: Dict[str, Any]) -> str:
    """
    Format the analyzed table information into a comprehensive summary.
    Include enhanced column descriptions fetched from the JSON data.
    
    Args:
        query: The original user query
        table_analyses: List of TableAnalysis objects
        json_data: The loaded JSON data containing table information
        
    Returns:
        Formatted string with all relevant table and column information
    """
    formatted_output = f"# Database Analysis for Query\n\n"
    formatted_output += f"## Original Query\n{query}\n\n"
    formatted_output += f"## Relevant Tables and Columns\n\n"
    
    # Remove duplicate table names
    unique_table_names = list(set([analysis.table_name for analysis in table_analyses]))
    
    # Fetch enhanced column descriptions for unique tables from JSON data
    enhanced_descriptions = {}
    for table_name in unique_table_names:
        enhanced_descriptions.update(get_column_descriptions(table_name, json_data))
    
    for analysis in table_analyses:
        formatted_output += f"### Table: {analysis.table_name}\n\n"
        formatted_output += f"**Table Reasoning:** {analysis.table_reasoning}\n\n"
        
        if analysis.relevant_columns:
            formatted_output += "| Column | Data Type | Description |\n"
            formatted_output += "|--------|-----------|-------------|\n"
            
            for column in analysis.relevant_columns:
                # Get the enhanced description if available
                description = enhanced_descriptions.get(column.column_name, "")
                
                formatted_output += f"| {column.column_name} | {column.data_type} | {description} |\n"
        else:
            formatted_output += "No relevant columns identified for this table.\n"
        
        formatted_output += "\n"
    
    return formatted_output

def merge_table_analyses_to_json(query: str, table_analyses: List[TableAnalysis], json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge all table analyses into a structured JSON format without using LLM.
    
    Args:
        query: The original user query
        table_analyses: List of TableAnalysis objects
        json_data: The loaded JSON data containing table information
        
    Returns:
        Dictionary with merged analysis in structured JSON format
    """
    # Remove duplicate table names
    unique_table_names = list(set([analysis.table_name for analysis in table_analyses]))
    
    # Fetch enhanced column descriptions for unique tables from JSON data
    enhanced_descriptions = {}
    for table_name in unique_table_names:
        enhanced_descriptions.update(get_column_descriptions(table_name, json_data))
    
    # Build the merged structure
    merged_analysis = {
        "query": query,
        "analysis_timestamp": time.time(),
        "total_tables_analyzed": len(table_analyses),
        "tables": []
    }
    
    for analysis in table_analyses:
        table_info = {
            "table_name": analysis.table_name,
            "table_reasoning": analysis.table_reasoning,
            "columns": []
        }
        
        for column in analysis.relevant_columns:
            # Get the enhanced description if available
            description = enhanced_descriptions.get(column.column_name, "")
            
            column_info = {
                "column_name": column.column_name,
                "data_type": column.data_type,
                "description": description
            }
            table_info["columns"].append(column_info)
        
        merged_analysis["tables"].append(table_info)
    
    return merged_analysis

def extract_table_names_from_structured_output(structured_output: List[str]) -> List[str]:
    """
    Extract table names from the list output of the table_agent
    
    Args:
        structured_output: List of table names
        
    Returns:
        List of table names
    """
    try:
        # The output is already a list of table names, just return it
        return structured_output if isinstance(structured_output, list) else []
    except Exception as e:
        print(f"Unexpected error while processing table names: {e}")
        return []

async def process_tables_async(query: str, structured_output: str, json_file_path: str) -> Dict[str, str]:
    """
    Main async function to process table information and generate an SQL query.
    
    Args:
        query: The original user query
        structured_output: JSON string from table_agent containing the required tables
        json_file_path: Path to the JSON file containing table information
        
    Returns:
        Dictionary with table analyses and SQL query
    """
    try:
        # Initialize the LLM
        llm = initialize_llm_gemini_light()
        
        # Load the JSON data
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
        except UnicodeDecodeError:
            # Fallback to another encoding if UTF-8 fails
            with open(json_file_path, 'r', encoding='utf-8-sig') as file:
                json_data = json.load(file)
        
        # Extract table names from structured output
        table_names = extract_table_names_from_structured_output(structured_output)
        print(f"Analyzing tables: {', '.join(table_names)}")
        
        # Get detailed table information from JSON data
        concise_tables = get_detailed_tables_concise_output(query, table_names, json_data)
        
        # Analyze all tables in parallel
        tasks = []
        for table_info in concise_tables:
            task = analyze_table_with_llm_async(llm, query, table_info)
            tasks.append(task)
        
        # Wait for all analyses to complete
        table_analyses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions from the parallel processing
        valid_analyses = []
        for i, analysis in enumerate(table_analyses):
            if isinstance(analysis, Exception):
                print(f"Error analyzing table {concise_tables[i].table_name}: {analysis}")
                # Create a fallback analysis with empty columns
                fallback = TableAnalysis(
                    table_name=concise_tables[i].table_name,
                    relevant_columns=[],
                    table_reasoning="Error occurred during analysis"
                )
                valid_analyses.append(fallback)
            else:
                valid_analyses.append(analysis)
        
        # Format the table analyses using JSON data (traditional format)
        # formatted_analyses = generate_table_analysis_summary(query, valid_analyses, json_data)
        
        # Merge all analyses into structured JSON format
        merged_json = merge_table_analyses_to_json(query, valid_analyses, json_data)
        
        return {
            "merged_analysis_json": merged_json,
        }
    except Exception as e:
        print(f"Error in process_tables_async: {e}")
        # Return a minimal valid result to prevent cascading errors
        return {
            "table_analyses": f"# Database Analysis for Query\n\n## Original Query\n{query}\n\nError occurred during table analysis.",
            "merged_analysis_json": {
                "query": query,
                "analysis_timestamp": time.time(),
                "total_tables_analyzed": 0,
                "tables": [],
                "error": "Error occurred during table analysis"
            }
        }

def process_tables(query: str, structured_output: str, json_file_path: str) -> Dict[str, str]:
    """
    Synchronous wrapper for the async process_tables_async function.
    
    Args:
        query: The original user query
        structured_output: JSON string from table_agent containing the required tables
        json_file_path: Path to the JSON file containing table information
        
    Returns:
        Dictionary with table analyses and SQL query
    """
    # Run the async function in a new event loop
    return asyncio.run(process_tables_async(query, structured_output, json_file_path))

def column_agent(user_query: str, concise_table_result, json_file_path: str) -> Dict[str, str]:
    """
    Main function to get detailed table information based on user query.
    
    Args:
        user_query: The user's question or request
        concise_table_result: JSON string from table_agent containing the required tables
        json_file_path: Path to the JSON file containing table information
        
    Returns:
        Dictionary with table analyses information
    """
    
    return process_tables(user_query, concise_table_result, json_file_path)

# Example usage
if __name__ == "__main__":
    # Path to the JSON file containing table information
    json_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               "all_tables_with_descriptions_v1.json")
    JSON_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               "sub_intent_classification.json")
    SCHEMA_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               "all_tables_with_descriptions_v1.json")

    # Test query
    test_query = "Show me pending expenses for project id 7 additional info: pending means adastatus will be approved and pma status will be pending"
    
    try:
        # Load the JSON data
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
        except UnicodeDecodeError:
            # Fallback to another encoding if UTF-8 fails
            with open(json_file_path, 'r', encoding='utf-8-sig') as file:
                json_data = json.load(file)
        
        # 1. Get sub-intent result
        from subIntent_agent import get_subintent_tables_sync
        sub_intent_result = get_subintent_tables_sync(test_query, JSON_FILE_PATH)
        print("Sub-intent result:", sub_intent_result)
        
        # 2. Get structured output from table_agent
        structured_output = table_agent(test_query, sub_intent_result, SCHEMA_JSON_PATH)

        print("Structured output from table_agent:", structured_output)
        
        # 3. Process tables with column_agent
        start_time = time.time()

        result = column_agent(test_query, structured_output, json_file_path)
        end_time = time.time()
        print("time ", end_time - start_time)

        
        print("\n" + "="*80)
        print("MERGED JSON ANALYSIS:")
        print("="*80)
        print(json.dumps(result["merged_analysis_json"], indent=2))
    except Exception as e:
        print(f"Error: {e}")