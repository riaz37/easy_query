#!/usr/bin/env python3
"""
Navigation Tool for Pipecat voice agent.
Handles page navigation and page element interactions with context awareness.
"""

import json
import aiohttp
import asyncio
from typing import Dict, Any, Optional, Set, List
from pipecat.processors.frameworks.rtvi import RTVIProcessor
from pipecat.adapters.schemas.function_schema import FunctionSchema
from loguru import logger
from .base_tool import BaseTool

# Import the data types function directly to avoid SSL issues
try:
    from db_manager.utilites.new_table_creation import get_supported_sql_server_types
except ImportError:
    # Fallback if direct import fails
    get_supported_sql_server_types = None

# Import the user tables function directly to avoid HTTP connection issues
try:
    from db_manager.utilites.new_table_creation import get_user_created_tables
except ImportError:
    # Fallback if direct import fails
    get_user_created_tables = None


class NavigationTool(BaseTool):
    """Navigation tool that handles page navigation and element interactions with context awareness."""
    
    # Define valid pages and their interactive elements
    VALID_PAGES = {
        "dashboard": {"buttons": [], "forms": []},
        "database-query": {
            "buttons": ["view report", "report generation"], 
            "forms": [], 
            "search_enabled": True
        },
        "file-query": {"buttons": [], "forms": [], "file_search_enabled": True, "file_upload_enabled": True},
        "company-structure": {"buttons": [], "forms": []},
        "user-configuration": {"buttons": ["configure database", "configure business rule"], "forms": []},
        "tables": {
            "buttons": [
                "load tables", 
                "reload tables", 
                "table visualization", 
                "excel-import", 
                "get ai suggestion", 
                "continue to import",  # Added for Excel import workflow
                "back to mapping",  # Added for Excel import workflow
                "import data to database",  # Added for Excel import workflow
                "submit", 
                "set database",
                "create new table",  # Added create new table button
                "table management",  # Added table management button
                "manage business rule"  # Added manage business rule button
            ], 
            "forms": [],
            "table_creation_enabled": True,  # Enable table creation functionality
            "excel_import_enabled": True  # Enable Excel import functionality
        },
        "users": {"buttons": ["manage database access", "vector database access"], "forms": []}
    }
    
    def __init__(self, rtvi_processor: RTVIProcessor, task=None, initial_current_page: str = "dashboard"):
        super().__init__(rtvi_processor, task)
        # Session-based state management (per user)
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.initial_current_page = initial_current_page
        
        # Table creation sessions for each user
        self.table_creation_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Excel import sessions for each user
        self.excel_import_sessions: Dict[str, Dict[str, Any]] = {}
        
    def get_tool_definition(self) -> FunctionSchema:
        """Get the tool definition for the LLM using Pipecat's FunctionSchema."""
        return FunctionSchema(
            name="navigate_page",
            description="Navigate between pages, interact with page elements, perform database searches, file system operations, and table creation. Use 'navigate' when going to a different page, 'click' for same-page interactions, 'search' for database queries, 'file_search' for file system searches, 'file_upload' for file uploads, 'create_table' for starting table creation, 'add_column' for adding columns to a table, 'set_column_properties' for configuring column properties.",
            properties={
                "user_id": {
                    "type": "string",
                    "description": "User ID for session management"
                },
                "target": {
                    "type": "string",
                    "description": "Target page name, element name, search query, file operation, or table creation action (e.g., 'dashboard', 'user-configuration', 'configure database', 'Show all employee salaries', 'Find documents about project guidelines', 'users table', 'column one', 'nullable true')"
                },
                "action_type": {
                    "type": "string",
                    "enum": ["navigate", "click", "interact", "search", "file_search", "file_upload", "view_report", "generate_report", "create_table", "add_column", "set_column_properties", "complete_table", "table_status", "confirm_table_creation", "page_info", "update_table_name", "update_column_name", "update_column_properties", "submit_table", "update", "form_update", "table_management", "business_rule_management", "excel_import", "select_table", "list_available_tables", "get_ai_suggestion", "continue_to_import", "back_to_mapping", "import_data_to_database"],
                    "description": "Type of action: 'navigate' for going to a different page, 'click' for clicking buttons on same page, 'interact' for other interactions, 'search' for database queries, 'file_search' for file system searches, 'file_upload' for file uploads, 'view_report' for viewing existing reports, 'generate_report' for creating new reports, 'create_table' for starting table creation, 'add_column' for adding columns, 'set_column_properties' for configuring column properties, 'complete_table' for finalizing table creation, 'table_status' for checking table creation progress, 'confirm_table_creation' for confirming table creation, 'page_info' for getting information about what can be done on a page, 'update_table_name' for changing table name, 'update_column_name' for changing column name, 'update_column_properties' for updating column properties, 'submit_table' for submitting table creation, 'update' for general update actions, 'form_update' for form updating actions, 'table_management' for managing table operations, 'business_rule_management' for managing business rules, 'excel_import' for starting Excel import process, 'select_table' for selecting target table for Excel import, 'list_available_tables' for showing available tables for Excel import, 'get_ai_suggestion' for getting AI suggestions, 'continue_to_import' for proceeding with import, 'back_to_mapping' for returning to mapping step, 'import_data_to_database' for finalizing import"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context or parameters for the action (e.g., table names, file descriptions, column data types, column properties)"
                }
            },
            required=["user_id", "target", "action_type"]
        )
    
    async def execute(self, user_id: str, target: str, action_type: str, context: str = None, **kwargs) -> str:
        """Execute the navigation action."""
        try:
            # Initialize user session if not exists
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = {
                    "current_page": self.initial_current_page,  # Use initial current page
                    "previous_page": None,
                    "interaction_history": []
                }
            
            user_session = self.user_sessions[user_id]
            current_page = user_session["current_page"]
            
            logger.info(f"🧭 Navigation request - User: {user_id}, Target: {target}, Action: {action_type}, Current Page: {current_page}")
            
            if action_type == "navigate":
                result = await self._handle_page_navigation(user_id, target, current_page, action_type, context)
            elif action_type in ["click", "interact"]:
                result = await self._handle_page_interaction(user_id, target, action_type, current_page, action_type, context)
            elif action_type == "search":
                result = await self._handle_database_search(user_id, target, current_page, action_type, context)
            elif action_type == "file_search":
                result = await self._handle_file_search(user_id, target, current_page, action_type, context)
            elif action_type == "file_upload":
                result = await self._handle_file_upload(user_id, target, current_page, action_type, context)
            elif action_type == "view_report":
                result = await self._handle_view_report(user_id, target, current_page, action_type, context)
            elif action_type == "generate_report":
                result = await self._handle_generate_report(user_id, target, current_page, action_type, context)
            elif action_type == "create_table":
                result = await self._handle_create_table(user_id, target, current_page, action_type, context)
            elif action_type == "add_column":
                result = await self._handle_add_column(user_id, target, current_page, action_type, context)
            elif action_type == "set_column_properties":
                result = await self._handle_set_column_properties(user_id, target, current_page, action_type, context)
            elif action_type == "complete_table":
                result = await self._handle_complete_table(user_id, current_page, action_type, context)
            elif action_type == "table_status":
                result = await self._handle_table_status(user_id, current_page, action_type, context)
            elif action_type == "confirm_table_creation":
                result = await self._handle_confirm_table_creation(user_id, current_page, action_type, context)
            elif action_type == "page_info":
                result = await self._handle_page_info(user_id, target, current_page, action_type, context)
            elif action_type == "update_table_name":
                result = await self._handle_update_table_name(user_id, target, current_page, action_type, context)
            elif action_type == "update_column_name":
                result = await self._handle_update_column_name(user_id, target, current_page, action_type, context)
            elif action_type == "update_column_properties":
                result = await self._handle_update_column_properties(user_id, target, current_page, action_type, context)
            elif action_type == "submit_table":
                result = await self._handle_submit_table(user_id, current_page, action_type, context)
            elif action_type == "update":
                result = await self._handle_general_update(user_id, target, current_page, action_type, context)
            elif action_type == "form_update":
                result = await self._handle_general_update(user_id, target, current_page, action_type, context)
            elif action_type == "table_management":
                result = await self._handle_table_management(user_id, current_page, action_type, context)
            elif action_type == "business_rule_management":
                result = await self._handle_business_rule_management(user_id, current_page, action_type, context)
            elif action_type == "excel_import":
                result = await self._handle_excel_import(user_id, current_page, action_type, context)
            elif action_type == "select_table":
                result = await self._handle_select_table(user_id, current_page, action_type, context)
            elif action_type == "list_available_tables":
                result = await self._handle_list_available_tables(user_id, current_page, action_type, context)
            elif action_type == "get_ai_suggestion":
                result = await self._handle_get_ai_suggestion(user_id, current_page, action_type, context)
            elif action_type == "continue_to_import":
                result = await self._handle_continue_to_import(user_id, current_page, action_type, context)
            elif action_type == "back_to_mapping":
                result = await self._handle_back_to_mapping(user_id, current_page, action_type, context)
            elif action_type == "import_data_to_database":
                result = await self._handle_import_data_to_database(user_id, current_page, action_type, context)
            else:
                return f"Invalid action type: {action_type}. Use 'navigate', 'click', 'interact', 'search', 'file_search', 'file_upload', 'view_report', 'generate_report', 'create_table', 'add_column', 'set_column_properties', 'complete_table', 'table_status', 'confirm_table_creation', 'page_info', 'update_table_name', 'update_column_name', 'update_column_properties', 'submit_table', 'update', 'form_update', 'table_management', 'business_rule_management', 'excel_import', 'select_table', 'list_available_tables', 'get_ai_suggestion', 'continue_to_import', 'back_to_mapping', or 'import_data_to_database'."
            
            return result
                
        except Exception as e:
            logger.error(f"Error executing navigation tool: {e}")
            return f"Error processing navigation request: {str(e)}"
    
    async def _handle_page_navigation(self, user_id: str, target_page: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle navigation to a different page."""
        # Normalize page name
        target_page = target_page.lower().replace(" ", "-").replace("_", "-")
        
        # Validate target page
        if target_page not in self.VALID_PAGES:
            available_pages = ", ".join(self.VALID_PAGES.keys())
            return f"Invalid page '{target_page}'. Available pages: {available_pages}"
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["previous_page"] = current_page
        user_session["current_page"] = target_page
        user_session["interaction_history"].append({
            "action": "navigate",
            "from": current_page,
            "to": target_page,
            "timestamp": self._get_timestamp()
        })
        
        # Determine correct Action_type based on original action_type
        # Page navigation always returns "navigation"
        action_type_output = "navigation"

        # Create unified structured output
        structured_output = {
            "Action_type": action_type_output,
            "param": "name",
            "value": target_page,
            "page": target_page,
            "previous_page": current_page,
            "interaction_type": "page_navigation",
            "clicked": False,
            "element_name": None,
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None
        }
        
        logger.info(f"🧭 Page navigation: {current_page} → {target_page}")
        logger.info(f"🧭 Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="navigate",
            data=structured_output
        )
        
        return f"Navigated to {target_page} page."
    
    async def _handle_page_interaction(self, user_id: str, element_name: str, action_type: str, current_page: str, original_action_type: str, context: str = None) -> str:
        """Handle interaction with page elements (buttons, forms, etc.)."""
        # Check if current page has interactive elements
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        available_buttons = page_config.get("buttons", [])
        
        # Normalize element name for comparison
        element_name_normalized = element_name.lower()
        
        # Check if element exists on current page
        element_found = False
        matched_element = None
        
        for button in available_buttons:
            if button.lower() == element_name_normalized or element_name_normalized in button.lower():
                element_found = True
                matched_element = button
                break
        
        if not element_found:
            if available_buttons:
                available_elements = ", ".join(available_buttons)
                return f"Element '{element_name}' not found on page '{current_page}'. Available elements: {available_elements}"
            else:
                return f"Page '{current_page}' has no interactive elements."
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": action_type,
            "element": matched_element,
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })

        # Determine correct Action_type based on original action_type
        action_type_output = "clicked" if original_action_type in ["click", "interact"] else "navigation"

        # Handle special case for "set database" button that requires db_id
        if matched_element == "set database":
            # Parse db_id from context - MANDATORY for set database
            db_id = None
            if context:
                # Look for db_id in context (e.g., "db_id:123" or "database_id:456")
                import re
                db_id_match = re.search(r'db_id:(\w+)', context)
                if db_id_match:
                    db_id = db_id_match.group(1)
                else:
                    # Try alternative format
                    db_id_match = re.search(r'database_id:(\w+)', context)
                    if db_id_match:
                        db_id = db_id_match.group(1)
            
            # Check if db_id is provided - MANDATORY for set database
            if not db_id:
                return f"Error: 'set database' button requires a database ID. Please provide db_id in context (e.g., 'db_id:123')."

            # Create unified structured output for set database
            structured_output = {
                "Action_type": action_type_output,
                "param": "clicked,name,db_id",
                "value": f"true,{matched_element},{db_id}",
                "page": current_page,
                "previous_page": user_session.get("previous_page"),
                "interaction_type": "button_click",
                "clicked": True,
                "element_name": matched_element,
                "search_query": None,
                "report_request": None,
                "report_query": None,
                "upload_request": None,
                "db_id": db_id,
                "table_specific": False,
                "tables": [],
                "file_descriptions": [],
                "table_names": [],
                "context": context,
                "timestamp": self._get_timestamp(),
                "user_id": user_id,
                "success": True,
                "error_message": None
            }
        else:
            # Create unified structured output for regular buttons
            structured_output = {
                "Action_type": action_type_output,
                "param": "clicked,name",
                "value": f"true,{matched_element}",
                "page": current_page,
                "previous_page": user_session.get("previous_page"),
                "interaction_type": "button_click",
                "clicked": True,
                "element_name": matched_element,
                "search_query": None,
                "report_request": None,
                "report_query": None,
                "upload_request": None,
                "db_id": None,
                "table_specific": False,
                "tables": [],
                "file_descriptions": [],
                "table_names": [],
                "context": context,
                "timestamp": self._get_timestamp(),
                "user_id": user_id,
                "success": True,
                "error_message": None
            }
        
        logger.info(f"🧭 Page interaction: {action_type} '{matched_element}' on page '{current_page}'")
        logger.info(f"🧭 Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action=action_type,
            data=structured_output
        )
        
        return f"Performed {action_type} on '{matched_element}' on page '{current_page}'."
    
    async def _handle_database_search(self, user_id: str, search_query: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle database search when user is on database-query page."""
        # Check if current page supports database search
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("search_enabled", False):
            return f"Database search is not available on page '{current_page}'. Please navigate to the database-query page first."
        
        if not search_query or not search_query.strip():
            return "Please provide a search query for the database."
        
        # Store the search query
        search_query = search_query.strip()
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "search",
            "query": search_query,
            "page": current_page,
            "timestamp": self._get_timestamp()
        })

        # Determine correct Action_type based on original action_type
        action_type_output = "clicked"  # For search actions, use clicked (search button/form submission)

        # Create unified structured output for database search
        structured_output = {
            "Action_type": action_type_output,
            "param": "search,question",
            "value": f"true,{search_query}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "database_search",
            "clicked": True,  # Search is a user interaction (button/form submission)
            "element_name": "search",
            "search_query": search_query,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None
        }
        
        logger.info(f"🔍 Database search executed on page '{current_page}' with query: {search_query}")
        logger.info(f"🔍 Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="search",
            data=structured_output
        )
        
        return f"I've processed your database search: '{search_query}'. The search request has been sent to the MSSQL database."
    
    async def _handle_file_search(self, user_id: str, search_query: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle file system search when user is on file-query page."""
        # Check if current page supports file search
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("file_search_enabled", False):
            return f"File search is not available on page '{current_page}'. Please navigate to the file-query page first."
        
        if not search_query or not search_query.strip():
            return "Please provide a search query for the file system."
        
        # Store the search query
        search_query = search_query.strip()
        
        # Parse context for table-specific information
        # Parse context for agent decisions about table information
        table_specific = False
        tables = []
        
        if context:
            # Parse structured context from agent (e.g., "table_specific:true,tables:finance,hr")
            context_parts = context.split(',')
            for part in context_parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'table_specific':
                        table_specific = value.lower() == 'true'
                    elif key == 'tables':
                        # Split table names by comma or pipe
                        if value:
                            tables = [t.strip() for t in value.replace('|', ',').split(',') if t.strip()]
        
        # If no explicit agent decision but context mentions tables, fall back to natural language parsing
        if not table_specific and context:
            context_lower = context.lower()
            if "table" in context_lower or "tables" in context_lower:
                table_specific = True
                # Extract table names from context (fallback parsing)
                import re
                # Look for patterns like "finance table", "table finance", "in finance table"
                table_matches = re.findall(r'(?:table\s+(\w+)|(\w+)\s+table|in\s+(\w+)\s+table)', context_lower)
                if table_matches:
                    # Flatten the matches and filter out empty strings
                    tables = [match for group in table_matches for match in group if match]
        
        # If no tables found but table_specific is True, use default
        if table_specific and not tables:
            tables = ["string"]
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "file_search",
            "query": search_query,
            "table_specific": table_specific,
            "tables": tables,
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })

        # Determine correct Action_type based on original action_type
        action_type_output = "clicked"  # For file search actions, use clicked (search button/form submission)

        # Create unified structured output for file search
        structured_output = {
            "Action_type": action_type_output,
            "param": "query,table_specific,tables[]",
            "value": f"{search_query},{str(table_specific).lower()},{tables}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "file_search",
            "clicked": True,  # File search is a user interaction (button/form submission)
            "element_name": "search",
            "search_query": search_query,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": table_specific,
            "tables": tables,
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None
        }
        
        logger.info(f"🔍 File search executed on page '{current_page}' with query: {search_query}")
        logger.info(f"🔍 Table specific: {table_specific}, Tables: {tables}")
        logger.info(f"🔍 Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="file_search",
            data=structured_output
        )
        
        table_info = f" in tables {', '.join(tables)}" if table_specific else ""
        return f"I've processed your file search: '{search_query}'{table_info}. The vector search request has been sent to the file system database."
    
    async def _handle_file_upload(self, user_id: str, upload_request: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle file system upload when user is on file-query page."""
        # Check if current page supports file upload
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("file_upload_enabled", False):
            return f"File upload is not available on page '{current_page}'. Please navigate to the file-query page first."
        
        if not upload_request or not upload_request.strip():
            return "Please provide an upload request for the file system."
        
        # Store the upload request
        upload_request = upload_request.strip()
        
        # Parse context for agent decisions about file descriptions and table names
        file_descriptions = []
        table_names = []
        
        if context:
            # Parse structured context from agent (e.g., "file_descriptions:numeric data|text file,table_names:finance|hr")
            context_parts = context.split(',')
            for part in context_parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'file_descriptions':
                        # Split file descriptions by pipe or comma
                        if value:
                            file_descriptions = [desc.strip() for desc in value.replace('|', ',').split(',') if desc.strip()]
                    elif key == 'table_names':
                        # Split table names by pipe or comma
                        if value:
                            table_names = [table.strip() for table in value.replace('|', ',').split(',') if table.strip()]
        
        # If no explicit agent decision but context mentions descriptions/tables, fall back to natural language parsing
        if not file_descriptions and not table_names and context:
            context_lower = context.lower()
            
            # Extract file descriptions (fallback)
            if "description" in context_lower or "file" in context_lower:
                # Simple extraction - use upload request as description
                file_descriptions = [upload_request]
            
            # Extract table names (fallback parsing)
            import re
            # Look for patterns like "finance table", "table finance", "in finance table"
            table_matches = re.findall(r'(?:table\s+(\w+)|(\w+)\s+table|in\s+(\w+)\s+table)', context_lower)
            if table_matches:
                # Flatten the matches and filter out empty strings
                table_names = [match for group in table_matches for match in group if match]
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "file_upload",
            "upload_request": upload_request,
            "file_descriptions": file_descriptions,
            "table_names": table_names,
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })

        # Determine correct Action_type based on original action_type
        action_type_output = "clicked"  # For file upload actions, use clicked (upload button/form submission)

        # Create unified structured output for file upload
        structured_output = {
            "Action_type": action_type_output,
            "param": "file_descriptions[],table_names[]",
            "value": f"{file_descriptions},{table_names}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "file_upload",
            "clicked": True,  # File upload is a user interaction (button/form submission)
            "element_name": "upload",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": upload_request,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": file_descriptions,
            "table_names": table_names,
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None
        }
        
        logger.info(f"📤 File upload executed on page '{current_page}' with request: {upload_request}")
        logger.info(f"📤 File descriptions: {file_descriptions}, Table names: {table_names}")
        logger.info(f"📤 Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="file_upload",
            data=structured_output
        )
        
        file_info = f" {len(file_descriptions)} file(s)" if file_descriptions else " files"
        table_info = f" to table(s) {', '.join(table_names)}" if table_names else ""
        return f"I've processed your upload request for{file_info}{table_info}. The upload request has been sent to the file system."
    
    async def _handle_view_report(self, user_id: str, report_request: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle view report action when user is on database-query page."""
        # Check if current page supports report viewing
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        available_buttons = page_config.get("buttons", [])
        
        if "view report" not in available_buttons:
            return f"View report is not available on page '{current_page}'. Please navigate to the database-query page first."
        
        if not report_request or not report_request.strip():
            return "Please provide a report request to view."
        
        # Store the report request
        report_request = report_request.strip()
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "view_report",
            "report_request": report_request,
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })

        # Determine correct Action_type based on original action_type
        action_type_output = "clicked" if action_type == "view_report" else "navigation"

        # Create unified structured output for view report
        structured_output = {
            "Action_type": action_type_output,
            "param": "clicked,name,report_request",
            "value": f"true,view report,{report_request}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "view_report",
            "clicked": True,
            "element_name": "view report",
            "search_query": None,
            "report_request": report_request,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None
        }
        
        logger.info(f"📊 View report executed on page '{current_page}' with request: {report_request}")
        logger.info(f"📊 Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="view_report",
            data=structured_output
        )
        
        return f"I've processed your view report request: '{report_request}'. The report viewing request has been sent to the system."
    
    async def _handle_generate_report(self, user_id: str, report_query: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle report generation action when user is on database-query page."""
        # Check if current page supports report generation
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        available_buttons = page_config.get("buttons", [])
        
        if "report generation" not in available_buttons:
            return f"Report generation is not available on page '{current_page}'. Please navigate to the database-query page first."
        
        if not report_query or not report_query.strip():
            return "Please provide a query for report generation."
        
        # Store the report query
        report_query = report_query.strip()
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "generate_report",
            "report_query": report_query,
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })

        # Determine correct Action_type based on original action_type
        action_type_output = "clicked" if action_type == "generate_report" else "navigation"

        # Create unified structured output for report generation
        structured_output = {
            "Action_type": action_type_output,
            "param": "clicked,name,report_query",
            "value": f"true,report generation,{report_query}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "generate_report",
            "clicked": True,
            "element_name": "report generation",
            "search_query": None,
            "report_request": None,
            "report_query": report_query,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None
        }
        
        logger.info(f"📊 Report generation executed on page '{current_page}' with query: {report_query}")
        logger.info(f"📊 Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="generate_report",
            data=structured_output
        )
        
        return f"I've processed your report generation request: '{report_query}'. The report generation request has been sent to the system."
    
    async def _handle_create_table(self, user_id: str, table_name: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle table creation action."""
        # Check if table creation is enabled on the current page
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("table_creation_enabled", False):
            return f"Table creation is not available on page '{current_page}'. Please navigate to the tables page first."
        
        if not table_name or not table_name.strip():
            return "Please provide a table name for creation."
        
        # Store the table name
        table_name = table_name.strip()
        
        # Initialize table creation session if not exists
        if user_id not in self.table_creation_sessions:
            self.table_creation_sessions[user_id] = {
                "table_name": table_name,
                "schema": "dbo",
                "columns": [],
                "current_step": "table_name",
                "is_completed": False
            }
        else:
            # If already in a table creation session, update the table name
            self.table_creation_sessions[user_id]["table_name"] = table_name
            self.table_creation_sessions[user_id]["columns"] = [] # Clear previous columns
            self.table_creation_sessions[user_id]["current_step"] = "table_name"
            self.table_creation_sessions[user_id]["is_completed"] = False

        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "create_table",
            "table_name": table_name,
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })

        # Determine correct Action_type based on original action_type
        action_type_output = "form_fillup"  # Table creation initiation is form filling

        # Create unified structured output for table creation
        structured_output = {
            "Action_type": action_type_output,
            "param": "form_fillup,name,table_name",
            "value": f"true,create new table,{table_name}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "table_creation",
            "clicked": False,  # Not clicked, it's form filling
            "element_name": "create new table",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None,
            # Include table creation session info for frontend
            "table_creation_session": {
                "table_name": table_name,
                "schema": "dbo",
                "columns": [],
                "current_step": "table_name",
                "is_completed": False
            }
        }
        
        logger.info(f"✅ Table creation started: '{table_name}' on page '{current_page}'")
        logger.info(f"✅ Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="create_table",
            data=structured_output
        )
        
        return f"✅ Started creating new table '{table_name}' with schema 'dbo'. You can now add columns using 'add_column' action."
    
    async def _handle_add_column(self, user_id: str, column_name: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle adding a column to the table."""
        table_creation_session = self.table_creation_sessions.get(user_id)
        if not table_creation_session:
            return "You are not currently in a table creation session. Please start a new table creation session first."
        
        if table_creation_session["is_completed"]:
            return "Table creation is already completed. You can't add more columns."
        
        if not column_name or not column_name.strip():
            return "Please provide a column name for addition."
        
        # Store the column name
        column_name = column_name.strip()
        
        # Check for duplicate column names
        existing_columns = [col["name"].lower() for col in table_creation_session["columns"]]
        if column_name.lower() in existing_columns:
            return f"❌ Column name '{column_name}' already exists. Please use a different name."
        
        # Parse context for user-specified data type and properties
        user_data_type = None
        user_properties = {}
        comment = ""
        
        if context:
            context_parts = context.split(',')
            for part in context_parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key.lower() in ['data_type', 'type', 'datatype']:
                        user_data_type = value.upper()
                    elif key.lower() in ['nullable', 'is_primary_key', 'is_identity']:
                        user_properties[key.lower()] = value.lower() == 'true'
        
        # Determine data type: user-specified takes priority over auto-detection
        if user_data_type:
            # Validate user-specified data type
            available_types = await self._get_available_data_types()
            if user_data_type in available_types:
                final_data_type = user_data_type
                comment = f"User-specified as {user_data_type}"
            else:
                # Try to find closest match
                closest_match = self._find_closest_data_type(user_data_type, available_types)
                if closest_match:
                    final_data_type = closest_match
                    comment = f"Auto-corrected from '{user_data_type}' to '{closest_match}'"
                else:
                    # Get available types for better error message
                    available_types_str = ", ".join(available_types[:10])  # Show first 10 types
                    if len(available_types) > 10:
                        available_types_str += f" and {len(available_types) - 10} more..."
                    return f"Invalid data type '{user_data_type}'. Available types: {available_types_str}"
        else:
            # Auto-detect data type based on column name
            final_data_type = self._auto_detect_data_type(column_name)
            comment = f"Auto-detected as {final_data_type}"
        
        # Add the column to the session
        table_creation_session["columns"].append({
            "name": column_name,
            "data_type": final_data_type,
            "nullable": user_properties.get('nullable', True),  # User-specified or default
            "is_primary_key": user_properties.get('is_primary_key', False),  # User-specified or default
            "is_identity": user_properties.get('is_identity', False),  # User-specified or default
            "comment": comment
        })
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "add_column",
            "column_name": column_name,
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })

        # Determine correct Action_type based on original action_type
        action_type_output = "form_update"  # Adding columns is form updating

        # Create unified structured output for adding column
        structured_output = {
            "Action_type": action_type_output,
            "param": "form_update,name,column_name",
            "value": f"true,add_column,{column_name}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "table_creation",
            "clicked": False,  # Not clicked, it's form filling
            "element_name": "add_column",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None,
            # Include complete table creation session info for frontend
            "table_creation_session": {
                "table_name": table_creation_session["table_name"],
                "schema": table_creation_session["schema"],
                "columns": table_creation_session["columns"],  # Include all columns
                "current_step": "adding_columns",
                "is_completed": False
            }
        }
        
        logger.info(f"🔗 Column '{column_name}' added to table '{table_creation_session['table_name']}' on page '{current_page}'")
        logger.info(f"🔗 Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="add_column",
            data=structured_output
        )
        
        # Create informative return message
        if user_data_type:
            return f"✅ Column '{column_name}' added to table '{table_creation_session['table_name']}' with data type '{final_data_type}' (user-specified). Please proceed with setting additional properties or adding another column."
        else:
            return f"✅ Column '{column_name}' added to table '{table_creation_session['table_name']}' with auto-detected data type '{final_data_type}'. Please proceed with setting properties or adding another column."
    
    async def _handle_set_column_properties(self, user_id: str, column_name: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle setting properties for a column."""
        table_creation_session = self.table_creation_sessions.get(user_id)
        if not table_creation_session:
            return "You are not currently in a table creation session. Please start a new table creation session first."
        
        if table_creation_session["is_completed"]:
            return "Table creation is already completed. You can't set properties for more columns."
        
        if not column_name or not column_name.strip():
            return "Please provide a column name for property setting."
        
        # Store the column name
        column_name = column_name.strip()
        
        # Find the column in the session
        column_to_update = None
        for col in table_creation_session["columns"]:
            if col["name"].lower() == column_name.lower():
                column_to_update = col
                break
        
        if not column_to_update:
            return f"Column '{column_name}' not found in the current table creation session."
        
        # Parse context for properties
        properties_to_set = {}
        if context:
            context_parts = context.split(',')
            for part in context_parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    properties_to_set[key] = value
        
        # Apply properties to the column
        if "nullable" in properties_to_set:
            column_to_update["nullable"] = properties_to_set["nullable"].lower() == "true"
        if "is_primary_key" in properties_to_set:
            column_to_update["is_primary_key"] = properties_to_set["is_primary_key"].lower() == "true"
        if "is_identity" in properties_to_set:
            column_to_update["is_identity"] = properties_to_set["is_identity"].lower() == "true"
        if "data_type" in properties_to_set:
            # Validate data type against available types
            available_types = await self._get_available_data_types()
            if properties_to_set["data_type"] in available_types:
                column_to_update["data_type"] = properties_to_set["data_type"]
            else:
                # Try to find closest match
                closest_match = self._find_closest_data_type(properties_to_set["data_type"], available_types)
                if closest_match:
                    column_to_update["data_type"] = closest_match
                    logger.info(f"🔍 Auto-corrected data type '{properties_to_set['data_type']}' to '{closest_match}'")
                    # Update the comment to reflect the auto-correction
                    column_to_update["comment"] = f"Auto-corrected from '{properties_to_set['data_type']}' to '{closest_match}'"
                else:
                    # Get available types for better error message
                    available_types_str = ", ".join(available_types[:10])  # Show first 10 types
                    if len(available_types) > 10:
                        available_types_str += f" and {len(available_types) - 10} more..."
                    return f"Invalid data type '{properties_to_set['data_type']}'. Available types: {available_types_str}"
        if "comment" in properties_to_set:
            column_to_update["comment"] = properties_to_set["comment"]
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "set_column_properties",
            "column_name": column_name,
            "properties": properties_to_set,
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })

        # Determine correct Action_type based on original action_type
        action_type_output = "form_update"  # Setting properties is form updating

        # Create unified structured output for setting column properties
        structured_output = {
            "Action_type": action_type_output,
            "param": "form_update,name,column_name,properties",
            "value": f"true,set_column_properties,{column_name},{properties_to_set}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "table_creation",
            "clicked": False,  # Not clicked, it's form updating
            "element_name": "set_column_properties",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None,
            # Include complete table creation session info for frontend
            "table_creation_session": {
                "table_name": table_creation_session["table_name"],
                "schema": table_creation_session["schema"],
                "columns": table_creation_session["columns"],  # Include all columns
                "current_step": "setting_properties",
                "is_completed": False
            }
        }
        
        logger.info(f"🔧 Properties for column '{column_name}' set on page '{current_page}'")
        logger.info(f"🔧 Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="set_column_properties",
            data=structured_output
        )
        
        return f"Properties for column '{column_name}' have been set. Please proceed with adding another column or completing the table."
    
    async def _handle_complete_table(self, user_id: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle completing the table creation."""
        # Check if table creation is enabled on the current page
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("table_creation_enabled", False):
            return f"Table creation is not available on page '{current_page}'. Please navigate to the tables page first."
        
        # Check if user is confirming or just requesting completion
        if context and any(keyword in context.lower() for keyword in ['confirm', 'yes', 'okay', 'proceed']):
            # User confirmed, proceed with table creation
            result = await self.complete_table_creation(user_id, confirm=True)
        else:
            # User requested completion, ask for confirmation
            result = await self.complete_table_creation(user_id, confirm=False)
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "complete_table",
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })
        
        # Create unified structured output for table completion
        structured_output = {
            "Action_type": "clicked",
            "param": "clicked,name,action",
            "value": f"true,complete table,{action_type}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "table_completion",
            "clicked": True,
            "element_name": "complete table",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None
        }
        
        logger.info(f"✅ Table creation completed on page '{current_page}'")
        logger.info(f"✅ Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="complete_table",
            data=structured_output
        )
        
        return result
    
    async def _handle_table_status(self, user_id: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle getting table creation status."""
        # Check if table creation is enabled on the current page
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("table_creation_enabled", False):
            return f"Table creation is not available on page '{current_page}'. Please navigate to the tables page first."
        
        # Get table creation status
        status_message = self.get_table_creation_status_message(user_id)
        summary = self.get_table_creation_summary(user_id)
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "table_status",
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })
        
        # Create unified structured output for table status
        structured_output = {
            "Action_type": "clicked",
            "param": "clicked,name,action",
            "value": f"true,table status,{action_type}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "table_status",
            "clicked": True,
            "element_name": "table status",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None,
            "table_creation_status": summary
        }
        
        logger.info(f"📊 Table creation status requested on page '{current_page}'")
        logger.info(f"📊 Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="table_status",
            data=structured_output
        )
        
        return status_message
    
    async def _handle_confirm_table_creation(self, user_id: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle confirming table creation."""
        # Check if table creation is enabled on the current page
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("table_creation_enabled", False):
            return f"Table creation is not available on page '{current_page}'. Please navigate to the tables page first."
        
        # Complete the table creation with confirmation
        result = await self.complete_table_creation(user_id, confirm=True)
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "confirm_table_creation",
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })
        
        # Create unified structured output for table confirmation
        structured_output = {
            "Action_type": "clicked",  # Confirmation is a click action
            "param": "clicked,name,action",
            "value": f"true,confirm table creation,{action_type}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "table_confirmation",
            "clicked": True,  # Confirmation is clicked
            "element_name": "confirm table creation",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None
        }
        
        logger.info(f"✅ Table creation confirmed on page '{current_page}'")
        logger.info(f"✅ Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="confirm_table_creation",
            data=structured_output
        )
        
        return result
    
    async def _handle_page_info(self, user_id: str, target: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle getting page information."""
        # Get page information (not session-specific)
        page_info = self.get_page_information(current_page)
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "page_info",
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })
        
        # Create unified structured output for page info
        structured_output = {
            "Action_type": "information",
            "param": "page_info,current_page",
            "value": f"true,{current_page}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "page_information",
            "clicked": False,
            "element_name": "page info",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None,
            "page_information": page_info
        }
        
        logger.info(f"📋 Page information requested for '{current_page}'")
        logger.info(f"📋 Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="page_info",
            data=structured_output
        )
        
        return page_info
    
    async def _handle_update_table_name(self, user_id: str, new_table_name: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle updating the table name."""
        # Check if table creation is enabled on the current page
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("table_creation_enabled", False):
            return f"Table creation is not available on page '{current_page}'. Please navigate to the tables page first."
        
        # Check if user has an active table creation session
        if user_id not in self.table_creation_sessions:
            return "No active table creation session. Use 'create_table' to start creating a new table."
        
        table_creation_session = self.table_creation_sessions[user_id]
        old_table_name = table_creation_session["table_name"]
        
        # Update the table name
        table_creation_session["table_name"] = new_table_name.strip()
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "update_table_name",
            "old_name": old_table_name,
            "new_name": new_table_name,
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })
        
        # Create unified structured output for table name update
        structured_output = {
            "Action_type": "form_update",  # Updating is form updating
            "param": "form_update,table_name,old_name,new_name",
            "value": f"true,table_name,{old_table_name},{new_table_name}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "table_creation",
            "clicked": False,  # Not clicked, it's form updating
            "element_name": "update_table_name",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None,
            # Include complete table creation session info for frontend
            "table_creation_session": {
                "table_name": table_creation_session["table_name"],
                "schema": table_creation_session["schema"],
                "columns": table_creation_session["columns"],
                "current_step": "updating_table_name",
                "is_completed": False
            }
        }
        
        logger.info(f"✅ Table name updated from '{old_table_name}' to '{new_table_name}'")
        logger.info(f"✅ Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="update_table_name",
            data=structured_output
        )
        
        return f"✅ Table name updated from '{old_table_name}' to '{new_table_name}'"
    
    async def _handle_update_column_name(self, user_id: str, old_column_name: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle updating a column name."""
        # Check if table creation is enabled on the current page
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("table_creation_enabled", False):
            return f"Table creation is not available on page '{current_page}'. Please navigate to the tables page first."
        
        # Check if user has an active table creation session
        if user_id not in self.table_creation_sessions:
            return "No active table creation session. Use 'create_table' to start creating a new table."
        
        table_creation_session = self.table_creation_sessions[user_id]
        
        # Parse context for new column name
        new_column_name = None
        if context:
            # Look for new column name in context
            if "new_name:" in context:
                new_column_name = context.split("new_name:")[1].split(",")[0].strip()
            elif "name:" in context:
                new_column_name = context.split("name:")[1].split(",")[0].strip()
        
        if not new_column_name:
            return f"Please provide the new column name. Use context like 'new_name:new_column_name' or 'name:new_column_name'"
        
        # Find the column to update (use fuzzy matching)
        column_to_update = None
        best_match = None
        best_score = 0
        
        for col in table_creation_session["columns"]:
            # Exact match
            if col["name"].lower() == old_column_name.lower():
                column_to_update = col
                break
            # Partial match
            elif old_column_name.lower() in col["name"].lower() or col["name"].lower() in old_column_name.lower():
                score = len(set(old_column_name.lower()) & set(col["name"].lower()))
                if score > best_score:
                    best_score = score
                    best_match = col
        
        # Use best match if no exact match found
        if not column_to_update and best_match:
            column_to_update = best_match
            logger.info(f"🔍 Using fuzzy match: '{old_column_name}' → '{best_match['name']}'")
        
        if not column_to_update:
            available_columns = [col["name"] for col in table_creation_session["columns"]]
            return f"Column '{old_column_name}' not found. Available columns: {', '.join(available_columns)}"
        
        # Update the column name
        old_name = column_to_update["name"]
        column_to_update["name"] = new_column_name.strip()
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "update_column_name",
            "old_name": old_name,
            "new_name": new_column_name,
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })
        
        # Create unified structured output for column name update
        structured_output = {
            "Action_type": "form_update",  # Updating is form updating
            "param": "form_update,column_name,old_name,new_name",
            "value": f"true,column_name,{old_name},{new_column_name}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "table_creation",
            "clicked": False,  # Not clicked, it's form updating
            "element_name": "update_column_name",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None,
            # Include complete table creation session info for frontend
            "table_creation_session": {
                "table_name": table_creation_session["table_name"],
                "schema": table_creation_session["schema"],
                "columns": table_creation_session["columns"],
                "current_step": "updating_column_name",
                "is_completed": False
            }
        }
        
        logger.info(f"✅ Column name updated from '{old_name}' to '{new_column_name}'")
        logger.info(f"✅ Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="update_column_name",
            data=structured_output
        )
        
        return f"✅ Column name updated from '{old_name}' to '{new_column_name}'"
    
    async def _handle_update_column_properties(self, user_id: str, column_name: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle updating column properties."""
        # Check if table creation is enabled on the current page
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("table_creation_enabled", False):
            return f"Table creation is not available on page '{current_page}'. Please navigate to the tables page first."
        
        # Check if user has an active table creation session
        if user_id not in self.table_creation_sessions:
            return "No active table creation session. Use 'create_table' to start creating a new table."
        
        table_creation_session = self.table_creation_sessions[user_id]
        
        # Find the column to update (use fuzzy matching)
        column_to_update = None
        best_match = None
        best_score = 0
        
        for col in table_creation_session["columns"]:
            # Exact match
            if col["name"].lower() == column_name.lower():
                column_to_update = col
                break
            # Partial match
            elif column_name.lower() in col["name"].lower() or col["name"].lower() in column_name.lower():
                score = len(set(column_name.lower()) & set(col["name"].lower()))
                if score > best_score:
                    best_score = score
                    best_match = col
        
        # Use best match if no exact match found
        if not column_to_update and best_match:
            column_to_update = best_match
            logger.info(f"🔍 Using fuzzy match: '{column_name}' → '{best_match['name']}'")
        
        if not column_to_update:
            available_columns = [col["name"] for col in table_creation_session["columns"]]
            return f"Column '{column_name}' not found. Available columns: {', '.join(available_columns)}"
        
        # Parse context for properties to update
        properties_to_update = {}
        if context:
            context_parts = context.split(',')
            for part in context_parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    properties_to_update[key] = value
        
        if not properties_to_update:
            return f"Please provide properties to update. Use context like 'data_type:VARCHAR(100),nullable:false'"
        
        # Apply property updates
        updated_properties = []
        for key, value in properties_to_update.items():
            if key == "data_type":
                # Validate data type against available types
                available_types = await self._get_available_data_types()
                if value in available_types:
                    column_to_update["data_type"] = value
                    updated_properties.append(f"data_type: {value}")
                else:
                    # Try to find closest match
                    closest_match = self._find_closest_data_type(value, available_types)
                    if closest_match:
                        column_to_update["data_type"] = closest_match
                        column_to_update["comment"] = f"Auto-corrected from '{value}' to '{closest_match}'"
                        updated_properties.append(f"data_type: {closest_match} (auto-corrected)")
                        logger.info(f"🔍 Auto-corrected data type '{value}' to '{closest_match}'")
                    else:
                        return f"Invalid data type '{value}'. Available types: {', '.join(available_types[:10])}..."
            
            elif key == "nullable":
                column_to_update["nullable"] = value.lower() == "true"
                updated_properties.append(f"nullable: {value}")
            
            elif key == "is_primary_key":
                column_to_update["is_primary_key"] = value.lower() == "true"
                updated_properties.append(f"primary_key: {value}")
            
            elif key == "is_identity":
                column_to_update["is_identity"] = value.lower() == "true"
                updated_properties.append(f"identity: {value}")
            
            elif key == "comment":
                column_to_update["comment"] = value
                updated_properties.append(f"comment: {value}")
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "update_column_properties",
            "column_name": column_to_update["name"],
            "properties": properties_to_update,
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })
        
        # Create unified structured output for column properties update
        structured_output = {
            "Action_type": "form_update",  # Updating is form updating
            "param": "form_update,column_properties,column_name,properties",
            "value": f"true,column_properties,{column_to_update['name']},{','.join(updated_properties)}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "table_creation",
            "clicked": False,  # Not clicked, it's form updating
            "element_name": "update_column_properties",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None,
            # Include complete table creation session info for frontend
            "table_creation_session": {
                "table_name": table_creation_session["table_name"],
                "schema": table_creation_session["schema"],
                "columns": table_creation_session["columns"],
                "current_step": "updating_properties",
                "is_completed": False
            }
        }
        
        logger.info(f"✅ Column properties updated for '{column_to_update['name']}': {', '.join(updated_properties)}")
        logger.info(f"✅ Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="update_column_properties",
            data=structured_output
        )
        
        return f"✅ Column properties updated for '{column_to_update['name']}': {', '.join(updated_properties)}"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_user_session(self, user_id: str) -> Dict[str, Any]:
        """Get user session information."""
        return self.user_sessions.get(user_id, {
            "current_page": self.initial_current_page,
            "previous_page": None,
            "interaction_history": []
        })
    
    def get_table_creation_session(self, user_id: str) -> Dict[str, Any]:
        """Get table creation session information for a user."""
        return self.table_creation_sessions.get(user_id, {
            "table_name": None,
            "columns": [],
            "current_step": None,
            "is_completed": False
        })
    
    def get_current_page(self, user_id: str) -> str:
        """Get current page for a user."""
        return self.get_user_session(user_id).get("current_page", self.initial_current_page)
    
    def get_available_pages(self) -> list:
        """Get list of available pages."""
        return list(self.VALID_PAGES.keys())
    
    def get_page_elements(self, page: str) -> Dict[str, list]:
        """Get available elements for a page."""
        return self.VALID_PAGES.get(page, {"buttons": [], "forms": []})
    
    def get_page_information(self, page: str) -> str:
        """Get user-friendly information about what can be done on a specific page."""
        if page not in self.VALID_PAGES:
            return f"Unknown page: {page}"
        
        page_config = self.VALID_PAGES[page]
        buttons = page_config.get("buttons", [])
        features = []
        
        # Add button information
        if buttons:
            features.append(f"Available buttons: {', '.join(buttons)}")
        
        # Add special features
        if page_config.get("search_enabled"):
            features.append("Database search functionality")
        if page_config.get("file_search_enabled"):
            features.append("File search functionality")
        if page_config.get("file_upload_enabled"):
            features.append("File upload functionality")
        if page_config.get("table_creation_enabled"):
            features.append("Table creation functionality")
        
        if features:
            return f"On the {page} page, you can:\n" + "\n".join(f"• {feature}" for feature in features)
        else:
            return f"The {page} page provides navigation to other pages in the application."
    
    def _auto_detect_data_type(self, column_name: str) -> str:
        """Auto-detect data type based on column name patterns."""
        column_name_lower = column_name.lower()
        
        # ID and key columns
        if any(keyword in column_name_lower for keyword in ['id', 'key', 'pk', 'primary']):
            return "INT"
        
        # Date and time columns
        if any(keyword in column_name_lower for keyword in ['date', 'time', 'created', 'updated', 'modified', 'timestamp']):
            return "DATETIME"
        
        # Numeric columns - be more conservative with decimal vs integer
        if any(keyword in column_name_lower for keyword in ['count', 'number', 'quantity']):
            return "INT"  # Counts and quantities are usually integers
        
        # Money/currency columns - these are typically decimal
        if any(keyword in column_name_lower for keyword in ['amount', 'price', 'cost', 'salary', 'wage', 'income', 'revenue', 'profit', 'loss']):
            return "DECIMAL(18,2)"  # Money values need decimal precision
        
        # Age and rating columns - these could be either
        if any(keyword in column_name_lower for keyword in ['age', 'score', 'rating', 'level', 'grade']):
            return "INT"  # Default to integer for these
        
        # Boolean columns
        if any(keyword in column_name_lower for keyword in ['is_', 'has_', 'active', 'enabled', 'valid', 'approved']):
            return "BIT"
        
        # Text columns (default)
        return "VARCHAR(255)"
    
    async def _get_available_data_types(self) -> List[str]:
        """Fetch available data types from the endpoint."""
        try:
            # Use direct function import if available (avoids SSL issues)
            if get_supported_sql_server_types:
                data_types_dict = get_supported_sql_server_types()
                data_types = []
                
                # Flatten the categorized data types
                for category, types in data_types_dict.items():
                    if isinstance(types, list):
                        data_types.extend(types)
                    elif isinstance(types, str):
                        data_types.append(types)
                
                logger.info(f"✅ Fetched {len(data_types)} data types from direct function")
                return data_types
            
            # Fallback to HTTP request if direct import fails
            import os
            backend_url = os.getenv("DEV_BACKEND_URL", "http://localhost:8200")  # Use HTTP for local dev
            
            # Make HTTP request to get data types
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{backend_url}/new-table/data-types") as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "success":
                            # Flatten the categorized data types
                            all_types = []
                            for category_types in data.get("data", {}).values():
                                if isinstance(category_types, list):
                                    all_types.extend(category_types)
                            return all_types
            
            # Fallback to common types if endpoint fails
            return ["INT", "VARCHAR(255)", "DATETIME", "DECIMAL(18,2)", "BIT", "NVARCHAR(255)", "DATE", "TIME"]
            
        except Exception as e:
            logger.error(f"❌ Error fetching data types: {e}")
            # Fallback to common types
            return ["INT", "VARCHAR(255)", "DATETIME", "DECIMAL(18,2)", "BIT", "NVARCHAR(255)", "DATE", "TIME"]
    
    def _find_closest_data_type(self, input_type: str, available_types: List[str]) -> Optional[str]:
        """Find the closest matching data type from available types."""
        input_type_upper = input_type.upper()
        
        # Exact match
        if input_type_upper in available_types:
            return input_type_upper
        
        # Partial matches (more flexible)
        for available_type in available_types:
            # Check if input is contained in available type or vice versa
            if input_type_upper in available_type or available_type in input_type_upper:
                return available_type
        
        # Enhanced fuzzy matching for common variations
        type_mappings = {
            "INTEGER": "INT",
            "STRING": "VARCHAR(255)",
            "TEXT": "VARCHAR(255)",
            "NUMBER": "DECIMAL(18,2)",
            "DECIMAL": "DECIMAL(18,2)",
            "BOOL": "BIT",
            "BOOLEAN": "BIT",
            "DATETIME": "DATETIME",
            "TIMESTAMP": "DATETIME",
            "VARCHAR": "VARCHAR(255)",
            "CHAR": "CHAR(10)",
            "NVARCHAR": "NVARCHAR(50)",
            "NCHAR": "NCHAR(10)",
            "FLOAT": "FLOAT",
            "DOUBLE": "FLOAT",
            "REAL": "REAL",
            "MONEY": "MONEY",
            "SMALLMONEY": "SMALLMONEY",
            "BIGINT": "BIGINT",
            "SMALLINT": "SMALLINT",
            "TINYINT": "TINYINT",
            "UNIQUEIDENTIFIER": "UNIQUEIDENTIFIER",
            "GUID": "UNIQUEIDENTIFIER",
            "XML": "XML",
            "SQL_VARIANT": "SQL_VARIANT"
        }
        
        if input_type_upper in type_mappings:
            mapped_type = type_mappings[input_type_upper]
            if mapped_type in available_types:
                return mapped_type
        
        # Try to find closest match by character similarity
        best_match = None
        best_score = 0
        
        for available_type in available_types:
            # Calculate similarity score
            score = self._calculate_similarity(input_type_upper, available_type)
            if score > best_score and score > 0.6:  # Minimum similarity threshold
                best_score = score
                best_match = available_type
        
        return best_match
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using character overlap."""
        if not str1 or not str2:
            return 0.0
        
        # Convert to sets of characters
        set1 = set(str1.upper())
        set2 = set(str2.upper())
        
        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def get_table_creation_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a summary of the current table creation session."""
        session = self.table_creation_sessions.get(user_id)
        if not session:
            return {"error": "No active table creation session"}
        
        return {
            "table_name": session["table_name"],
            "schema": "dbo",  # Default schema
            "columns_count": len(session["columns"]),
            "columns": session["columns"],
            "current_step": session["current_step"],
            "is_completed": session["is_completed"]
        }
    
    def get_table_creation_status_message(self, user_id: str) -> str:
        """Get a user-friendly status message for the current table creation session."""
        session = self.table_creation_sessions.get(user_id)
        if not session:
            return "No active table creation session. Use 'create_table' to start creating a new table."
        
        table_name = session["table_name"]
        columns_count = len(session["columns"])
        
        if columns_count == 0:
            return f"Creating table '{table_name}'. Please add columns using 'add_column' action."
        elif columns_count == 1:
            return f"Table '{table_name}' has {columns_count} column. You can add more columns or set properties for existing columns."
        else:
            return f"Table '{table_name}' has {columns_count} columns. You can add more columns, set properties, or complete the table creation."
    
    async def complete_table_creation(self, user_id: str, confirm: bool = False) -> str:
        """Complete the table creation by preparing for confirmation."""
        session = self.table_creation_sessions.get(user_id)
        if not session:
            return "No active table creation session to complete."
        
        if not session["columns"]:
            return "Cannot create table without columns. Please add at least one column first."
        
        # If not confirmed, return confirmation request with table summary
        if not confirm:
            table_summary = self._format_table_summary_for_confirmation(session)
            return f"📋 Table Creation Confirmation Required\n\n{table_summary}\n\nPlease confirm by saying 'confirm' or 'yes' to proceed with table creation."
        
        # If confirmed, mark as ready for submission
        session["is_completed"] = True
        session["current_step"] = "ready_for_submission"
        
        # Send confirmation message via WebSocket
        await self.send_websocket_message(
            message_type="table_creation_confirmed",
            action="confirmed",
            data={
                "table_name": session["table_name"],
                "schema": "dbo",
                "columns_count": len(session["columns"]),
                "columns": session["columns"],
                "timestamp": self._get_timestamp(),
                "user_id": user_id
            }
        )
        
        return f"✅ Table '{session['table_name']}' confirmed and ready for submission with {len(session['columns'])} columns!"
    
    async def _handle_submit_table(self, user_id: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle submitting table creation without API calls."""
        # Check if table creation is enabled on the current page
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("table_creation_enabled", False):
            return f"Table creation is not available on page '{current_page}'. Please navigate to the tables page first."
        
        # Check if user has an active table creation session
        if user_id not in self.table_creation_sessions:
            return "No active table creation session. Use 'create_table' to start creating a new table."
        
        table_creation_session = self.table_creation_sessions[user_id]
        
        if not table_creation_session["columns"]:
            return "Cannot submit table without columns. Please add at least one column first."
        
        # Mark session as submitted (no API calls)
        table_creation_session["is_completed"] = True
        table_creation_session["current_step"] = "submitted"
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "submit_table",
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })
        
        # Create unified structured output for table submission
        structured_output = {
            "Action_type": "clicked",  # Submission is a click action
            "param": "clicked,name,action",
            "value": f"true,submit table creation,{action_type}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "table_submission",
            "clicked": True,  # Submission is clicked
            "element_name": "submit table creation",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None,
            # Include complete table creation session info for frontend
            "table_creation_session": {
                "table_name": table_creation_session["table_name"],
                "columns": table_creation_session["columns"],
                "current_step": "submitted",
                "is_completed": True
            }
        }
        
        logger.info(f"✅ Table creation submitted on page '{current_page}'")
        logger.info(f"✅ Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="submit_table",
            data=structured_output
        )
        
        # Return complete table summary
        table_summary = self._format_table_summary_for_confirmation(table_creation_session)
        return f"✅ Table creation submitted successfully!\n\n{table_summary}\n\nTable '{table_creation_session['table_name']}' has been finalized and is ready for use."
    
    async def _handle_general_update(self, user_id: str, target: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle general update actions."""
        # Check if table creation is enabled on the current page
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("table_creation_enabled", False):
            return f"Table creation is not available on page '{current_page}'. Please navigate to the tables page first."
        
        # Check if user has an active table creation session
        if user_id not in self.table_creation_sessions:
            return "No active table creation session. Use 'create_table' to start creating a new table."
        
        table_creation_session = self.table_creation_sessions[user_id]
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "general_update",
            "target": target,
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })
        
        # Create unified structured output for general update
        structured_output = {
            "Action_type": "form_update",  # General update action
            "param": "form_update,general,target,context",
            "value": f"true,general_update,{target},{context or 'none'}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "table_creation",
            "clicked": False,  # Not clicked, it's form updating
            "element_name": "general_update",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None,
            # Include complete table creation session info for frontend
            "table_creation_session": {
                "table_name": table_creation_session["table_name"],
                "schema": table_creation_session["schema"],
                "columns": table_creation_session["columns"],
                "current_step": "general_update",
                "is_completed": False
            }
        }
        
        logger.info(f"✅ General update performed on page '{current_page}'")
        logger.info(f"✅ Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="general_update",
            data=structured_output
        )
        
        return f"✅ Update action performed. Current table '{table_creation_session['table_name']}' has {len(table_creation_session['columns'])} columns."
    
    def reset_table_creation_session(self, user_id: str) -> str:
        """Reset the table creation session for a user."""
        if user_id in self.table_creation_sessions:
            del self.table_creation_sessions[user_id]
            return f"Table creation session reset for user {user_id}"
        else:
            return f"No active table creation session found for user {user_id}"
    
    def _format_table_summary_for_confirmation(self, session: Dict[str, Any]) -> str:
        """Format table summary for confirmation display."""
        summary = f"📋 Table: {session['table_name']}\n"
        summary += f"🏗️ Schema: dbo\n"
        summary += f"📊 Columns ({len(session['columns'])}):\n"
        
        for i, col in enumerate(session['columns'], 1):
            summary += f"  {i}. {col['name']} ({col['data_type']})\n"
            summary += f"     - Nullable: {'Yes' if col['nullable'] else 'No'}\n"
            summary += f"     - Primary Key: {'Yes' if col['is_primary_key'] else 'No'}\n"
            summary += f"     - Identity: {'Yes' if col['is_identity'] else 'No'}\n"
            if col['comment']:
                summary += f"     - Comment: {col['comment']}\n"
        
        # Add action instructions
        summary += f"\n💡 Actions Available:\n"
        summary += f"  • Say 'confirm' or 'yes' to prepare for submission\n"
        summary += f"  • Say 'submit' to finalize table creation\n"
        summary += f"  • Say 'cancel' to abort table creation\n"
        
        return summary
    
    async def _handle_table_management(self, user_id: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle table management operations."""
        # Check if table management is available on the current page
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("table_creation_enabled", False):
            return f"Table management is not available on page '{current_page}'. Please navigate to the tables page first."
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "table_management",
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })
        
        # Create unified structured output for table management
        structured_output = {
            "Action_type": "clicked",  # Table management is a click action
            "param": "clicked,name,action",
            "value": f"true,table management,{action_type}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "table_management",
            "clicked": True,  # Button clicked
            "element_name": "table management",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None,
            # Include table creation session info if available
            "table_creation_session": None
        }
        
        # Add table creation session if user has one
        if user_id in self.table_creation_sessions:
            table_creation_session = self.table_creation_sessions[user_id]
            structured_output["table_creation_session"] = {
                "table_name": table_creation_session["table_name"],
                "schema": table_creation_session["schema"],
                "columns": table_creation_session["columns"],
                "current_step": "table_management",
                "is_completed": table_creation_session["is_completed"]
            }
        
        logger.info(f"✅ Table management accessed on page '{current_page}'")
        logger.info(f"✅ Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="table_management",
            data=structured_output
        )
        
        return f"✅ Table management accessed. You can now manage table operations, view table structures, and perform administrative tasks."
    
    async def _handle_business_rule_management(self, user_id: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle business rule management operations."""
        # Check if business rule management is available on the current page
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("table_creation_enabled", False):
            return f"Business rule management is not available on page '{current_page}'. Please navigate to the tables page first."
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "business_rule_management",
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })
        
        # Create unified structured output for business rule management
        structured_output = {
            "Action_type": "clicked",  # Business rule management is a click action
            "param": "clicked,name,action",
            "value": f"true,manage business rule,{action_type}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "business_rule_management",
            "clicked": True,  # Button clicked
            "element_name": "manage business rule",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None,
            # Include table creation session info if available
            "table_creation_session": None
        }
        
        # Add table creation session if user has one
        if user_id in self.table_creation_sessions:
            table_creation_session = self.table_creation_sessions[user_id]
            structured_output["table_creation_session"] = {
                "table_name": table_creation_session["table_name"],
                "schema": table_creation_session["schema"],
                "columns": table_creation_session["columns"],
                "current_step": "business_rule_management",
                "is_completed": table_creation_session["is_completed"]
            }
        
        logger.info(f"✅ Business rule management accessed on page '{current_page}'")
        logger.info(f"✅ Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="business_rule_management",
            data=structured_output
        )
        
        return f"✅ Business rule management accessed. You can now configure business rules, validation rules, and data integrity constraints for your tables."
    
    async def _handle_excel_import(self, user_id: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle Excel import initiation."""
        # Check if Excel import is available on the current page
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("excel_import_enabled", False):
            return f"Excel import is not available on page '{current_page}'. Please navigate to the tables page first."
        
        # Initialize Excel import session
        self.excel_import_sessions[user_id] = {
            "current_step": "excel_import_initiated",
            "file_uploaded": False,
            "ai_suggestion_received": False,
            "mapping_completed": False,
            "import_confirmed": False,
            "selected_table": None,  # Added selected_table field
            "timestamp": self._get_timestamp()
        }
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "excel_import",
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })
        
        # Create unified structured output for Excel import
        structured_output = {
            "Action_type": "clicked",  # Excel import is a click action
            "param": "clicked,name,action",
            "value": f"true,excel-import,{action_type}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "excel_import",
            "clicked": True,  # Button clicked
            "element_name": "excel-import",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None,
            # Include Excel import session info
            "excel_import_session": self.excel_import_sessions[user_id]
        }
        
        logger.info(f"✅ Excel import initiated on page '{current_page}'")
        logger.info(f"✅ Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="excel_import",
            data=structured_output
        )
        
        return f"✅ Excel import initiated. You can now upload an Excel file and get AI suggestions for table structure."
    
    async def _handle_select_table(self, user_id: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle table selection for Excel import."""
        # Check if Excel import is available on the current page
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("excel_import_enabled", False):
            return f"Table selection is not available on page '{current_page}'. Please navigate to the tables page first."
        
        # Check if user has an active Excel import session
        if user_id not in self.excel_import_sessions:
            return "No active Excel import session. Use 'excel_import' to start the import process first."
        
        excel_import_session = self.excel_import_sessions[user_id]
        
        # Check if context contains table name
        if not context:
            return "Please specify which table you want to select for Excel import. For example: 'select table testriaz' or 'I want to use the test table'."
        
        try:
            # Fetch available tables for the user
            tables_data = await self._fetch_user_tables(user_id)
            
            if not tables_data or "tables" not in tables_data:
                return "❌ Failed to fetch available tables. Please try again."
            
            tables = tables_data["tables"]
            if not tables:
                return "❌ No tables found for your user account. Please create a table first."
            
            # Extract table names in format schema_name.table_name
            table_names = [f"{table['schema_name']}.{table['table_name']}" for table in tables]
            
            # Find the best match for the user's input
            selected_table = self._find_best_table_match(context, table_names, tables)
            
            if not selected_table:
                return f"❌ Could not find a table matching '{context}'. Available tables: {', '.join(table_names)}"
            
            # Update session with selected table
            excel_import_session["selected_table"] = selected_table
            excel_import_session["current_step"] = "table_selected"
            excel_import_session["timestamp"] = self._get_timestamp()
            
            # Update user session
            user_session = self.user_sessions[user_id]
            user_session["interaction_history"].append({
                "action": "select_table",
                "page": current_page,
                "context": context,
                "selected_table": selected_table,
                "timestamp": self._get_timestamp()
            })
            
            # Create unified structured output for table selection
            structured_output = {
                "Action_type": "form_update",  # Table selection is a form update
                "param": "clicked,name,action",
                "value": f"true,select_table,{action_type}",
                "page": current_page,
                "previous_page": user_session.get("previous_page"),
                "interaction_type": "table_selection",
                "clicked": True,
                "element_name": "select_table",
                "search_query": None,
                "report_request": None,
                "report_query": None,
                "upload_request": None,
                "db_id": None,
                "table_specific": False,
                "tables": [],
                "file_descriptions": [],
                "table_names": table_names,  # Include available table names
                "context": context,
                "timestamp": self._get_timestamp(),
                "user_id": user_id,
                "success": True,
                "error_message": None,
                # Include Excel import session info
                "excel_import_session": excel_import_session
            }
            
            logger.info(f"✅ Table '{selected_table}' selected for Excel import")
            logger.info(f"✅ Structured output: {json.dumps(structured_output, indent=2)}")
            
            # Send via WebSocket
            await self.send_websocket_message(
                message_type="navigation_result",
                action="table_selection",
                data=structured_output
            )
            
            return f"✅ Table '{selected_table}' selected successfully! You can now say 'Get AI Mapping Suggestions' to proceed."
            
        except Exception as e:
            logger.error(f"❌ Error in table selection: {e}")
            return f"❌ Error selecting table: {str(e)}"
    
    async def _handle_list_available_tables(self, user_id: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle listing available tables for Excel import."""
        # Check if Excel import is available on the current page
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("excel_import_enabled", False):
            return f"Table listing is not available on page '{current_page}'. Please navigate to the tables page first."
        
        # Check if user has an active Excel import session
        if user_id not in self.excel_import_sessions:
            return "No active Excel import session. Use 'excel_import' to start the import process first."
        
        try:
            # Fetch available tables for the user
            tables_data = await self._fetch_user_tables(user_id)
            
            if not tables_data or "tables" not in tables_data:
                return "❌ Failed to fetch available tables. Please try again."
            
            tables = tables_data["tables"]
            if not tables:
                return "❌ No tables found for your user account. Please create a table first."
            
            # Extract table names in format schema_name.table_name
            table_names = [f"{table['schema_name']}.{table['table_name']}" for table in tables]
            
            # Create a user-friendly table list
            table_list = []
            for i, table in enumerate(tables, 1):
                table_name = f"{table['schema_name']}.{table['table_name']}"
                column_count = len(table.get('table_schema', {}).get('columns', []))
                table_list.append(f"{i}. **{table_name}** ({column_count} columns)")
            
            # Update user session
            user_session = self.user_sessions[user_id]
            user_session["interaction_history"].append({
                "action": "list_available_tables",
                "page": current_page,
                "context": context,
                "timestamp": self._get_timestamp()
            })
            
            # Create unified structured output for table listing
            structured_output = {
                "Action_type": "information",  # Table listing is informational
                "param": "clicked,name,action",
                "value": f"true,list_available_tables,{action_type}",
                "page": current_page,
                "previous_page": user_session.get("previous_page"),
                "interaction_type": "table_listing",
                "clicked": True,
                "element_name": "list_available_tables",
                "search_query": None,
                "report_request": None,
                "report_query": None,
                "upload_request": None,
                "db_id": None,
                "table_specific": False,
                "tables": [],
                "file_descriptions": [],
                "table_names": table_names,  # Include available table names
                "context": context,
                "timestamp": self._get_timestamp(),
                "user_id": user_id,
                "success": True,
                "error_message": None,
                # Include Excel import session info
                "excel_import_session": self.excel_import_sessions[user_id]
            }
            
            logger.info(f"✅ Available tables listed for user {user_id}: {len(tables)} tables")
            logger.info(f"✅ Structured output: {json.dumps(structured_output, indent=2)}")
            
            # Send via WebSocket
            await self.send_websocket_message(
                message_type="navigation_result",
                action="table_listing",
                data=structured_output
            )
            
            # Return user-friendly message
            table_count = len(tables)
            selection_guide = f"""
💡 **To select a table, you can say:**

**By name:**
• "select table testriaz"
• "select table test"

**By position:**
• "select the first table"
• "select the last table" 
• "select table number 2"
• "select the 3rd table"

**Examples:**
• "I want to use the first table"
• "Select the last table for import"
• "Use table number 2"
"""
            
            return f"📋 **Available Tables for Excel Import** ({table_count} tables):\n\n" + "\n".join(table_list) + selection_guide
            
        except Exception as e:
            logger.error(f"❌ Error listing available tables: {e}")
            return f"❌ Error listing available tables: {str(e)}"
    
    async def _fetch_user_tables(self, user_id: str) -> Dict[str, Any]:
        """Fetch available tables for a user from the API."""
        try:
            # First try direct function import to avoid HTTP connection issues
            if get_user_created_tables:
                try:
                    # Call the function directly
                    result = await get_user_created_tables(user_id)
                    if result and hasattr(result, 'data'):
                        return result.data
                    elif result and isinstance(result, dict):
                        return result
                    else:
                        logger.warning("Direct function call returned unexpected format, falling back to HTTP")
                except Exception as direct_error:
                    logger.warning(f"Direct function call failed: {direct_error}, falling back to HTTP")
            
            # Fallback to HTTP request if direct import fails
            url = f"http://localhost:8200/new-table/user-tables/{user_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "success":
                            return data.get("data", {})
                        else:
                            logger.error(f"❌ API returned error: {data.get('message', 'Unknown error')}")
                            return None
                    else:
                        logger.error(f"❌ HTTP {response.status}: {await response.text()}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Error fetching user tables: {e}")
            return None
    
    def _find_best_table_match(self, user_input: str, table_names: List[str], tables: List[Dict[str, Any]]) -> Optional[str]:
        """Find the best matching table name from user input."""
        if not user_input or not table_names:
            return None
        
        # Convert user input to lowercase for comparison
        user_input_lower = user_input.lower().strip()
        
        # First, try position-based selection (first, last, number)
        position_match = self._find_position_based_match(user_input_lower, table_names)
        if position_match:
            return position_match
        
        # Second, try exact matches
        for table_name in table_names:
            if table_name.lower() == user_input_lower:
                return table_name
        
        # Third, try partial matches (table name only, without schema)
        for table_name in table_names:
            table_name_only = table_name.split('.')[-1].lower()
            if table_name_only == user_input_lower:
                return table_name
        
        # Fourth, try fuzzy matching for table names
        for table_name in table_names:
            table_name_only = table_name.split('.')[-1].lower()
            
            # Check if user input contains the table name
            if table_name_only in user_input_lower:
                return table_name
            
            # Check if table name contains user input
            if user_input_lower in table_name_only:
                return table_name
        
        # Fifth, try fuzzy matching with schema names
        for table_name in table_names:
            if user_input_lower in table_name.lower():
                return table_name
        
        return None
    
    def _find_position_based_match(self, user_input: str, table_names: List[str]) -> Optional[str]:
        """Find table based on position references like 'first', 'last', 'number 2'."""
        if not user_input or not table_names:
            return None
        
        # Check for "first" table
        if any(word in user_input for word in ["first", "1st", "one", "1"]):
            return table_names[0]
        
        # Check for "last" table
        if any(word in user_input for word in ["last", "final", "end"]):
            return table_names[-1]
        
        # Check for specific number references (e.g., "number 2", "table 3", "2nd")
        import re
        
        # Look for number patterns
        number_patterns = [
            r"number\s+(\d+)",      # "number 2"
            r"table\s+(\d+)",       # "table 3"
            r"(\d+)(?:st|nd|rd|th)", # "2nd", "3rd", "4th"
            r"(\d+)",               # just "2", "3"
        ]
        
        for pattern in number_patterns:
            match = re.search(pattern, user_input)
            if match:
                try:
                    number = int(match.group(1))
                    # Convert to 0-based index
                    index = number - 1
                    if 0 <= index < len(table_names):
                        return table_names[index]
                except (ValueError, IndexError):
                    continue
        
        return None
    
    async def _handle_get_ai_suggestion(self, user_id: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle getting AI suggestions for Excel import."""
        # Check if Excel import is available on the current page
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("excel_import_enabled", False):
            return f"AI suggestions are not available on page '{current_page}'. Please navigate to the tables page first."
        
        # Check if user has an active Excel import session
        if user_id not in self.excel_import_sessions:
            return "No active Excel import session. Use 'excel_import' to start the import process first."
        
        excel_import_session = self.excel_import_sessions[user_id]
        
        # Check if table is selected
        if not excel_import_session.get("selected_table"):
            return "❌ Please select a target table first before getting AI suggestions. Say 'select table [table_name]' to choose a table."
        
        # Update session state
        excel_import_session["ai_suggestion_received"] = True
        excel_import_session["current_step"] = "ai_suggestion_received"
        excel_import_session["timestamp"] = self._get_timestamp()
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "get_ai_suggestion",
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })
        
        # Create unified structured output for AI suggestion
        structured_output = {
            "Action_type": "clicked",  # AI suggestion is a click action
            "param": "clicked,name,action",
            "value": f"true,get ai suggestion,{action_type}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "ai_suggestion",
            "clicked": True,  # Button clicked
            "element_name": "get ai suggestion",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None,
            # Include Excel import session info
            "excel_import_session": excel_import_session
        }
        
        logger.info(f"✅ AI suggestion requested on page '{current_page}'")
        logger.info(f"✅ Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="ai_suggestion",
            data=structured_output
        )
        
        selected_table = excel_import_session.get("selected_table", "Unknown")
        return f"✅ AI suggestion received for table '{selected_table}'. You can now say 'continue to import' to proceed with the import process."
    
    async def _handle_continue_to_import(self, user_id: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle continuing to import after AI suggestion."""
        # Check if Excel import is available on the current page
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("excel_import_enabled", False):
            return f"Continue to import is not available on page '{current_page}'. Please navigate to the tables page first."
        
        # Check if user has an active Excel import session
        if user_id not in self.excel_import_sessions:
            return "No active Excel import session. Use 'excel_import' to start the import process first."
        
        excel_import_session = self.excel_import_sessions[user_id]
        
        # Check if AI suggestion was received
        if not excel_import_session.get("ai_suggestion_received", False):
            return "AI suggestion must be received before continuing to import. Use 'get ai suggestion' first."
        
        # Update session state
        excel_import_session["current_step"] = "continue_to_import_clicked"
        excel_import_session["timestamp"] = self._get_timestamp()
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "continue_to_import",
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })
        
        # Create unified structured output for continue to import
        structured_output = {
            "Action_type": "clicked",  # Continue to import is a click action
            "param": "clicked,name,action",
            "value": f"true,continue to import,{action_type}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "continue_to_import",
            "clicked": True,  # Button clicked
            "element_name": "continue to import",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None,
            # Include Excel import session info
            "excel_import_session": excel_import_session
        }
        
        logger.info(f"✅ Continue to import clicked on page '{current_page}'")
        logger.info(f"✅ Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="continue_to_import",
            data=structured_output
        )
        
        return f"✅ Continue to import clicked. You can now choose to go 'back to mapping' or 'import data to database'."
    
    async def _handle_back_to_mapping(self, user_id: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle going back to mapping step."""
        # Check if Excel import is available on the current page
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("excel_import_enabled", False):
            return f"Back to mapping is not available on page '{current_page}'. Please navigate to the tables page first."
        
        # Check if user has an active Excel import session
        if user_id not in self.excel_import_sessions:
            return "No active Excel import session. Use 'excel_import' to start the import process first."
        
        excel_import_session = self.excel_import_sessions[user_id]
        
        # Check if continue to import was clicked
        if excel_import_session.get("current_step") != "continue_to_import_clicked":
            return "Continue to import must be clicked before going back to mapping. Use 'continue to import' first."
        
        # Update session state
        excel_import_session["current_step"] = "back_to_mapping"
        excel_import_session["timestamp"] = self._get_timestamp()
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "back_to_mapping",
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })
        
        # Create unified structured output for back to mapping
        structured_output = {
            "Action_type": "clicked",  # Back to mapping is a click action
            "param": "clicked,name,action",
            "value": f"true,back to mapping,{action_type}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "back_to_mapping",
            "clicked": True,  # Button clicked
            "element_name": "back to mapping",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None,
            # Include Excel import session info
            "excel_import_session": excel_import_session
        }
        
        logger.info(f"✅ Back to mapping clicked on page '{current_page}'")
        logger.info(f"✅ Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="back_to_mapping",
            data=structured_output
        )
        
        return f"✅ Back to mapping clicked. You are now returned to the 'continue to import' step. You can choose to go 'back to mapping' again or 'import data to database'."
    
    async def _handle_import_data_to_database(self, user_id: str, current_page: str, action_type: str, context: str = None) -> str:
        """Handle final import data to database action."""
        # Check if Excel import is available on the current page
        if current_page not in self.VALID_PAGES:
            return f"Unknown current page: {current_page}"
        
        page_config = self.VALID_PAGES[current_page]
        if not page_config.get("excel_import_enabled", False):
            return f"Import data to database is not available on page '{current_page}'. Please navigate to the tables page first."
        
        # Check if user has an active Excel import session
        if user_id not in self.excel_import_sessions:
            return "No active Excel import session. Use 'excel_import' to start the import process first."
        
        excel_import_session = self.excel_import_sessions[user_id]
        
        # Check if continue to import was clicked
        if excel_import_session.get("current_step") != "continue_to_import_clicked":
            return "Continue to import must be clicked before importing data. Use 'continue to import' first."
        
        # Update session state
        excel_import_session["current_step"] = "import_data_to_database"
        excel_import_session["import_confirmed"] = True
        excel_import_session["timestamp"] = self._get_timestamp()
        
        # Update user session
        user_session = self.user_sessions[user_id]
        user_session["interaction_history"].append({
            "action": "import_data_to_database",
            "page": current_page,
            "context": context,
            "timestamp": self._get_timestamp()
        })
        
        # Create unified structured output for import data to database
        structured_output = {
            "Action_type": "clicked",  # Import data to database is a click action
            "param": "clicked,name,action",
            "value": f"true,import data to database,{action_type}",
            "page": current_page,
            "previous_page": user_session.get("previous_page"),
            "interaction_type": "import_data_to_database",
            "clicked": True,  # Button clicked
            "element_name": "import data to database",
            "search_query": None,
            "report_request": None,
            "report_query": None,
            "upload_request": None,
            "db_id": None,
            "table_specific": False,
            "tables": [],
            "file_descriptions": [],
            "table_names": [],
            "context": context,
            "timestamp": self._get_timestamp(),
            "user_id": user_id,
            "success": True,
            "error_message": None,
            # Include Excel import session info
            "excel_import_session": excel_import_session
        }
        
        logger.info(f"✅ Import data to database clicked on page '{current_page}'")
        logger.info(f"✅ Structured output: {json.dumps(structured_output, indent=2)}")
        
        # Send via WebSocket
        await self.send_websocket_message(
            message_type="navigation_result",
            action="import_data_to_database",
            data=structured_output
        )
        
        return f"✅ Import data to database clicked. Your Excel data is now being imported to the database. The import process has been completed successfully."
    
    def reset_excel_import_session(self, user_id: str) -> str:
        """Reset the Excel import session for a user."""
        if user_id in self.excel_import_sessions:
            del self.excel_import_sessions[user_id]
            return f"Excel import session reset for user {user_id}"
        else:
            return f"No active Excel import session found for user {user_id}"
    
    def get_excel_import_status_message(self, user_id: str) -> str:
        """Get a descriptive message based on Excel import session state."""
        if user_id not in self.excel_import_sessions:
            return "No active Excel import session."
        
        session = self.excel_import_sessions[user_id]
        current_step = session.get("current_step", "unknown")
        
        status_messages = {
            "excel_import_initiated": "Excel import process initiated. Please upload a file and get AI suggestions.",
            "ai_suggestion_received": "AI suggestions received. You can now continue to import.",
            "continue_to_import_clicked": "Continue to import clicked. Choose to go back to mapping or import data to database.",
            "back_to_mapping": "Returned to mapping step. You can choose to go back to mapping again or import data to database.",
            "import_data_to_database": "Import data to database clicked. Import process completed."
        }
        
        return status_messages.get(current_step, f"Unknown step: {current_step}")


# Factory function to create the tool
def create_navigation_tool(rtvi_processor: RTVIProcessor, task=None, initial_current_page: str = "dashboard") -> NavigationTool:
    """Create and return a NavigationTool instance."""
    return NavigationTool(rtvi_processor, task, initial_current_page)
