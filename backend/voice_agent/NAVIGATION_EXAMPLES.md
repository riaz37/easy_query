# Navigation Tool Examples

## Overview
The navigation tool handles context-aware page navigation, element interactions, and database searches. It maintains user session state and can distinguish between navigating to a new page, interacting with elements, or performing database searches when on the database-query page.

## Example Scenarios

### Scenario 1: Navigate to User Configuration Page

**User Input**: "I want to go to user-configuration"

**Tool Call**:
```json
{
  "function_name": "navigate_page",
  "arguments": {
    "user_id": "current_user",
    "target": "user-configuration",
    "action_type": "navigate"
  }
}
```

**Expected Output**:
```json
{
  "Action_type": "navigation",
  "param": "name",
  "value": "user-configuration",
  "page": "user-configuration",
  "previous_page": "dashboard",
  "interaction_type": "page_navigation",
  "clicked": false
}
```

### Scenario 2: Click Configure Database Button

**User Input**: "Click configure database" (user is now on user-configuration page)

**Tool Call**:
```json
{
  "function_name": "navigate_page",
  "arguments": {
    "user_id": "current_user",
    "target": "configure database",
    "action_type": "click"
  }
}
```

**Expected Output**:
```json
{
  "Action_type": "navigation",
  "param": "clicked,name",
  "value": "true,configure database",
  "page": "user-configuration",
  "previous_page": "dashboard",
  "interaction_type": "button_click",
  "clicked": true,
  "element_name": "configure database",
  "context": null
}
```

### Scenario 3: Database Search

**User Input**: "Show all employee salaries" (user is on database-query page)

**Tool Call**:
```json
{
  "function_name": "navigate_page",
  "arguments": {
    "user_id": "current_user",
    "target": "Show all employee salaries",
    "action_type": "search"
  }
}
```

**Expected Output**:
```json
{
  "Action_type": "navigation",
  "param": "search,question",
  "value": "true,Show all employee salaries",
  "page": "database-query",
  "previous_page": "dashboard",
  "interaction_type": "database_search",
  "clicked": false,
  "search_query": "Show all employee salaries"
}
```

### Scenario 4: File System Search

**User Input**: "Find documents about project guidelines" (user is on file-query page)

**Tool Call**:
```json
{
  "function_name": "navigate_page",
  "arguments": {
    "user_id": "current_user",
    "target": "Find documents about project guidelines",
    "action_type": "file_search"
  }
}
```

**Expected Output**:
```json
{
  "Action_type": "vector_search_db",
  "param": "query,table_specific,tables[]",
  "value": "Find documents about project guidelines,false,['string']",
  "page": "file-query",
  "current_page": "file-query",
  "interaction_type": "file_search",
  "search_query": "Find documents about project guidelines",
  "table_specific": false,
  "tables": ["string"]
}
```

**Example with Table Mention**:
```json
{
  "Action_type": "vector_search_db",
  "param": "query,table_specific,tables[]",
  "value": "search for high-paid employees in Bangalore,true,['finance']",
  "page": "file-query",
  "current_page": "file-query",
  "interaction_type": "file_search",
  "search_query": "search for high-paid employees in Bangalore",
  "table_specific": true,
  "tables": ["finance"]
}
```

### Scenario 5: File System Upload

**User Input**: "Upload a document" (user is on file-query page)

**Tool Call**:
```json
{
  "function_name": "navigate_page",
  "arguments": {
    "user_id": "current_user",
    "target": "Upload a document",
    "action_type": "file_upload"
  }
}
```

**Expected Output**:
```json
{
  "Action_type": "vector_upload_db",
  "param": "file_descriptions[],table_names[]",
  "value": "['Upload a document'],[]",
  "page": "file-query",
  "current_page": "file-query",
  "interaction_type": "file_upload",
  "upload_request": "Upload a document",
  "file_descriptions": ["Upload a document"],
  "table_names": []
}
```

**Example with Table Mention**:
```json
{
  "Action_type": "vector_upload_db",
  "param": "file_descriptions[],table_names[]",
  "value": "['Upload employee data'],['finance']",
  "page": "file-query",
  "current_page": "file-query",
  "interaction_type": "file_upload",
  "upload_request": "Upload employee data",
  "file_descriptions": ["Upload employee data"],
  "table_names": ["finance"]
}
```

## Available Pages and Elements

### Pages:
- `dashboard` - No interactive elements
- `database-query` - Database search enabled (users can perform MSSQL queries)
- `file-query` - File system search and upload enabled (users can perform file operations)
- `company-structure` - No interactive elements
- `user-configuration` - Has buttons: "configure database", "configure business rule"
- `table-management` - No interactive elements
- `users` - Has buttons: "manage database access", "vector database access"

## Context Awareness

The tool maintains user session state:
- **Current Page**: Tracks where the user currently is
- **Previous Page**: Remembers where they came from
- **Interaction History**: Logs all navigation and interaction events

## Smart Recognition

The tool can understand various ways users might express their intent:

### Page Navigation Examples (Different Page):
- "Go to user configuration" → `user-configuration` (action_type: "navigate")
- "Take me to the users page" → `users` (action_type: "navigate")
- "I want to see the dashboard" → `dashboard` (action_type: "navigate")
- "Navigate to database query" → `database-query` (action_type: "navigate")

### Button Click Examples (Same Page):
- "Click configure database" → Clicks "configure database" button on current page (action_type: "click")
- "Open configure business rule" → Clicks "configure business rule" button (action_type: "click")
- "Go to manage database access" → Clicks "manage database access" button (action_type: "click")
- "Access vector database" → Clicks "vector database access" button (action_type: "click")

### Database Search Examples (Same Page):
- "Show all employee salaries" → Performs database search on database-query page (action_type: "search")
- "Find customer records" → Performs database search on database-query page (action_type: "search")
- "Get sales data" → Performs database search on database-query page (action_type: "search")
- "Search for products" → Performs database search on database-query page (action_type: "search")

### File System Search Examples (Same Page):
- "Find documents about project guidelines" → Performs file search on file-query page (action_type: "file_search")
- "Search for employee policies" → Performs file search on file-query page (action_type: "file_search")
- "Find customer contracts" → Performs file search on file-query page (action_type: "file_search")
- "Search in table hr_docs" → Performs file search in specific table (action_type: "file_search")

### File System Upload Examples (Same Page):
- "Upload a document" → Performs file upload on file-query page (action_type: "file_upload")
- "Upload files to table bill" → Performs file upload to specific table (action_type: "file_upload")
- "Upload numeric file" → Performs file upload with description (action_type: "file_upload")

## WebSocket Message Format

All navigation actions send structured data via WebSocket:

```json
{
  "type": "navigation_result" | "file_system_search_result" | "file_system_upload_result",
  "action": "navigate" | "click" | "interact" | "search" | "vector_search_db" | "vector_upload_db",
  "data": {
    "Action_type": "navigation" | "vector_search_db" | "vector_upload_db",
    "param": "name" | "clicked,name" | "search,question" | "query,table_specific,tables[]" | "file_descriptions[],table_names[]",
    "value": "page-name" | "true,element-name" | "true,search-query" | "query,table_specific,tables" | "file_descriptions,table_names", 
    "page": "current-page" | "vector_db_search" | "vector_upload_db",
    "previous_page": "previous-page" (for navigation),
    "current_page": "current-page" (for file operations),
    "interaction_type": "page_navigation" | "button_click" | "database_search" | "file_search" | "file_upload",
    "clicked": false | true (for navigation/click),
    "element_name": "button-name" (if applicable),
    "search_query": "search-query" (if applicable),
    "file_descriptions": "file-descriptions" (if applicable),
    "table_names": "table-names" (if applicable),
    "context": "additional-info" (if applicable)
  }
}
```

## Testing

Use the test endpoints to validate functionality:

```bash
# Navigate to a page
curl -X POST "http://localhost:8002/test-navigation?user_id=test_user&target=user-configuration&action_type=navigate"

# Click a button
curl -X POST "http://localhost:8002/test-navigation?user_id=test_user&target=configure%20database&action_type=click"

# Test database search (navigate to database-query first, then search)
curl -X POST "http://localhost:8002/test-navigation?user_id=test_user&target=database-query&action_type=navigate"
curl -X POST "http://localhost:8002/test-navigation?user_id=test_user&target=Show%20all%20employee%20salaries&action_type=search"

# Test integrated database search
curl -X GET "http://localhost:8002/test-database-search"

# Run comprehensive tests
python test_navigation_scenarios.py
```

## Future Extensibility

The tool is designed to support future enhancements:

- **Form Interactions**: Can be extended to handle form filling
- **Dropdown Selections**: Support for dropdown menus
- **Additional Parameters**: Context field can carry extra data
- **Complex Workflows**: Multi-step interactions with validation

## Error Handling

The tool provides helpful error messages:

- **Invalid Page**: "Invalid page 'xyz'. Available pages: dashboard, users, ..."
- **Invalid Element**: "Element 'xyz' not found on page 'abc'. Available elements: ..."
- **No Elements**: "Page 'abc' has no interactive elements."
