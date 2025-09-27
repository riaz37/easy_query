# Navigation Tool Unified Return Format

## Overview
All navigation tool outputs now follow a **unified, consistent schema** that is easy for frontend applications to parse and handle. This ensures predictable data structure regardless of the action type.

## Unified Schema Structure

Every navigation action returns the same base structure with fields set to `null`/`false`/`[]` when not applicable:

```json
{
  "Action_type": "navigation",
  "param": "action_specific_parameters",
  "value": "formatted_value_string",
  "page": "current_page_name",
  "previous_page": "previous_page_name_or_null",
  "interaction_type": "specific_interaction_type",
  "clicked": true_or_false,
  "element_name": "element_name_or_null",
  "search_query": "query_string_or_null",
  "report_request": "report_request_or_null",
  "report_query": "report_query_or_null",
  "upload_request": "upload_request_or_null",
  "db_id": "database_id_or_null",
  "table_specific": true_or_false,
  "tables": ["table1", "table2"],
  "file_descriptions": ["desc1", "desc2"],
  "table_names": ["table1", "table2"],
  "context": "context_string_or_null",
  "timestamp": "iso_timestamp_string",
  "user_id": "user_id_string",
  "success": true_or_false,
  "error_message": "error_message_or_null"
}
```

## Field Descriptions

| Field | Type | Description | Always Present |
|-------|------|-------------|----------------|
| `Action_type` | string | Always "navigation" for consistency | ✅ |
| `param` | string | Action-specific parameter format | ✅ |
| `value` | string | Formatted value string for the action | ✅ |
| `page` | string | Current page where action occurred | ✅ |
| `previous_page` | string/null | Previous page name or null | ✅ |
| `interaction_type` | string | Specific type: "page_navigation", "button_click", "database_search", etc. | ✅ |
| `clicked` | boolean | True if button/element was clicked | ✅ |
| `element_name` | string/null | Name of clicked element or null | ✅ |
| `search_query` | string/null | Database/file search query or null | ✅ |
| `report_request` | string/null | View report request or null | ✅ |
| `report_query` | string/null | Generate report query or null | ✅ |
| `upload_request` | string/null | File upload request or null | ✅ |
| `db_id` | string/null | Database ID for set database or null | ✅ |
| `table_specific` | boolean | True if search is table-specific | ✅ |
| `tables` | array | List of tables for table-specific operations | ✅ |
| `file_descriptions` | array | List of file descriptions for uploads | ✅ |
| `table_names` | array | List of table names for uploads | ✅ |
| `context` | string/null | Additional context or null | ✅ |
| `timestamp` | string | ISO timestamp of action | ✅ |
| `user_id` | string | User ID who performed action | ✅ |
| `success` | boolean | True if action succeeded | ✅ |
| `error_message` | string/null | Error message or null | ✅ |

## Action-Specific Examples

### 1. Page Navigation
**User Action**: "Go to database-query page"

```json
{
  "Action_type": "navigation",
  "param": "name",
  "value": "database-query",
  "page": "database-query",
  "previous_page": "dashboard",
  "interaction_type": "page_navigation",
  "clicked": false,
  "element_name": null,
  "search_query": null,
  "report_request": null,
  "report_query": null,
  "upload_request": null,
  "db_id": null,
  "table_specific": false,
  "tables": [],
  "file_descriptions": [],
  "table_names": [],
  "context": null,
  "timestamp": "2024-01-15T10:30:00Z",
  "user_id": "user123",
  "success": true,
  "error_message": null
}
```

### 2. Button Click (Regular)
**User Action**: "Click load tables"

```json
{
  "Action_type": "navigation",
  "param": "clicked,name",
  "value": "true,load tables",
  "page": "tables",
  "previous_page": "database-query",
  "interaction_type": "button_click",
  "clicked": true,
  "element_name": "load tables",
  "search_query": null,
  "report_request": null,
  "report_query": null,
  "upload_request": null,
  "db_id": null,
  "table_specific": false,
  "tables": [],
  "file_descriptions": [],
  "table_names": [],
  "context": null,
  "timestamp": "2024-01-15T10:31:00Z",
  "user_id": "user123",
  "success": true,
  "error_message": null
}
```

### 3. Button Click (Set Database)
**User Action**: "Set database with ID 123"

```json
{
  "Action_type": "navigation",
  "param": "clicked,name,db_id",
  "value": "true,set database,123",
  "page": "tables",
  "previous_page": "database-query",
  "interaction_type": "button_click",
  "clicked": true,
  "element_name": "set database",
  "search_query": null,
  "report_request": null,
  "report_query": null,
  "upload_request": null,
  "db_id": "123",
  "table_specific": false,
  "tables": [],
  "file_descriptions": [],
  "table_names": [],
  "context": "db_id:123",
  "timestamp": "2024-01-15T10:32:00Z",
  "user_id": "user123",
  "success": true,
  "error_message": null
}
```

### 4. Database Search
**User Action**: "Show all employee salaries"

```json
{
  "Action_type": "navigation",
  "param": "search,question",
  "value": "true,Show all employee salaries",
  "page": "database-query",
  "previous_page": "dashboard",
  "interaction_type": "database_search",
  "clicked": false,
  "element_name": null,
  "search_query": "Show all employee salaries",
  "report_request": null,
  "report_query": null,
  "upload_request": null,
  "db_id": null,
  "table_specific": false,
  "tables": [],
  "file_descriptions": [],
  "table_names": [],
  "context": null,
  "timestamp": "2024-01-15T10:33:00Z",
  "user_id": "user123",
  "success": true,
  "error_message": null
}
```

### 5. File Search (General)
**User Action**: "Find documents about project guidelines"

```json
{
  "Action_type": "navigation",
  "param": "query,table_specific,tables[]",
  "value": "Find documents about project guidelines,false,[]",
  "page": "file-query",
  "previous_page": "database-query",
  "interaction_type": "file_search",
  "clicked": false,
  "element_name": null,
  "search_query": "Find documents about project guidelines",
  "report_request": null,
  "report_query": null,
  "upload_request": null,
  "db_id": null,
  "table_specific": false,
  "tables": [],
  "file_descriptions": [],
  "table_names": [],
  "context": null,
  "timestamp": "2024-01-15T10:34:00Z",
  "user_id": "user123",
  "success": true,
  "error_message": null
}
```

### 6. File Search (Table-Specific)
**User Action**: "Search for employee policies in table hr_docs"

```json
{
  "Action_type": "navigation",
  "param": "query,table_specific,tables[]",
  "value": "Search for employee policies,true,[\"hr_docs\"]",
  "page": "file-query",
  "previous_page": "database-query",
  "interaction_type": "file_search",
  "clicked": false,
  "element_name": null,
  "search_query": "Search for employee policies",
  "report_request": null,
  "report_query": null,
  "upload_request": null,
  "db_id": null,
  "table_specific": true,
  "tables": ["hr_docs"],
  "file_descriptions": [],
  "table_names": [],
  "context": "table_specific:true,tables:hr_docs",
  "timestamp": "2024-01-15T10:35:00Z",
  "user_id": "user123",
  "success": true,
  "error_message": null
}
```

### 7. File Upload
**User Action**: "Upload file with description 'numeric data' to table 'finance'"

```json
{
  "Action_type": "navigation",
  "param": "file_descriptions[],table_names[]",
  "value": "[\"numeric data\"],[\"finance\"]",
  "page": "file-query",
  "previous_page": "database-query",
  "interaction_type": "file_upload",
  "clicked": false,
  "element_name": null,
  "search_query": null,
  "report_request": null,
  "report_query": null,
  "upload_request": "Upload file with description 'numeric data' to table 'finance'",
  "db_id": null,
  "table_specific": false,
  "tables": [],
  "file_descriptions": ["numeric data"],
  "table_names": ["finance"],
  "context": "file_descriptions:numeric data,table_names:finance",
  "timestamp": "2024-01-15T10:36:00Z",
  "user_id": "user123",
  "success": true,
  "error_message": null
}
```

### 8. View Report
**User Action**: "Show me the sales report"

```json
{
  "Action_type": "navigation",
  "param": "clicked,name,report_request",
  "value": "true,view report,Show me the sales report from last month",
  "page": "database-query",
  "previous_page": "dashboard",
  "interaction_type": "view_report",
  "clicked": true,
  "element_name": "view report",
  "search_query": null,
  "report_request": "Show me the sales report from last month",
  "report_query": null,
  "upload_request": null,
  "db_id": null,
  "table_specific": false,
  "tables": [],
  "file_descriptions": [],
  "table_names": [],
  "context": null,
  "timestamp": "2024-01-15T10:37:00Z",
  "user_id": "user123",
  "success": true,
  "error_message": null
}
```

### 9. Generate Report
**User Action**: "Generate a financial report"

```json
{
  "Action_type": "navigation",
  "param": "clicked,name,report_query",
  "value": "true,report generation,Generate a financial report for Q1 2024",
  "page": "database-query",
  "previous_page": "dashboard",
  "interaction_type": "generate_report",
  "clicked": true,
  "element_name": "report generation",
  "search_query": null,
  "report_request": null,
  "report_query": "Generate a financial report for Q1 2024",
  "upload_request": null,
  "db_id": null,
  "table_specific": false,
  "tables": [],
  "file_descriptions": [],
  "table_names": [],
  "context": "financial data analysis",
  "timestamp": "2024-01-15T10:38:00Z",
  "user_id": "user123",
  "success": true,
  "error_message": null
}
```

## Frontend Integration Guidelines

### 1. Consistent Parsing
Since all responses have the same structure, you can use a single parser:

```javascript
function parseNavigationResponse(response) {
  // All responses have these base fields
  const baseData = {
    actionType: response.Action_type,
    page: response.page,
    previousPage: response.previous_page,
    interactionType: response.interaction_type,
    clicked: response.clicked,
    success: response.success,
    timestamp: response.timestamp,
    userId: response.user_id
  };

  // Action-specific fields (may be null)
  const specificData = {
    elementName: response.element_name,
    searchQuery: response.search_query,
    reportRequest: response.report_request,
    reportQuery: response.report_query,
    uploadRequest: response.upload_request,
    dbId: response.db_id,
    tableSpecific: response.table_specific,
    tables: response.tables,
    fileDescriptions: response.file_descriptions,
    tableNames: response.table_names,
    context: response.context,
    errorMessage: response.error_message
  };

  return { baseData, specificData };
}
```

### 2. Switch-Based Handling
```javascript
function handleNavigationResponse(response) {
  switch(response.interaction_type) {
    case 'page_navigation':
      handlePageNavigation(response);
      break;
    case 'button_click':
      handleButtonClick(response);
      break;
    case 'database_search':
      handleDatabaseSearch(response);
      break;
    case 'file_search':
      handleFileSearch(response);
      break;
    case 'file_upload':
      handleFileUpload(response);
      break;
    case 'view_report':
      handleViewReport(response);
      break;
    case 'generate_report':
      handleGenerateReport(response);
      break;
  }
}
```

### 3. Error Handling
```javascript
function processResponse(response) {
  if (!response.success) {
    showError(response.error_message);
    return;
  }

  // Process successful response
  handleNavigationResponse(response);
}
```

## Benefits for Frontend

### 1. **Predictable Structure**
- Every response has the same base fields
- No need to check if fields exist
- Consistent null/false/empty array values

### 2. **Easy Parsing**
- Single schema for all navigation actions
- No conditional logic for different response types
- Simple field access patterns

### 3. **Reliable Defaults**
- Unavailable fields are always set to logical defaults
- No undefined/null reference errors
- Consistent array types for list fields

### 4. **Future-Proof**
- Adding new action types won't break existing parsers
- New fields can be added with default values
- Backward compatibility maintained

### 5. **Debugging-Friendly**
- All responses include timestamp and user_id
- Success/error status clearly indicated
- Action context preserved for debugging

This unified format ensures your frontend can reliably handle all navigation tool outputs with minimal conditional logic and maximum predictability.
