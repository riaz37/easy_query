# Navigation Tool Format Consistency Summary

## Overview
The navigation tool now uses a **consistent format** for all actions (navigation, button clicks, and database searches). All outputs follow the same structure with `Action_type: "navigation"`.

## Unified Format Structure

### Base Format
```json
{
  "Action_type": "navigation",
  "param": "action-specific-parameters",
  "value": "action-specific-values",
  "page": "current-page",
  "previous_page": "previous-page",
  "interaction_type": "action-type",
  "clicked": true/false
}
```

## Action-Specific Formats

### 1. Page Navigation
**User Input**: "Go to user-configuration"

**Output**:
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

### 2. Button Click
**User Input**: "Click configure database" (on user-configuration page)

**Output**:
```json
{
  "Action_type": "navigation",
  "param": "clicked,name",
  "value": "true,configure database",
  "page": "user-configuration",
  "previous_page": "dashboard",
  "interaction_type": "button_click",
  "clicked": true,
  "element_name": "configure database"
}
```

### 3. Database Search
**User Input**: "Show all employee salaries" (on database-query page)

**Output**:
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

## WebSocket Message Format

All actions send consistent WebSocket messages:

```json
{
  "type": "navigation_result",
  "action": "navigate" | "click" | "search",
  "data": {
    // Same structure as above
  }
}
```

## Key Consistency Points

### ✅ Action Types
- **Navigation/Click**: `Action_type: "navigation"`
- **Database Search**: `Action_type: "navigation"`
- **File Search**: `Action_type: "vector_search_db"`
- **File Upload**: `Action_type: "vector_upload_db"`

### ✅ Parameter Structure
- **Page Navigation**: `param: "name"`
- **Button Clicks**: `param: "clicked,name"`
- **Database Search**: `param: "search,question"`
- **File Search**: `param: "query,table_specific,tables[]"`
- **File Upload**: `param: "file_descriptions[],table_names[]"`

### ✅ Value Structure
- **Page Navigation**: `value: "page-name"`
- **Button Clicks**: `value: "true,button-name"`
- **Database Search**: `value: "true,search-query"`
- **File Search**: `value: "query,table_specific,tables"`
- **File Upload**: `value: "file_descriptions,table_names"`

### ✅ Consistent Fields
- **All actions include**: `page`, `previous_page`, `interaction_type`, `clicked`
- **Context-specific fields**: `element_name` (for clicks), `search_query` (for searches)

## Migration Summary

### Before (Inconsistent):
```json
// Navigation
{"Action_type": "navigation", "param": "name", "value": "page"}

// Database Search (Different format)
{"Action_type": "search_db", "param": "question", "value": "query"}
```

### After (Correct Format):
```json
// Navigation
{"Action_type": "navigation", "param": "name", "value": "page"}

// Database Search
{"Action_type": "navigation", "param": "search,question", "value": "true,query"}

// File Search
{"Action_type": "vector_search_db", "param": "query,table_specific,tables[]", "value": "query,false,['string']", "page": "file-query"}

// File Search with Table
{"Action_type": "vector_search_db", "param": "query,table_specific,tables[]", "value": "query,true,['finance']", "page": "file-query"}

// File Upload
{"Action_type": "vector_upload_db", "param": "file_descriptions[],table_names[]", "value": "['desc'],['table']", "page": "file-query"}
```

## Benefits

1. **Unified Processing**: Frontend can handle all actions with the same logic
2. **Consistent WebSocket Messages**: All actions use `type: "navigation_result"`
3. **Simplified Integration**: No need for different message types
4. **Better Maintainability**: Single format to maintain and debug
5. **Future-Proof**: Easy to add new action types following the same pattern

## Testing

Run the format consistency test:
```bash
python test_format_consistency.py
```

This will verify that all actions follow the consistent format and WebSocket messages are properly structured.
