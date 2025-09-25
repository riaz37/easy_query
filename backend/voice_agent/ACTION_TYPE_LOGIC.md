# Navigation Tool Action Type Logic

## Overview
The navigation tool uses different action types based on whether the user is staying on the same page or moving to a different page.

## Action Type Rules

### 1. **Navigate** (`action_type: "navigate"`)
**Use when**: User wants to go to a **different page**

**Examples**:
- User is on `dashboard` and says "Go to user-configuration" → `action_type: "navigate"`
- User is on `user-configuration` and says "Take me to users page" → `action_type: "navigate"`
- User is on `users` and says "Go to database-query" → `action_type: "navigate"`

**Output Format**:
```json
{
  "Action_type": "navigation",
  "param": "name",
  "value": "target-page-name",
  "page": "target-page-name",
  "previous_page": "current-page",
  "interaction_type": "page_navigation",
  "clicked": false
}
```

### 2. **Click** (`action_type: "click"`)
**Use when**: User wants to interact with elements on the **same page**

**Examples**:
- User is on `user-configuration` and says "Click configure database" → `action_type: "click"`
- User is on `user-configuration` and says "Open configure business rule" → `action_type: "click"`
- User is on `users` and says "Click manage database access" → `action_type: "click"`

**Output Format**:
```json
{
  "Action_type": "navigation",
  "param": "clicked,name",
  "value": "true,element-name",
  "page": "current-page",
  "previous_page": "previous-page",
  "interaction_type": "button_click",
  "clicked": true,
  "element_name": "element-name"
}
```

### 3. **Search** (`action_type: "search"`)
**Use when**: User wants to perform database search on the **same page** (database-query page)

**Examples**:
- User is on `database-query` and says "Show all employee salaries" → `action_type: "search"`
- User is on `database-query` and says "Find customer records" → `action_type: "search"`
- User is on `database-query` and says "Get sales data" → `action_type: "search"`

**Output Format**:
```json
{
  "Action_type": "navigation",
  "param": "search,question",
  "value": "true,search-query",
  "page": "database-query",
  "previous_page": "previous-page",
  "interaction_type": "database_search",
  "clicked": false,
  "search_query": "search-query"
}
```

### 4. **File Search** (`action_type: "file_search"`)
**Use when**: User wants to perform file system search on the **same page** (file-query page)

**Examples**:
- User is on `file-query` and says "Find documents about project guidelines" → `action_type: "file_search"`
- User is on `file-query` and says "Search for employee policies" → `action_type: "file_search"`
- User is on `file-query` and says "Search in table hr_docs" → `action_type: "file_search"`

**Output Format**:
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

### 5. **File Upload** (`action_type: "file_upload"`)
**Use when**: User wants to perform file system upload on the **same page** (file-query page)

**Examples**:
- User is on `file-query` and says "Upload a document" → `action_type: "file_upload"`
- User is on `file-query` and says "Upload files to table bill" → `action_type: "file_upload"`
- User is on `file-query` and says "Upload numeric file" → `action_type: "file_upload"`

**Output Format**:
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

## Decision Flow

```
User Request → Check Current Page → Determine Action Type
     ↓
1. If target is a different page → action_type: "navigate"
2. If target is a button/element on same page → action_type: "click"
3. If target is a search query on database-query page → action_type: "search"
4. If target is a file search query on file-query page → action_type: "file_search"
5. If target is a file upload request on file-query page → action_type: "file_upload"
```

## Examples by Scenario

### Scenario 1: Page Navigation
```
Current Page: dashboard
User Says: "Go to user-configuration"
Action: navigate_page(target="user-configuration", action_type="navigate")
Result: User moves to user-configuration page
```

### Scenario 2: Button Click
```
Current Page: user-configuration
User Says: "Click configure database"
Action: navigate_page(target="configure database", action_type="click")
Result: Button click on user-configuration page
```

### Scenario 3: Database Search
```
Current Page: database-query
User Says: "Show all employee salaries"
Action: navigate_page(target="Show all employee salaries", action_type="search")
Result: Database search on database-query page
```

### Scenario 4: File System Operations
```
Current Page: dashboard
User Says: "Go to file-query" → action_type: "navigate"
Current Page: file-query
User Says: "Find documents about project guidelines" → action_type: "file_search"
Current Page: file-query
User Says: "Upload a document" → action_type: "file_upload"
Current Page: file-query
User Says: "Go to database-query" → action_type: "navigate"
```

### Scenario 5: Mixed Actions
```
Current Page: dashboard
User Says: "Go to database-query" → action_type: "navigate"
Current Page: database-query
User Says: "Show employee data" → action_type: "search"
Current Page: database-query
User Says: "Go to file-query" → action_type: "navigate"
Current Page: file-query
User Says: "Find documents about project guidelines" → action_type: "file_search"
Current Page: file-query
User Says: "Go to users page" → action_type: "navigate"
Current Page: users
User Says: "Click manage database access" → action_type: "click"
```

## Key Points

1. **Navigate**: Always for page-to-page movement
2. **Click**: Always for same-page element interactions
3. **Search**: Always for same-page database queries (only on database-query page)
4. **File Search**: Always for same-page file system searches (only on file-query page)
5. **File Upload**: Always for same-page file system uploads (only on file-query page)
6. **Context Matters**: The current page determines what action type to use
7. **Parameter Consistency**: 
   - Navigation/Click: `Action_type: "navigation"`
   - Database Search: `Action_type: "navigation"`
   - File Search: `Action_type: "vector_search_db"`
   - File Upload: `Action_type: "vector_upload_db"`
