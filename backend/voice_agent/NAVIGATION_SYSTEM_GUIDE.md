# Navigation System Guide

## Overview
The voice agent uses a unified navigation system that handles all user interactions through a single `navigate_page` function. This system provides a consistent interface for page navigation, button interactions, database queries, file operations, and report generation.

## Core Function: `navigate_page`

### Parameters
- `user_id` (string): User identifier for session management
- `target` (string): Target page name, element name, search query, or operation request
- `action_type` (string): Type of action to perform
- `context` (string, optional): Additional context or parameters

### Action Types

| Action Type | Description | Valid Pages | Example |
|-------------|-------------|-------------|---------|
| `navigate` | Go to a different page | All pages | "Go to database-query page" |
| `click` | Interact with buttons on same page | All pages with buttons | "Click view report" |
| `interact` | Other interactions | All pages | "Interact with form" |
| `search` | Perform database queries | database-query only | "Show all employee salaries" |
| `file_search` | Search file system | file-query only | "Find documents about guidelines" |
| `file_upload` | Upload files | file-query only | "Upload a document" |
| `view_report` | View existing reports | database-query only | "Show me the sales report" |
| `generate_report` | Create new reports | database-query only | "Generate a financial report" |

## Available Pages

### 📊 database-query
**Purpose**: Database queries and report operations

**Buttons**:
- `view report` - View existing reports
- `report generation` - Create new reports

**Features**:
- Database search enabled
- Report viewing enabled
- Report generation enabled

**Valid Actions**:
- `search` - Database queries
- `view_report` - View reports
- `generate_report` - Create reports
- `click` - Button interactions

**Example Usage**:
```python
# Database search
navigate_page(user_id="user", target="Show all employee salaries", action_type="search")

# View report
navigate_page(user_id="user", target="Show me the sales report", action_type="view_report")

# Generate report
navigate_page(user_id="user", target="Generate financial report", action_type="generate_report")

# Click button
navigate_page(user_id="user", target="view report", action_type="click")
```

### 📁 file-query
**Purpose**: File system operations

**Buttons**: None

**Features**:
- File search enabled
- File upload enabled

**Valid Actions**:
- `file_search` - Search file system
- `file_upload` - Upload files

**Context Format for File Operations**:
```
# Table-specific search
context="table_specific:true,tables:hr_docs,finance"

# File upload with descriptions
context="file_descriptions:numeric data,table_names:finance"

# Multiple files
context="file_descriptions:file1|file2,table_names:table1|table2"
```

**Example Usage**:
```python
# File search
navigate_page(user_id="user", target="Find employee policies", action_type="file_search")

# Table-specific search
navigate_page(user_id="user", target="Search for guidelines", action_type="file_search", 
             context="table_specific:true,tables:hr_docs")

# File upload
navigate_page(user_id="user", target="Upload document", action_type="file_upload", 
             context="file_descriptions:numeric data,table_names:finance")
```

### ⚙️ user-configuration
**Purpose**: User and system configuration

**Buttons**:
- `configure database` - Configure database settings
- `configure business rule` - Configure business rules

**Valid Actions**:
- `click` - Button interactions

**Example Usage**:
```python
navigate_page(user_id="user", target="configure database", action_type="click")
navigate_page(user_id="user", target="configure business rule", action_type="click")
```

### 📋 tables
**Purpose**: Table management and operations

**Buttons**:
- `load tables` - Load table data
- `reload tables` - Reload table data
- `table visualization` - Show table visualization
- `excel-import` - Import from Excel
- `get ai suggestion` - Get AI suggestions
- `submit` - Submit changes
- `set database` - Set database (requires db_id)

**Valid Actions**:
- `click` - Button interactions

**Special Requirements**:
- `set database` button requires `db_id` in context

**Context Format for Set Database**:
```
context="db_id:123"
```

**Example Usage**:
```python
# Regular button clicks
navigate_page(user_id="user", target="load tables", action_type="click")
navigate_page(user_id="user", target="reload tables", action_type="click")

# Set database (requires db_id)
navigate_page(user_id="user", target="set database", action_type="click", context="db_id:123")
```

### 👥 users
**Purpose**: User management

**Buttons**:
- `manage database access` - Manage database access
- `vector database access` - Manage vector database access

**Valid Actions**:
- `click` - Button interactions

**Example Usage**:
```python
navigate_page(user_id="user", target="manage database access", action_type="click")
navigate_page(user_id="user", target="vector database access", action_type="click")
```

### 🏠 dashboard
**Purpose**: Main dashboard page

**Buttons**: None

**Valid Actions**:
- `navigate` - Navigate to other pages

**Example Usage**:
```python
navigate_page(user_id="user", target="dashboard", action_type="navigate")
```

### 🏢 company-structure
**Purpose**: Company structure information

**Buttons**: None

**Valid Actions**:
- `navigate` - Navigate to other pages

**Example Usage**:
```python
navigate_page(user_id="user", target="company-structure", action_type="navigate")
```

## Context Format Guidelines

### File Operations
```
# Table-specific search
table_specific:true,tables:table1,table2

# File upload with descriptions
file_descriptions:description1,table_names:table1

# Multiple files (use | as separator)
file_descriptions:desc1|desc2,table_names:table1|table2
```

### Database Operations
```
# Set database (MANDATORY for set database button)
db_id:123
```

### Report Operations
```
# Include specific details about the report
context="financial data analysis"
context="Q1 2024 sales data"
```

## Error Handling

### Common Error Scenarios
1. **Invalid Page**: User tries to perform action on wrong page
2. **Missing Context**: Required context not provided (e.g., db_id for set database)
3. **Invalid Action**: Action type not valid for current page
4. **Empty Target**: No target specified

### Error Responses
- Clear, professional error messages
- Guidance on correct usage
- Suggestions for rephrasing requests

## Best Practices

### For AI Prompting
1. **Always check page context** before suggesting actions
2. **Use appropriate action types** for the target page
3. **Include required context** for special operations
4. **Validate user requests** before executing
5. **Provide clear feedback** on success/failure

### For Development
1. **Maintain consistent naming** for pages and buttons
2. **Document new features** in this guide
3. **Test all action types** on each page
4. **Validate context formats** for special operations
5. **Update system instructions** when adding new functionality

## Integration with Voice Agent

### System Instruction Updates
When adding new pages, buttons, or action types:
1. Update `VALID_PAGES` in `navigation_tool.py`
2. Add new handler methods if needed
3. Update `MSSQL_SEARCH_AI_SYSTEM_INSTRUCTION` in `agent_manager.py`
4. Update this documentation
5. Test with voice agent

### Function Registration
All navigation functionality is handled through the single `navigate_page` function, which is registered in the voice agent router.

## Future Enhancements

### Potential Additions
- New pages for additional functionality
- More sophisticated context parsing
- Enhanced error handling and recovery
- User preference management
- Session state persistence

### Considerations
- Maintain backward compatibility
- Keep the unified interface approach
- Document all changes thoroughly
- Test thoroughly before deployment
