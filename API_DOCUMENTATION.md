# API Documentation

This document provides detailed information about the Easy Query Knowledge Base API endpoints.

## Base URL

All endpoints are relative to: `https://localhost:8200`

## Authentication

Most endpoints require authentication using JWT Bearer tokens. The authentication system uses a comprehensive RBAC (Role-Based Access Control) system.

### Headers
```
Authorization: Bearer <your-jwt-token>
Content-Type: application/json
```

### Authentication Flow
1. **Signup**: Create a new user account
2. **Login**: Authenticate and receive JWT token
3. **Protected Endpoints**: Include token in Authorization header
4. **Logout**: Revoke token and end session

## Database Query Endpoints

### POST /mssql/query

Execute a natural language query against the configured database.

**Request Body:**
```json
{
  "question": "Show me all users who registered last month",
  "user_id": "default",
  "model": "gemini"
}
```

**Parameters:**
- `question` (string, required): Natural language question
- `user_id` (string, optional): User identifier (defaults to "default")
- `model` (string, optional): AI model to use ("gemini", "llama-3.3-70b-versatile", etc.)

**Response:**
```json
{
  "status_code": 200,
  "payload": {
    "sql": "SELECT * FROM users WHERE registration_date >= DATEADD(month, -1, GETDATE())",
    "data": [
      {
        "id": 1,
        "name": "John Doe",
        "email": "john@example.com",
        "registration_date": "2024-01-15"
      }
    ],
    "history": [],
    "model_used": "gemini"
  }
}
```

### POST /mssql/reload-db

Reload database schema and business rules for a user.

**Request Body:**
```json
{
  "user_id": "default"
}
```

**Response:**
```json
{
  "status_code": 200,
  "message": "reloaded from database for user: default",
  "table_info_preview": "Table: users\nColumns: id, name, email, registration_date\n\nTable: orders\nColumns: id, user_id, amount, order_date",
  "source": "database"
}
```

### GET /mssql/conversation-history/{user_id}

Retrieve conversation history for a user.

**Response:**
```json
{
  "status_code": 200,
  "message": "History loaded successfully.",
  "payload": [
    {
      "question": "Show me all users",
      "query": "SELECT * FROM users",
      "results": [
        {"id": 1, "name": "John Doe"},
        {"id": 2, "name": "Jane Smith"}
      ],
      "timestamp": "2024-01-15T10:30:00"
    }
  ]
}
```

### POST /mssql/clear-history/{user_id}

Clear conversation history for a user.

**Response:**
```json
{
  "status_code": 200,
  "message": "Conversation history cleared for user default"
}
```

### GET /mssql/available-models

Get list of available AI models.

**Response:**
```json
{
  "status_code": 200,
  "available_models": ["gemini", "llama-3.3-70b-versatile", "openai/gpt-oss-120b"],
  "default_model": "gemini",
  "model_details": {
    "gemini": {
      "provider": "google",
      "model_id": "gemini-2.0-flash"
    },
    "llama-3.3-70b-versatile": {
      "provider": "groq",
      "model_id": "llama-3.3-70b-versatile"
    }
  }
}
```

### GET /mssql/get_business-rules

Get business rules for a user.

**Response:**
```text
# Database Business Rules
Rule 1: All financial queries must include the current fiscal year
Rule 2: User data must be filtered by department for non-admin users

# User-Specific Business Rules
Rule 3: Only show active users in query results
```

### GET /mssql/data-source

Get information about the current data source.

**Response:**
```json
{
  "status_code": 200,
  "data_source": "database",
  "user_id": "default",
  "config_manager_url": "https://localhost:8200",
  "has_business_rules": true,
  "has_table_info": true,
  "table_count": 15,
  "business_rules_length": 256,
  "business_rules_sources": {
    "database_rules": true,
    "user_rules": true,
    "merged_rules": true,
    "database_rules_length": 128,
    "user_rules_length": 128,
    "merged_rules_length": 256
  }
}
```

### POST /mssql/switch-user

Switch to a different user context.

**Request Body:**
```json
{
  "user_id": "user123"
}
```

**Response:**
```json
{
  "status_code": 200,
  "message": "Switched to user: user123",
  "success": true,
  "data_source": "database"
}
```

## Authentication & RBAC Endpoints (`/Auth`)

The authentication system provides comprehensive user management with role-based access control.

### POST /Auth/signup

Create a new user account with validation.

**Request Body:**
```json
{
  "username": "newuser",
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response:**
```json
{
  "id": "uuid",
  "username": "newuser",
  "email": "user@example.com",
  "is_active": true,
  "created_at": "2024-01-15T10:30:00"
}
```

### POST /Auth/login

Authenticate user and receive JWT access token.

**Request Body:**
```json
{
  "username": "newuser",
  "password": "securepassword123"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

### POST /Auth/logout

Revoke current access token and end session.

**Headers:** `Authorization: Bearer <token>`

**Response:**
```json
{
  "message": "Successfully signed out"
}
```

### POST /Auth/change-password

Change user password with current password verification.

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "current_password": "oldpassword123",
  "new_password": "newpassword456"
}
```

**Response:**
```json
{
  "message": "Password changed successfully. Please login again."
}
```

### GET /Auth/profile

Get current user profile information.

**Headers:** `Authorization: Bearer <token>`

**Response:**
```json
{
  "id": "uuid",
  "username": "newuser",
  "email": "user@example.com",
  "is_active": true,
  "roles": ["user"],
  "permissions": ["read", "write"],
  "created_at": "2024-01-15T10:30:00"
}
```

### Company Management Endpoints

#### POST /Auth/companies

Create a new company with hierarchical structure.

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "name": "Acme Corp",
  "description": "Technology company",
  "parent_company_id": null
}
```

#### GET /Auth/companies

List companies with user permissions.

**Headers:** `Authorization: Bearer <token>`

**Response:**
```json
{
  "companies": [
    {
      "id": "uuid",
      "name": "Acme Corp",
      "description": "Technology company",
      "parent_company_id": null,
      "users": [],
      "created_at": "2024-01-15T10:30:00"
    }
  ]
}
```

## File Processing Endpoints (`/files`)

Advanced file processing system with intelligent pipeline routing and background processing.

### POST /files/smart_file_system

Smart file upload with automatic pipeline routing.

**Form Data:**
- `files`: Multiple files to upload (array)
- `file_descriptions`: JSON array of descriptions or single string
- `table_names`: JSON array of table names or single string
- `user_ids`: JSON array of user IDs or single string

**Response:**
```json
{
  "message": "Smart file processing initiated",
  "total_files": 3,
  "semi_structured_files": 1,
  "unstructured_files": 2,
  "processing_mode": "individual",
  "task_ids": ["task-uuid-1", "task-uuid-2", "task-uuid-3"]
}
```

### POST /files/smart_file_system_backend

Advanced file processing with custom parameters.

**Form Data:**
- `files`: Multiple files to upload
- `preserve_filenames`: Whether to preserve original filenames
- `delay_between_files`: Delay in seconds between processing files
- `max_pages_per_chunk`: Maximum pages per chunk for unstructured files
- `embed_batch_size`: Batch size for embedding generation
- `intent_similarity_threshold`: Similarity threshold for intent mapping
- `process_individually`: Process files individually or in batches

### POST /files/unstructured_file_system

Process unstructured documents (PDFs, Word, etc.).

**Form Data:**
- `files`: Multiple unstructured files
- `max_pages_per_chunk`: Pages per processing chunk
- `boundary_sentences`: Sentences for boundary detection
- `embed_batch_size`: Embedding generation batch size
- `sub_intent_similarity_threshold`: Sub-intent similarity threshold

### POST /files/semi_structured_file_system

Process structured documents (Excel, CSV, etc.).

**Form Data:**
- `file`: Single structured file
- `preserve_layout_alignment_across_pages`: Layout preservation
- `result_type`: Output format (markdown, html)
- `chunk_size`: Processing chunk size
- `similarity_threshold`: Similarity threshold for processing

### GET /files/task-status/{task_id}

Check processing task status.

**Response:**
```json
{
  "task_id": "task-uuid-1",
  "status": "completed",
  "progress": 100,
  "message": "Processing completed successfully",
  "results": {
    "files_processed": 3,
    "tables_created": 2,
    "embeddings_generated": 150,
    "processing_time": "00:02:30"
  }
}
```

### GET /files/list

List uploaded files and processing results.

**Response:**
```json
{
  "files": [
    {
      "id": "uuid",
      "name": "sales_data.xlsx",
      "description": "Monthly sales data",
      "status": "processed",
      "processing_type": "semi_structured",
      "tables_created": ["sales_data", "sales_summary"],
      "uploaded_at": "2024-01-15T10:30:00",
      "processed_at": "2024-01-15T10:32:00"
    }
  ]
}
```

## Report Generation Endpoints

### POST /reports/generate

Generate a report from query results.

**Request Body:**
```json
{
  "title": "Sales Report",
  "data": [
    {"month": "January", "sales": 1000},
    {"month": "February", "sales": 1200}
  ],
  "format": "html"
}
```

### POST /graph/generate

Generate charts and graphs.

**Request Body:**
```json
{
  "type": "bar",
  "data": [
    {"label": "Product A", "value": 100},
    {"label": "Product B", "value": 150}
  ],
  "title": "Product Sales"
}
```

## Database Configuration Endpoints

### GET /mssql-config/user-current-db/{user_id}

Get current database configuration for a user.

### POST /mssql-config/mssql-config/{db_id}

Update database configuration.

### GET /new-table/user-business-rule/{user_id}

Get user-specific business rules.

## Voice Agent Endpoints (`/voice`)

Real-time voice interaction system with ultra-low latency and comprehensive capabilities.

### GET /voice/health

Health check for voice agent with environment information.

**Response:**
```json
{
  "status": "healthy",
  "environment": {
    "mode": "development",
    "backend_url": "https://localhost:8200",
    "websocket_base": "wss://localhost:8200"
  },
  "llm_service": {
    "provider": "Gemini Live Multimodal + LangChain",
    "available": true,
    "function_calling": true,
    "native_audio": true,
    "text_agent": true,
    "expected_latency": "sub-500ms with native audio streaming"
  },
  "websockets": {
    "conversation": "wss://localhost:8200/voice/ws?user_id=your_user_id&current_page=database-query",
    "tools": "wss://localhost:8200/voice/ws/tools?user_id=your_user_id&current_page=database-query",
    "text_conversation": "wss://localhost:8200/voice/ws/text-conversation?user_id=your_user_id&current_page=database-query"
  },
  "sessions": {
    "active_sessions": 5,
    "total_users": 10,
    "users_with_tool_websockets": 8,
    "text_agents": 6
  }
}
```

### POST /voice/connect

Get WebSocket connection URL for voice interaction.

**Request Body:**
```json
{
  "user_id": "user123",
  "current_page": "database-query"
}
```

**Response:**
```json
{
  "websocket_url": "wss://localhost:8200/voice/ws?user_id=user123&current_page=database-query",
  "tools_websocket_url": "wss://localhost:8200/voice/ws/tools?user_id=user123&current_page=database-query",
  "text_websocket_url": "wss://localhost:8200/voice/ws/text-conversation?user_id=user123&current_page=database-query",
  "environment": "development",
  "features": {
    "voice_conversation": true,
    "tool_execution": true,
    "text_conversation": true,
    "page_awareness": true,
    "function_calling": true
  }
}
```

### WebSocket Endpoints

#### WS /voice/ws

Real-time voice conversation WebSocket.

**Query Parameters:**
- `user_id`: User identifier (required)
- `current_page`: Current page context (optional, defaults to "dashboard")

**Features:**
- Real-time voice input/output
- Function calling capabilities
- Page-aware responses
- Sub-500ms latency

#### WS /voice/ws/tools

Tool execution WebSocket for voice-controlled operations.

**Query Parameters:**
- `user_id`: User identifier (required)
- `current_page`: Current page context (optional)

**Capabilities:**
- Database query execution
- File operations
- Report generation
- Navigation commands

#### WS /voice/ws/text-conversation

Text-based conversation WebSocket with LangChain integration.

**Query Parameters:**
- `user_id`: User identifier (required)
- `current_page`: Current page context (optional)

**Features:**
- Text-based AI conversation
- Comprehensive navigation capabilities
- Tool execution
- Page-aware interactions

### Memory Management Endpoints

#### GET /voice/memory/{user_id}

Get conversation memory for a user.

**Response:**
```json
{
  "user_id": "user123",
  "memory": {
    "conversation_history": [...],
    "preferences": {...},
    "context": {...}
  }
}
```

#### POST /voice/memory/{user_id}/save

Save conversation memory for a user.

#### GET /voice/memory/{user_id}/load

Load conversation memory for a user.

#### DELETE /voice/memory/{user_id}

Clear conversation memory for a user.

### Test Endpoints

#### GET /voice/test-database-search

Test database search functionality.

#### GET /voice/test-function-call

Test function calling capabilities.

## Error Responses

All endpoints follow standard HTTP status codes:

- **200**: Success
- **400**: Bad Request - Invalid parameters
- **401**: Unauthorized - Missing or invalid authentication
- **403**: Forbidden - Insufficient permissions
- **404**: Not Found - Resource not found
- **500**: Internal Server Error - Server-side error

**Error Response Format:**
```json
{
  "status_code": 400,
  "error": "Invalid model specified",
  "message": "Model 'invalid-model' is not supported. Available models: gemini, llama-3.3-70b-versatile"
}
```