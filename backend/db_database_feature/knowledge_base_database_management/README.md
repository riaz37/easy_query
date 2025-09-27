# Company Management API

This module provides comprehensive company management functionality with authentication, authorization, and role-based access control (RBAC).

## 🚀 Features

- **Company Management**: Full CRUD operations for companies with hierarchical structure
- **User Authentication**: Secure signup, login, logout, and password management
- **Role-Based Access Control (RBAC)**: Comprehensive permission and role management
- **Data Source Management**: File upload and management for company data sources
- **JWT Token Authentication**: Secure token-based authentication with revocation
- **Database Integration**: PostgreSQL with SQLAlchemy ORM

## 📋 API Endpoints

### 🔐 Authentication Endpoints

#### 1. User Signup
**POST** `/company-management/signup`

Create a new user account.

**Request Body:**
```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "securepassword123"
}
```

**Response:**
```json
{
  "id": "uuid",
  "username": "john_doe",
  "email": "john@example.com",
  "is_active": true,
  "created_at": "2024-01-15T10:30:00"
}
```

#### 2. User Login
**POST** `/company-management/login`

Authenticate user and receive access token.

**Request Body:**
```json
{
  "username": "john_doe",
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

#### 3. User Logout
**POST** `/company-management/logout`

Revoke current access token.

**Headers:** `Authorization: Bearer <token>`

**Response:**
```json
{
  "message": "Successfully signed out"
}
```

#### 4. Change Password
**POST** `/company-management/change-password`

Change user password (requires current password).

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

### 🏢 Company Management Endpoints

#### 1. Create Company
**POST** `/company-management/companies`

Create a new company.

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "name": "Acme Corporation",
  "description": "A leading technology company",
  "parent_id": null
}
```

#### 2. Get Company
**GET** `/company-management/companies/{company_id}`

Retrieve company details by ID.

**Headers:** `Authorization: Bearer <token>`

#### 3. Get All Companies
**GET** `/company-management/companies`

Retrieve all companies.

**Headers:** `Authorization: Bearer <token>`

#### 4. Update Company
**PUT** `/company-management/companies/{company_id}`

Update company details.

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "name": "Updated Company Name",
  "description": "Updated description"
}
```

#### 5. Delete Company
**DELETE** `/company-management/companies/{company_id}`

Delete a company.

**Headers:** `Authorization: Bearer <token>`

### 📁 Data Source Management

#### 1. Create Data Source
**POST** `/company-management/companies/{company_id}/data-sources`

Upload and create a data source for a company.

**Headers:** `Authorization: Bearer <token>`

**Form Data:**
- `name`: Data source name
- `description`: Optional description
- `business_rules_file_name`: Optional business rules file name
- `file`: File upload

### 🔐 Permission Management

#### 1. Create Permission
**POST** `/company-management/permissions`

Create a new permission.

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "resource": "companies",
  "action": "create",
  "description": "Create new companies"
}
```

#### 2. Get Permissions
**GET** `/company-management/permissions`

Retrieve all permissions with pagination.

**Headers:** `Authorization: Bearer <token>`

**Query Parameters:**
- `skip`: Number of records to skip (default: 0)
- `limit`: Maximum number of records (default: 100)

#### 3. Check Permission
**POST** `/company-management/check-permission`

Check if current user has a specific permission.

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "resource": "companies",
  "action": "create"
}
```

**Response:**
```json
{
  "has_permission": true,
  "message": "User has permission: companies:create"
}
```

#### 4. Get My Permissions
**GET** `/company-management/my-permissions`

Get all permissions for the current user.

**Headers:** `Authorization: Bearer <token>`

### 👥 Role Management

#### 1. Create Role
**POST** `/company-management/roles`

Create a new role.

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "name": "admin",
  "description": "Administrator role with full access"
}
```

#### 2. Get Roles
**GET** `/company-management/roles`

Retrieve all roles with pagination.

**Headers:** `Authorization: Bearer <token>`

#### 3. Assign Roles to User
**POST** `/company-management/users/{user_id}/roles`

Assign roles to a specific user.

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "role_ids": ["uuid1", "uuid2"]
}
```

### 🛡️ Protected Endpoints

#### 1. User Profile
**GET** `/company-management/profile`

Get current user profile.

**Headers:** `Authorization: Bearer <token>`

#### 2. Protected Route Example
**GET** `/company-management/protected`

Example of a protected route.

**Headers:** `Authorization: Bearer <token>`

## 🔧 Configuration

### Environment Variables
- `DATABASE_URL`: PostgreSQL connection string
- `SECRET_KEY`: JWT secret key for token signing
- `ALGORITHM`: JWT algorithm (default: HS256)
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Token expiration time

### Database Setup
The API automatically creates database tables on startup. Ensure your PostgreSQL database is running and accessible.

## 🎯 Usage Examples

### Python Client Example
```python
import requests

# Base configuration
BASE_URL = "https://localhost:8200"
headers = {"Content-Type": "application/json"}

# 1. Signup
signup_data = {
    "username": "testuser",
    "email": "test@example.com",
    "password": "password123"
}

response = requests.post(f"{BASE_URL}/company-management/signup", json=signup_data)
print(f"Signup: {response.status_code}")

# 2. Login
login_data = {
    "username": "testuser",
    "password": "password123"
}

response = requests.post(f"{BASE_URL}/company-management/login", json=login_data)
if response.status_code == 200:
    token = response.json()["access_token"]
    headers["Authorization"] = f"Bearer {token}"

# 3. Create Company
company_data = {
    "name": "My Company",
    "description": "A test company",
    "parent_id": None
}

response = requests.post(f"{BASE_URL}/company-management/companies", 
                        json=company_data, headers=headers)
print(f"Company created: {response.status_code}")
```

### cURL Examples
```bash
# Signup
curl -X POST "https://localhost:8200/company-management/signup" \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","email":"test@example.com","password":"password123"}' \
  --insecure

# Login
curl -X POST "https://localhost:8200/company-management/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"password123"}' \
  --insecure

# Create Company (with token)
curl -X POST "https://localhost:8200/company-management/companies" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -d '{"name":"My Company","description":"Test company"}' \
  --insecure
```

## 🧪 Testing

Use the provided test script to verify all endpoints:

```bash
python test_company_management_api.py
```

The test script will:
- Test authentication flow (signup, login, logout)
- Test company CRUD operations
- Test protected endpoints
- Test permission and role management
- Provide detailed success/failure reporting

## ⚠️ Error Handling

The API provides comprehensive error handling:

- **400 Bad Request**: Invalid request parameters
- **401 Unauthorized**: Invalid or missing authentication
- **403 Forbidden**: Insufficient permissions
- **404 Not Found**: Resource not found
- **500 Internal Server Error**: Server-side errors

## 🔐 Security Features

- **Password Hashing**: Secure bcrypt password hashing
- **JWT Tokens**: Stateless authentication with configurable expiration
- **Token Revocation**: Secure logout with token blacklisting
- **RBAC**: Fine-grained permission control
- **Input Validation**: Comprehensive request validation
- **SQL Injection Protection**: SQLAlchemy ORM protection

## 📝 Dependencies

- FastAPI
- SQLAlchemy
- PostgreSQL (psycopg2)
- Python-Jose (JWT)
- Passlib (password hashing)
- Pydantic (data validation)

## 🚀 Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```bash
   export DATABASE_URL="postgresql://user:password@localhost/dbname"
   export SECRET_KEY="your-secret-key"
   ```

3. Start the FastAPI server:
   ```bash
   python main.py
   ```

4. Access the API documentation:
   ```
   https://localhost:8200/docs
   ```

## 📞 Support

For issues or questions:
1. Check the server logs for detailed error information
2. Verify database connectivity and configuration
3. Test with the provided test script
4. Review the error handling section above
