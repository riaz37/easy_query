# Role-Based Access Control (RBAC) System

This document describes the RBAC system implemented in the Knowledge Base Database Management API.

## Overview

The RBAC system provides fine-grained access control through a three-tier hierarchy:
- **Users** - Individual system users
- **Roles** - Collections of permissions assigned to users
- **Permissions** - Granular access rights for specific resources and actions

## Database Schema

### Core Tables

#### Users Table
```sql
CREATE TABLE dbo.users (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BIT DEFAULT 1 NOT NULL,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2
);
```

#### Roles Table
```sql
CREATE TABLE dbo.roles (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description VARCHAR(255),
    is_active BIT DEFAULT 1 NOT NULL,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2
);
```

#### Permissions Table
```sql
CREATE TABLE dbo.permissions (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description VARCHAR(255),
    resource VARCHAR(100) NOT NULL,  -- e.g., "companies", "users", "roles"
    action VARCHAR(50) NOT NULL,     -- e.g., "create", "read", "update", "delete"
    is_active BIT DEFAULT 1 NOT NULL,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2
);
```

### Association Tables

#### User-Roles Association
```sql
CREATE TABLE dbo.user_roles (
    user_id UNIQUEIDENTIFIER REFERENCES dbo.users(id),
    role_id UNIQUEIDENTIFIER REFERENCES dbo.roles(id),
    PRIMARY KEY (user_id, role_id)
);
```

#### Role-Permissions Association
```sql
CREATE TABLE dbo.role_permissions (
    role_id UNIQUEIDENTIFIER REFERENCES dbo.roles(id),
    permission_id UNIQUEIDENTIFIER REFERENCES dbo.permissions(id),
    PRIMARY KEY (role_id, permission_id)
);
```

## Default Roles and Permissions

### Default Roles

1. **super_admin** - Full system access
2. **admin** - Company and data source management
3. **manager** - Read and update access to companies and data sources
4. **user** - Read-only access to companies and data sources

### Default Permissions

The system includes permissions for:
- **User Management**: create, read, update, delete
- **Role Management**: create, read, update, delete
- **Permission Management**: create, read, update, delete
- **Company Management**: create, read, update, delete
- **Data Source Management**: create, read, update, delete

## API Endpoints

### Authentication Endpoints

#### POST `/login`
Enhanced login that includes user roles and permissions in JWT token.

**Response:**
```json
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer"
}
```

**JWT Token Payload:**
```json
{
    "sub": "username",
    "user_id": "uuid",
    "roles": ["super_admin", "admin"],
    "permissions": ["companies:create", "users:read"],
    "exp": 1234567890,
    "jti": "unique-token-id"
}
```

### Permission Management

#### POST `/permissions`
Create a new permission.

**Request:**
```json
{
    "name": "companies_create",
    "description": "Create new companies",
    "resource": "companies",
    "action": "create"
}
```

#### GET `/permissions`
Get all permissions with pagination.

#### GET `/permissions/{permission_id}`
Get permission by ID.

#### PUT `/permissions/{permission_id}`
Update permission.

#### DELETE `/permissions/{permission_id}`
Delete permission.

### Role Management

#### POST `/roles`
Create a new role with optional permissions.

**Request:**
```json
{
    "name": "data_analyst",
    "description": "Data analyst role",
    "permission_ids": ["uuid1", "uuid2"]
}
```

#### GET `/roles`
Get all roles with pagination.

#### GET `/roles/{role_id}`
Get role by ID.

#### PUT `/roles/{role_id}`
Update role.

#### DELETE `/roles/{role_id}`
Delete role.

### User-Role Assignment

#### POST `/users/{user_id}/roles`
Assign roles to a user.

**Request:**
```json
{
    "user_id": "uuid",
    "role_ids": ["uuid1", "uuid2"]
}
```

#### GET `/users/{user_id}/roles`
Get all roles for a user.

#### DELETE `/users/{user_id}/roles`
Remove specific roles from a user.

#### GET `/roles/{role_id}/users`
Get all users with a specific role.

### Permission Checking

#### POST `/check-permission`
Check if current user has a specific permission.

**Request:**
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

#### GET `/my-permissions`
Get all permissions for the current user.

**Response:**
```json
{
    "user_id": "uuid",
    "username": "john_doe",
    "permissions": ["companies:read", "users:update"]
}
```

### Utility Endpoints

#### GET `/permissions/resource/{resource}`
Get all permissions for a specific resource.

#### GET `/permissions/action/{action}`
Get all permissions for a specific action.

## Usage Examples

### Protecting Endpoints

#### Using Permission Decorators
```python
@app.post("/companies", response_model=CompanyOut)
def create_company(
    company: CompanyCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("companies", "create"))
):
    # Only users with companies:create permission can access this
    return create_company(db, company)
```

#### Using Role Decorators
```python
@app.get("/admin/dashboard")
def admin_dashboard(
    current_user: User = Depends(require_role("admin"))
):
    # Only users with admin role can access this
    return {"message": "Admin dashboard"}
```

### Checking Permissions Programmatically
```python
def some_business_logic(user: User, db: Session):
    if check_permission(user, "companies", "delete"):
        # User can delete companies
        pass
    else:
        # User cannot delete companies
        raise HTTPException(status_code=403, detail="Insufficient permissions")
```

## Initialization

### Running the Initialization Script
```bash
cd db_database_feature/knowledge_base_database_management
python init_rbac.py
```

This script will:
1. Create database tables
2. Create default permissions
3. Create default roles
4. Create a default admin user

### Default Admin Credentials
- **Username**: admin
- **Email**: admin@example.com
- **Password**: admin123
- **Role**: super_admin

⚠️ **IMPORTANT**: Change the admin password in production!

## Security Features

### JWT Token Enhancement
- Tokens now include user roles and permissions
- Automatic permission checking on protected endpoints
- Token revocation support

### Permission Granularity
- Resource-based permissions (companies, users, roles, etc.)
- Action-based permissions (create, read, update, delete)
- Hierarchical role system

### Access Control
- Automatic permission validation on API endpoints
- Role-based access control
- User status checking (active/inactive)

## Best Practices

### Permission Naming Convention
- Format: `{resource}_{action}`
- Examples: `companies_create`, `users_read`, `roles_update`

### Role Design
- Create roles based on job functions
- Assign minimal required permissions
- Use descriptive role names and descriptions

### Security Considerations
1. **Principle of Least Privilege**: Assign only necessary permissions
2. **Regular Audits**: Review user roles and permissions regularly
3. **Password Security**: Use strong passwords and change defaults
4. **Token Management**: Implement proper token expiration and revocation
5. **Input Validation**: Validate all user inputs

## Migration from Previous Version

If you're upgrading from a version without RBAC:

1. **Backup your database**
2. **Run the initialization script**: `python init_rbac.py`
3. **Update existing users**: Assign appropriate roles to existing users
4. **Test the system**: Verify all endpoints work with new permission system

## Troubleshooting

### Common Issues

1. **Permission Denied Errors**
   - Check if user has the required role/permission
   - Verify the permission exists in the database
   - Ensure the user account is active

2. **Role Assignment Issues**
   - Verify the role exists and is active
   - Check if the user exists and is active
   - Ensure proper UUID format for IDs

3. **Token Issues**
   - Check if token is valid and not expired
   - Verify token hasn't been revoked
   - Ensure proper Authorization header format

### Debug Endpoints

Use these endpoints to debug permission issues:
- `GET /my-permissions` - Check current user permissions
- `POST /check-permission` - Test specific permission
- `GET /users/{user_id}/roles` - Check user role assignments

## Future Enhancements

1. **Permission Groups**: Group related permissions
2. **Dynamic Permissions**: Runtime permission creation
3. **Audit Logging**: Track permission usage
4. **Permission Inheritance**: Hierarchical permission system
5. **Time-based Permissions**: Temporary permission grants
6. **Multi-tenant Support**: Company-specific permissions
