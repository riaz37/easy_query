# 🔐 Complete Workflow Guide: Authentication, Permissions, Roles & User Management

## 📋 Overview
This guide provides a step-by-step workflow to get your RBAC (Role-Based Access Control) system fully operational. Follow each step in order to ensure everything works correctly.

## 🚀 Phase 1: Database Setup & Migration (COMPLETED ✅)
We've already completed this phase:
- ✅ Database migration script created and run
- ✅ Missing columns (`is_active`, `updated_at`) added to tables
- ✅ RBAC system initialized with default roles and permissions
- ✅ Default admin user created

## 🔧 Phase 2: Start the Server

### Step 1: Start FastAPI Server
```bash
cd db_database_feature/knowledge_base_database_management
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [XXXX] using StatReload
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Step 2: Verify Server is Running
Open your browser and go to: `http://localhost:8000/docs`

You should see the FastAPI interactive documentation (Swagger UI).

## 🔐 Phase 3: Test Authentication System

### Step 3: Test Admin Login
**Endpoint:** `POST /login`
**Payload:**
```json
{
    "username": "admin",
    "password": "admin123"
}
```

**Expected Response:**
```json
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer"
}
```

**Test using curl:**
```bash
curl -X POST "http://localhost:8000/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "admin123"}'
```

### Step 4: Test User Creation
**Endpoint:** `POST /signup`
**Headers:** `Authorization: Bearer {admin_token}`
**Payload:**
```json
{
    "username": "testuser",
    "email": "test@example.com",
    "password": "testpass123"
}
```

**Expected Response:**
```json
{
    "id": "uuid-here",
    "username": "testuser",
    "email": "test@example.com",
    "is_active": true,
    "created_at": "2024-01-01T00:00:00",
    "updated_at": null
}
```

## 👥 Phase 4: Test Role Management

### Step 5: View All Roles
**Endpoint:** `GET /roles`
**Headers:** `Authorization: Bearer {admin_token}`

**Expected Response:**
```json
[
    {
        "id": "uuid-1",
        "name": "super_admin",
        "description": "Super Administrator with all permissions",
        "is_active": true,
        "created_at": "2024-01-01T00:00:00",
        "updated_at": null,
        "permissions": [...]
    },
    {
        "id": "uuid-2",
        "name": "admin",
        "description": "Administrator with company and data source management permissions",
        "is_active": true,
        "created_at": "2024-01-01T00:00:00",
        "updated_at": null,
        "permissions": [...]
    },
    {
        "id": "uuid-3",
        "name": "manager",
        "description": "Manager with read and update permissions for companies and data sources",
        "is_active": true,
        "created_at": "2024-01-01T00:00:00",
        "updated_at": null,
        "permissions": [...]
    },
    {
        "id": "uuid-4",
        "name": "user",
        "description": "Regular user with read-only access to companies and data sources",
        "is_active": true,
        "created_at": "2024-01-01T00:00:00",
        "updated_at": null,
        "permissions": [...]
    }
]
```

### Step 6: View All Permissions
**Endpoint:** `GET /permissions`
**Headers:** `Authorization: Bearer {admin_token}`

**Expected Response:**
```json
[
    {
        "id": "uuid-1",
        "name": "users_create",
        "description": "Create new users",
        "resource": "users",
        "action": "create",
        "is_active": true,
        "created_at": "2024-01-01T00:00:00",
        "updated_at": null
    },
    {
        "id": "uuid-2",
        "name": "users_read",
        "description": "Read user information",
        "resource": "users",
        "action": "read",
        "is_active": true,
        "created_at": "2024-01-01T00:00:00",
        "updated_at": null
    }
    // ... more permissions
]
```

## 🔑 Phase 5: Test Permission System

### Step 7: Check User Permissions
**Endpoint:** `GET /my-permissions`
**Headers:** `Authorization: Bearer {admin_token}`

**Expected Response:**
```json
{
    "user_id": "uuid-here",
    "username": "admin",
    "permissions": [
        "users:create",
        "users:read",
        "users:update",
        "users:delete",
        "roles:create",
        "roles:read",
        "roles:update",
        "roles:delete",
        "permissions:create",
        "permissions:read",
        "permissions:update",
        "permissions:delete",
        "companies:create",
        "companies:read",
        "companies:update",
        "companies:delete",
        "data_sources:create",
        "data_sources:read",
        "data_sources:update",
        "data_sources:delete"
    ]
}
```

### Step 8: Test Permission Check
**Endpoint:** `POST /check-permission`
**Headers:** `Authorization: Bearer {admin_token}`
**Payload:**
```json
{
    "resource": "companies",
    "action": "create"
}
```

**Expected Response:**
```json
{
    "has_permission": true,
    "message": "User has permission: companies:create"
}
```

## 👤 Phase 6: Test User Role Assignment

### Step 9: Assign Role to User
**Endpoint:** `POST /users/{user_id}/roles`
**Headers:** `Authorization: Bearer {admin_token}`
**Payload:**
```json
{
    "user_id": "uuid-of-test-user",
    "role_ids": ["uuid-of-user-role"]
}
```

**Expected Response:**
```json
{
    "message": "Roles assigned successfully",
    "user_id": "uuid-of-test-user",
    "assigned_roles": [
        {
            "id": "uuid-of-user-role",
            "name": "user",
            "description": "Regular user with read-only access to companies and data sources"
        }
    ]
}
```

### Step 10: View User Roles
**Endpoint:** `GET /users/{user_id}/roles`
**Headers:** `Authorization: Bearer {admin_token}`

**Expected Response:**
```json
[
    {
        "id": "uuid-of-user-role",
        "name": "user",
        "description": "Regular user with read-only access to companies and data sources",
        "is_active": true,
        "created_at": "2024-01-01T00:00:00",
        "updated_at": null
    }
]
```

## 🧪 Phase 7: Test Protected Endpoints

### Step 11: Test Company Access with Different Permissions

**As Admin (Full Access):**
```bash
# Create company
curl -X POST "http://localhost:8000/companies" \
     -H "Authorization: Bearer {admin_token}" \
     -H "Content-Type: application/json" \
     -d '{"name": "Test Company", "description": "Test Description"}'

# Read company
curl -X GET "http://localhost:8000/companies/1" \
     -H "Authorization: Bearer {admin_token}"

# Update company
curl -X PUT "http://localhost:8000/companies/1" \
     -H "Authorization: Bearer {admin_token}" \
     -H "Content-Type: application/json" \
     -d '{"name": "Updated Company", "description": "Updated Description"}'

# Delete company
curl -X DELETE "http://localhost:8000/companies/1" \
     -H "Authorization: Bearer {admin_token}"
```

**As Regular User (Read Only):**
```bash
# This should work (read permission)
curl -X GET "http://localhost:8000/companies/1" \
     -H "Authorization: Bearer {user_token}"

# This should fail (no create permission)
curl -X POST "http://localhost:8000/companies" \
     -H "Authorization: Bearer {user_token}" \
     -H "Content-Type: application/json" \
     -d '{"name": "Test Company", "description": "Test Description"}'
```

## 🚪 Phase 8: Test Logout & Security

### Step 12: Test Logout
**Endpoint:** `POST /logout`
**Headers:** `Authorization: Bearer {token}`

**Expected Response:**
```json
{
    "message": "Successfully signed out",
    "success": true
}
```

### Step 13: Test Token Revocation
After logout, try to use the same token:
```bash
curl -X GET "http://localhost:8000/companies/1" \
     -H "Authorization: Bearer {revoked_token}"
```

**Expected Response:**
```json
{
    "detail": "Token has been revoked"
}
```

## 🎯 Phase 9: Run Complete Workflow Test

### Step 14: Run Automated Workflow Test
```bash
python workflow_test.py
```

This script will automatically test all the functionality step by step and provide detailed feedback.

## 📊 Phase 10: Monitor & Verify

### Step 15: Check Database Tables
Verify that all data is properly stored:

```sql
-- Check users
SELECT * FROM dbo.users;

-- Check roles
SELECT * FROM dbo.roles;

-- Check permissions
SELECT * FROM dbo.permissions;

-- Check user-role assignments
SELECT u.username, r.name as role_name
FROM dbo.users u
JOIN dbo.user_roles ur ON u.id = ur.user_id
JOIN dbo.roles r ON ur.role_id = r.id;

-- Check role-permission assignments
SELECT r.name as role_name, p.name as permission_name
FROM dbo.roles r
JOIN dbo.role_permissions rp ON r.id = rp.role_id
JOIN dbo.permissions p ON rp.permission_id = p.id
ORDER BY r.name, p.name;
```

## 🔒 Phase 11: Production Security Checklist

### Step 16: Security Hardening
- [ ] Change default admin password
- [ ] Set up proper environment variables
- [ ] Configure HTTPS
- [ ] Set up rate limiting
- [ ] Implement audit logging
- [ ] Set up monitoring and alerting

## 🎉 Success Criteria

Your RBAC system is fully operational when:

✅ **Authentication works:**
- Admin can login with `admin`/`admin123`
- Users can be created and login
- JWT tokens are generated and validated

✅ **Permissions work:**
- Permission checking endpoints respond correctly
- Users can only access resources they have permission for
- Permission denied errors are returned for unauthorized access

✅ **Roles work:**
- All default roles are created
- Roles can be assigned to users
- Role-based access control is enforced

✅ **User Management works:**
- Users can be created, updated, and deleted
- User roles can be assigned and modified
- User status (active/inactive) is respected

✅ **Protected Endpoints work:**
- Endpoints respect user permissions
- Different user types have appropriate access levels
- Unauthorized access is properly blocked

## 🚨 Troubleshooting

### Common Issues:

1. **"Invalid column name" errors:**
   - Run `python migrate_db.py` to add missing columns

2. **"User not found" errors:**
   - Run `python init_rbac.py` to create default admin user

3. **"Permission denied" errors:**
   - Check if user has the required role
   - Verify role has the required permission
   - Ensure user account is active

4. **"Token has expired" errors:**
   - Check token expiration time in `auth.py`
   - Re-login to get new token

5. **Database connection errors:**
   - Verify `.env` file has correct database credentials
   - Ensure SQL Server is running
   - Check ODBC driver installation

## 📞 Support

If you encounter issues:
1. Check the logs for detailed error messages
2. Verify database connectivity
3. Ensure all required packages are installed
4. Run the workflow test script for automated diagnostics

---

**🎯 Your RBAC system is now ready for production use!**
