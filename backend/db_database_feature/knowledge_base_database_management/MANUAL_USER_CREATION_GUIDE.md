# 👤 Manual User Creation & Role Assignment Guide

## 📋 Overview
This guide shows you how to manually create new users and assign roles step-by-step using the API endpoints. Follow each step carefully to ensure proper user management.

## 🚀 Prerequisites
Before starting, make sure:
- ✅ FastAPI server is running on `http://localhost:8000`
- ✅ You have admin access (admin/admin123)
- ✅ Database is properly initialized

## 🔐 Step 1: Login as Admin

### 1.1 Get Admin Token
**Endpoint:** `POST /login`
**URL:** `http://localhost:8000/login`
**Headers:** `Content-Type: application/json`
**Body:**
```json
{
    "username": "admin",
    "password": "admin123"
}
```

**Using curl:**
```bash
curl -X POST "http://localhost:8000/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "admin123"}'
```

**Expected Response:**
```json
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer"
}
```

**Save the `access_token` value - you'll need it for all subsequent requests!**

### 1.2 Verify Admin Permissions
**Endpoint:** `GET /my-permissions`
**URL:** `http://localhost:8000/my-permissions`
**Headers:** 
```
Authorization: Bearer YOUR_ADMIN_TOKEN_HERE
Content-Type: application/json
```

**Using curl:**
```bash
curl -X GET "http://localhost:8000/my-permissions" \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN_HERE"
```

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

## 👥 Step 2: View Available Roles

### 2.1 Get All Roles
**Endpoint:** `GET /roles`
**URL:** `http://localhost:8000/roles`
**Headers:** 
```
Authorization: Bearer YOUR_ADMIN_TOKEN_HERE
Content-Type: application/json
```

**Using curl:**
```bash
curl -X GET "http://localhost:8000/roles" \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN_HERE"
```

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

**Note down the role IDs you want to assign to users!**

## 👤 Step 3: Create New User

### 3.1 Create User Account
**Endpoint:** `POST /signup`
**URL:** `http://localhost:8000/signup`
**Headers:** 
```
Authorization: Bearer YOUR_ADMIN_TOKEN_HERE
Content-Type: application/json
```
**Body:**
```json
{
    "username": "john_doe",
    "email": "john.doe@company.com",
    "password": "SecurePass123!"
}
```

**Using curl:**
```bash
curl -X POST "http://localhost:8000/signup" \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN_HERE" \
     -H "Content-Type: application/json" \
     -d '{
         "username": "john_doe",
         "email": "john.doe@company.com",
         "password": "SecurePass123!"
     }'
```

**Expected Response:**
```json
{
    "id": "new-user-uuid-here",
    "username": "john_doe",
    "email": "john.doe@company.com",
    "is_active": true,
    "created_at": "2024-01-01T00:00:00",
    "updated_at": null
}
```

**Save the user `id` - you'll need it for role assignment!**

### 3.2 Verify User Creation
**Endpoint:** `GET /users/{user_id}` (if available) or check the signup response
**Note:** The user is created but has no roles yet, so they have no permissions.

## 🔑 Step 4: Assign Roles to User

### 4.1 Assign Single Role
**Endpoint:** `POST /users/{user_id}/roles`
**URL:** `http://localhost:8000/users/{user_id}/roles`
**Headers:** 
```
Authorization: Bearer YOUR_ADMIN_TOKEN_HERE
Content-Type: application/json
```
**Body:**
```json
{
    "user_id": "new-user-uuid-here",
    "role_ids": ["uuid-of-role-to-assign"]
}
```

**Example - Assign 'user' role:**
```json
{
    "user_id": "new-user-uuid-here",
    "role_ids": ["uuid-of-user-role"]
}
```

**Using curl:**
```bash
curl -X POST "http://localhost:8000/users/new-user-uuid-here/roles" \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN_HERE" \
     -H "Content-Type: application/json" \
     -d '{
         "user_id": "new-user-uuid-here",
         "role_ids": ["uuid-of-user-role"]
     }'
```

**Expected Response:**
```json
{
    "message": "Roles assigned successfully",
    "user_id": "new-user-uuid-here",
    "assigned_roles": [
        {
            "id": "uuid-of-user-role",
            "name": "user",
            "description": "Regular user with read-only access to companies and data sources"
        }
    ]
}
```

### 4.2 Assign Multiple Roles
**Example - Assign both 'user' and 'manager' roles:**
```json
{
    "user_id": "new-user-uuid-here",
    "role_ids": ["uuid-of-user-role", "uuid-of-manager-role"]
}
```

## ✅ Step 5: Verify Role Assignment

### 5.1 Check User Roles
**Endpoint:** `GET /users/{user_id}/roles`
**URL:** `http://localhost:8000/users/{user_id}/roles`
**Headers:** 
```
Authorization: Bearer YOUR_ADMIN_TOKEN_HERE
Content-Type: application/json
```

**Using curl:**
```bash
curl -X GET "http://localhost:8000/users/new-user-uuid-here/roles" \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN_HERE"
```

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

### 5.2 Check User Permissions
**Endpoint:** `GET /my-permissions` (when logged in as the new user)
**Note:** This requires the new user to login first to see their permissions.

## 🔐 Step 6: Test New User Login

### 6.1 Login as New User
**Endpoint:** `POST /login`
**URL:** `http://localhost:8000/login`
**Headers:** `Content-Type: application/json`
**Body:**
```json
{
    "username": "john_doe",
    "password": "SecurePass123!"
}
```

**Using curl:**
```bash
curl -X POST "http://localhost:8000/login" \
     -H "Content-Type: application/json" \
     -d '{
         "username": "john_doe",
         "password": "SecurePass123!"
     }'
```

**Expected Response:**
```json
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer"
}
```

### 6.2 Check New User Permissions
**Endpoint:** `GET /my-permissions`
**URL:** `http://localhost:8000/my-permissions`
**Headers:** 
```
Authorization: Bearer NEW_USER_TOKEN_HERE
Content-Type: application/json
```

**Using curl:**
```bash
curl -X GET "http://localhost:8000/my-permissions" \
     -H "Authorization: Bearer NEW_USER_TOKEN_HERE"
```

**Expected Response (for 'user' role):**
```json
{
    "user_id": "new-user-uuid-here",
    "username": "john_doe",
    "permissions": [
        "companies:read",
        "data_sources:read"
    ]
}
```

## 🧪 Step 7: Test Permission Enforcement

### 7.1 Test Allowed Actions
**Test reading companies (should work):**
```bash
curl -X GET "http://localhost:8000/companies/1" \
     -H "Authorization: Bearer NEW_USER_TOKEN_HERE"
```

### 7.2 Test Denied Actions
**Test creating companies (should fail):**
```bash
curl -X POST "http://localhost:8000/companies" \
     -H "Authorization: Bearer NEW_USER_TOKEN_HERE" \
     -H "Content-Type: application/json" \
     -d '{"name": "Test Company", "description": "Test"}'
```

**Expected Response (403 Forbidden):**
```json
{
    "detail": "Insufficient permissions. Required: companies:create"
}
```

## 📝 Complete Example Workflow

Here's a complete example creating a manager user:

### 1. Admin Login
```bash
# Get admin token
ADMIN_TOKEN=$(curl -s -X POST "http://localhost:8000/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "admin123"}' | \
     jq -r '.access_token')

echo "Admin token: $ADMIN_TOKEN"
```

### 2. Create Manager User
```bash
# Create user
USER_RESPONSE=$(curl -s -X POST "http://localhost:8000/signup" \
     -H "Authorization: Bearer $ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
         "username": "sarah_manager",
         "email": "sarah.manager@company.com",
         "password": "ManagerPass456!"
     }')

USER_ID=$(echo $USER_RESPONSE | jq -r '.id')
echo "Created user with ID: $USER_ID"
```

### 3. Get Manager Role ID
```bash
# Get roles to find manager role ID
ROLES_RESPONSE=$(curl -s -X GET "http://localhost:8000/roles" \
     -H "Authorization: Bearer $ADMIN_TOKEN")

MANAGER_ROLE_ID=$(echo $ROLES_RESPONSE | jq -r '.[] | select(.name=="manager") | .id')
echo "Manager role ID: $MANAGER_ROLE_ID"
```

### 4. Assign Manager Role
```bash
# Assign manager role
curl -X POST "http://localhost:8000/users/$USER_ID/roles" \
     -H "Authorization: Bearer $ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d "{
         \"user_id\": \"$USER_ID\",
         \"role_ids\": [\"$MANAGER_ROLE_ID\"]
     }"
```

### 5. Test Manager Login
```bash
# Login as manager
MANAGER_TOKEN=$(curl -s -X POST "http://localhost:8000/login" \
     -H "Content-Type: application/json" \
     -d '{
         "username": "sarah_manager",
         "password": "ManagerPass456!"
     }' | jq -r '.access_token')

echo "Manager token: $MANAGER_TOKEN"
```

### 6. Check Manager Permissions
```bash
# Check permissions
curl -X GET "http://localhost:8000/my-permissions" \
     -H "Authorization: Bearer $MANAGER_TOKEN"
```

## 🔄 Step 8: Modify User Roles

### 8.1 Remove Specific Roles
**Endpoint:** `DELETE /users/{user_id}/roles`
**URL:** `http://localhost:8000/users/{user_id}/roles`
**Headers:** 
```
Authorization: Bearer YOUR_ADMIN_TOKEN_HERE
Content-Type: application/json
```
**Body:**
```json
{
    "user_id": "user-uuid-here",
    "role_ids": ["role-uuid-to-remove"]
}
```

### 8.2 Update User Roles (Replace All)
**Endpoint:** `PUT /users/{user_id}/roles` (if available) or use DELETE + POST

## 🚨 Common Issues & Solutions

### Issue 1: "User not found"
**Solution:** Verify the user ID is correct and the user exists.

### Issue 2: "Role not found"
**Solution:** Check the role ID and ensure the role exists and is active.

### Issue 3: "Insufficient permissions"
**Solution:** Ensure the admin user has the required permissions to manage users and roles.

### Issue 4: "Invalid token"
**Solution:** Re-login to get a fresh token, tokens expire after 30 minutes.

### Issue 5: "User already exists"
**Solution:** Use a different username or email, or update the existing user.

## 📊 Role Summary

| Role | Permissions | Use Case |
|------|-------------|----------|
| **super_admin** | All permissions | System administrators |
| **admin** | Company & data source management | Department heads |
| **manager** | Read & update companies/data sources | Team leaders |
| **user** | Read-only access | Regular employees |

## 🎯 Best Practices

1. **Password Security:**
   - Use strong passwords (12+ characters, mixed case, numbers, symbols)
   - Never share passwords
   - Consider implementing password policies

2. **Role Assignment:**
   - Follow principle of least privilege
   - Only assign necessary roles
   - Regularly review user permissions

3. **User Management:**
   - Use descriptive usernames
   - Use company email addresses
   - Document role assignments

4. **Security:**
   - Change default admin password
   - Use HTTPS in production
   - Implement rate limiting
   - Monitor access logs

## 🔍 Verification Checklist

After creating a user, verify:
- [ ] User can login successfully
- [ ] User has correct roles assigned
- [ ] User has appropriate permissions
- [ ] Permission enforcement works correctly
- [ ] User can access allowed resources
- [ ] User cannot access restricted resources

---

**🎉 You now know how to manually create users and assign roles!**

**Next Steps:**
1. Practice with the examples above
2. Create users for your team members
3. Assign appropriate roles based on job functions
4. Test permission enforcement
5. Set up production security measures
