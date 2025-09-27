# 🚀 Quick Reference Card - User Creation & Role Assignment

## 🔐 Admin Login (Get Token)
```bash
curl -X POST "http://localhost:8000/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "admin123"}'
```

## 👥 View Available Roles
```bash
curl -X GET "http://localhost:8000/roles" \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

## 👤 Create New User
```bash
curl -X POST "http://localhost:8000/signup" \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
         "username": "john_doe",
         "email": "john@company.com",
         "password": "SecurePass123!"
     }'
```

## 🔑 Assign Role to User
```bash
curl -X POST "http://localhost:8000/users/USER_ID/roles" \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
         "user_id": "USER_ID",
         "role_ids": ["ROLE_ID"]
     }'
```

## ✅ Verify User Roles
```bash
curl -X GET "http://localhost:8000/users/USER_ID/roles" \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

## 🔐 Test User Login
```bash
curl -X POST "http://localhost:8000/login" \
     -H "Content-Type: application/json" \
     -d '{
         "username": "john_doe",
         "password": "SecurePass123!"
     }'
```

## 🔍 Check User Permissions
```bash
curl -X GET "http://localhost:8000/my-permissions" \
     -H "Authorization: Bearer USER_TOKEN"
```

## 📋 Role Summary
| Role | Permissions | Use Case |
|------|-------------|----------|
| **super_admin** | All permissions | System administrators |
| **admin** | Company & data source management | Department heads |
| **manager** | Read & update companies/data sources | Team leaders |
| **user** | Read-only access | Regular employees |

## 🎯 Quick Start Commands

### 1. Start Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Run Practice Script
```bash
python practice_user_creation.py
```

### 3. Run Complete Tests
```bash
python workflow_test.py
```

## 🚨 Common Issues
- **"Invalid token"** → Re-login to get fresh token
- **"User not found"** → Check user ID
- **"Role not found"** → Check role ID
- **"Permission denied"** → Check user roles and permissions

## 📚 Full Guides
- **MANUAL_USER_CREATION_GUIDE.md** - Complete step-by-step guide
- **COMPLETE_WORKFLOW_GUIDE.md** - Full system workflow
- **README_RBAC.md** - System documentation

---

**💡 Tip:** Use the practice script for interactive learning!
