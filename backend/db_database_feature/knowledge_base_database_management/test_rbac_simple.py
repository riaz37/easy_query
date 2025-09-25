#!/usr/bin/env python3
"""
Simple test script for RBAC functionality
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_rbac():
    print("🚀 Testing RBAC System")
    
    # 1. Admin login
    print("\n1. Testing admin login...")
    login_data = {"username": "admin", "password": "admin123"}
    response = requests.post(f"{BASE_URL}/login", json=login_data)
    
    if response.status_code == 200:
        admin_token = response.json()["access_token"]
        print("✅ Admin login successful")
    else:
        print(f"❌ Admin login failed: {response.status_code}")
        return
    
    headers = {"Authorization": f"Bearer {admin_token}"}
    
    # 2. Get admin permissions
    print("\n2. Getting admin permissions...")
    response = requests.get(f"{BASE_URL}/my-permissions", headers=headers)
    if response.status_code == 200:
        permissions = response.json()
        print(f"✅ Admin has {len(permissions['permissions'])} permissions")
    else:
        print(f"❌ Failed to get permissions: {response.status_code}")
    
    # 3. Create test permission
    print("\n3. Creating test permission...")
    permission_data = {
        "name": "test_permission",
        "description": "Test permission",
        "resource": "test",
        "action": "read"
    }
    response = requests.post(f"{BASE_URL}/permissions", json=permission_data, headers=headers)
    if response.status_code == 200:
        permission_id = response.json()["id"]
        print(f"✅ Created permission: {permission_id}")
    else:
        print(f"❌ Failed to create permission: {response.status_code}")
        return
    
    # 4. Create test role
    print("\n4. Creating test role...")
    role_data = {
        "name": "test_role",
        "description": "Test role",
        "permission_ids": [permission_id]
    }
    response = requests.post(f"{BASE_URL}/roles", json=role_data, headers=headers)
    if response.status_code == 200:
        role_id = response.json()["id"]
        print(f"✅ Created role: {role_id}")
    else:
        print(f"❌ Failed to create role: {response.status_code}")
        return
    
    # 5. Create test user
    print("\n5. Creating test user...")
    user_data = {
        "username": "testuser_rbac",
        "email": "test@example.com",
        "password": "testpass123"
    }
    response = requests.post(f"{BASE_URL}/signup", json=user_data)
    if response.status_code == 200:
        user_id = response.json()["id"]
        print(f"✅ Created user: {user_id}")
    else:
        print(f"❌ Failed to create user: {response.status_code}")
        return
    
    # 6. Assign role to user
    print("\n6. Assigning role to user...")
    assignment_data = {"user_id": user_id, "role_ids": [role_id]}
    response = requests.post(f"{BASE_URL}/users/{user_id}/roles", json=assignment_data, headers=headers)
    if response.status_code == 200:
        print("✅ Role assigned to user")
    else:
        print(f"❌ Failed to assign role: {response.status_code}")
    
    # 7. Test user login
    print("\n7. Testing user login...")
    user_login = {"username": "testuser_rbac", "password": "testpass123"}
    response = requests.post(f"{BASE_URL}/login", json=user_login)
    if response.status_code == 200:
        user_token = response.json()["access_token"]
        print("✅ User login successful")
    else:
        print(f"❌ User login failed: {response.status_code}")
        return
    
    user_headers = {"Authorization": f"Bearer {user_token}"}
    
    # 8. Check user permissions
    print("\n8. Checking user permissions...")
    response = requests.get(f"{BASE_URL}/my-permissions", headers=user_headers)
    if response.status_code == 200:
        user_permissions = response.json()
        print(f"✅ User has {len(user_permissions['permissions'])} permissions")
    else:
        print(f"❌ Failed to get user permissions: {response.status_code}")
    
    # 9. Test permission check
    print("\n9. Testing permission check...")
    check_data = {"resource": "test", "action": "read"}
    response = requests.post(f"{BASE_URL}/check-permission", json=check_data, headers=user_headers)
    if response.status_code == 200:
        has_permission = response.json()["has_permission"]
        print(f"✅ Permission check: {has_permission}")
    else:
        print(f"❌ Permission check failed: {response.status_code}")
    
    print("\n🎉 RBAC test completed!")

if __name__ == "__main__":
    test_rbac()
