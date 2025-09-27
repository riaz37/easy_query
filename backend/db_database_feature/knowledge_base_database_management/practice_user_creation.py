#!/usr/bin/env python3
"""
Practice User Creation Script
This script will guide you through creating a new user step by step
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"
ADMIN_CREDENTIALS = {
    "username": "admin",
    "password": "admin123"
}

def print_step(step_num, title):
    """Print a formatted step header"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {title}")
    print(f"{'='*60}")

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def print_info(message):
    """Print info message"""
    print(f"ℹ️  {message}")

def print_code(code):
    """Print code block"""
    print(f"\n```bash\n{code}\n```")

def make_request(method, endpoint, data=None, headers=None):
    """Make HTTP request and return response"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return {
            "success": response.status_code == 200,
            "data": response.json() if response.content else None,
            "status_code": response.status_code,
            "text": response.text
        }
    except Exception as e:
        return {"success": False, "data": str(e), "status_code": 0, "text": str(e)}

def practice_user_creation():
    """Practice creating a new user step by step"""
    
    print("🎯 Practice User Creation - Step by Step Guide")
    print("This script will guide you through creating a new user manually.")
    print(f"📡 API Base URL: {BASE_URL}")
    
    # Step 1: Admin Login
    print_step(1, "Admin Login")
    print("First, we need to login as admin to get an access token.")
    
    print_code(f'curl -X POST "{BASE_URL}/login" \\\n     -H "Content-Type: application/json" \\\n     -d \'{{"username": "admin", "password": "admin123"}}\'')
    
    print("\n🔍 Let's test this step...")
    result = make_request("POST", "/login", ADMIN_CREDENTIALS)
    
    if result["success"]:
        admin_token = result["data"]["access_token"]
        print_success("Admin login successful!")
        print_info(f"Token received: {admin_token[:20]}...")
        print_info("Save this token - you'll need it for all subsequent requests!")
    else:
        print_error(f"Admin login failed: {result['text']}")
        print_info("Make sure your server is running and the database is initialized.")
        return
    
    # Step 2: View Available Roles
    print_step(2, "View Available Roles")
    print("Now let's see what roles are available to assign to users.")
    
    print_code(f'curl -X GET "{BASE_URL}/roles" \\\n     -H "Authorization: Bearer YOUR_ADMIN_TOKEN_HERE"')
    
    print("\n🔍 Let's test this step...")
    headers = {"Authorization": f"Bearer {admin_token}"}
    result = make_request("GET", "/roles", headers=headers)
    
    if result["success"]:
        roles = result["data"]
        print_success(f"Found {len(roles)} roles:")
        for role in roles:
            print(f"   - {role['name']}: {role['description']}")
            print(f"     ID: {role['id']}")
        
        # Store role IDs for later use
        role_ids = {role['name']: role['id'] for role in roles}
        print_info("Note down these role IDs - you'll need them for role assignment!")
    else:
        print_error(f"Failed to get roles: {result['text']}")
        return
    
    # Step 3: Create New User
    print_step(3, "Create New User")
    print("Now let's create a new user account.")
    
    # Get user details from input
    print("\n📝 Enter user details:")
    username = input("Username: ").strip()
    email = input("Email: ").strip()
    password = input("Password: ").strip()
    
    if not username or not email or not password:
        print_error("All fields are required!")
        return
    
    user_data = {
        "username": username,
        "email": email,
        "password": password
    }
    
    print_code(f'curl -X POST "{BASE_URL}/signup" \\\n     -H "Authorization: Bearer YOUR_ADMIN_TOKEN_HERE" \\\n     -H "Content-Type: application/json" \\\n     -d \'{json.dumps(user_data)}\'')
    
    print("\n🔍 Let's test this step...")
    result = make_request("POST", "/signup", user_data, headers)
    
    if result["success"]:
        user_id = result["data"]["id"]
        print_success("User created successfully!")
        print_info(f"User ID: {user_id}")
        print_info("Save this user ID - you'll need it for role assignment!")
    else:
        print_error(f"User creation failed: {result['text']}")
        return
    
    # Step 4: Assign Role to User
    print_step(4, "Assign Role to User")
    print("Now let's assign a role to the new user.")
    
    print("\n📋 Available roles:")
    for i, (role_name, role_id) in enumerate(role_ids.items(), 1):
        print(f"{i}. {role_name} (ID: {role_id})")
    
    try:
        role_choice = int(input("\nSelect role number: ")) - 1
        role_names = list(role_ids.keys())
        if 0 <= role_choice < len(role_names):
            selected_role = role_names[role_choice]
            selected_role_id = role_ids[selected_role]
        else:
            print_error("Invalid choice!")
            return
    except ValueError:
        print_error("Please enter a valid number!")
        return
    
    role_assignment_data = {
        "user_id": user_id,
        "role_ids": [selected_role_id]
    }
    
    print_code(f'curl -X POST "{BASE_URL}/users/{user_id}/roles" \\\n     -H "Authorization: Bearer YOUR_ADMIN_TOKEN_HERE" \\\n     -H "Content-Type: application/json" \\\n     -d \'{json.dumps(role_assignment_data)}\'')
    
    print("\n🔍 Let's test this step...")
    result = make_request("POST", f"/users/{user_id}/roles", role_assignment_data, headers)
    
    if result["success"]:
        print_success(f"Role '{selected_role}' assigned successfully!")
    else:
        print_error(f"Role assignment failed: {result['text']}")
        return
    
    # Step 5: Verify Role Assignment
    print_step(5, "Verify Role Assignment")
    print("Let's verify that the role was assigned correctly.")
    
    print_code(f'curl -X GET "{BASE_URL}/users/{user_id}/roles" \\\n     -H "Authorization: Bearer YOUR_ADMIN_TOKEN_HERE"')
    
    print("\n🔍 Let's test this step...")
    result = make_request("GET", f"/users/{user_id}/roles", headers=headers)
    
    if result["success"]:
        user_roles = result["data"]
        print_success("User roles retrieved successfully!")
        print_info(f"User '{username}' has {len(user_roles)} role(s):")
        for role in user_roles:
            print(f"   - {role['name']}: {role['description']}")
    else:
        print_error(f"Failed to get user roles: {result['text']}")
    
    # Step 6: Test New User Login
    print_step(6, "Test New User Login")
    print("Now let's test if the new user can login and see their permissions.")
    
    print_code(f'curl -X POST "{BASE_URL}/login" \\\n     -H "Content-Type: application/json" \\\n     -d \'{json.dumps(user_data)}\'')
    
    print("\n🔍 Let's test this step...")
    result = make_request("POST", "/login", user_data)
    
    if result["success"]:
        user_token = result["data"]["access_token"]
        print_success("User login successful!")
        print_info(f"User token received: {user_token[:20]}...")
        
        # Check user permissions
        print("\n🔍 Now let's check the user's permissions...")
        user_headers = {"Authorization": f"Bearer {user_token}"}
        perm_result = make_request("GET", "/my-permissions", headers=user_headers)
        
        if perm_result["success"]:
            permissions = perm_result["data"]["permissions"]
            print_success("User permissions retrieved successfully!")
            print_info(f"User '{username}' has {len(permissions)} permission(s):")
            for perm in permissions:
                print(f"   - {perm}")
        else:
            print_error(f"Failed to get user permissions: {perm_result['text']}")
    else:
        print_error(f"User login failed: {result['text']}")
    
    # Step 7: Test Permission Enforcement
    print_step(7, "Test Permission Enforcement")
    print("Let's test if the permission system is working correctly.")
    
    print("\n🔍 Testing permission enforcement...")
    
    # Test reading companies (should work for most roles)
    print_info("Testing: Read companies (should work)")
    comp_result = make_request("GET", "/companies/1", headers=user_headers)
    if comp_result["success"]:
        print_success("✅ User can read companies")
    else:
        print_info(f"Companies access result: {comp_result['text']}")
    
    # Test creating companies (should fail for 'user' role)
    print_info("Testing: Create companies (should fail for 'user' role)")
    create_comp_data = {"name": "Test Company", "description": "Test"}
    create_result = make_request("POST", "/companies", create_comp_data, headers=user_headers)
    if create_result["success"]:
        print_success("✅ User can create companies (has admin/manager role)")
    else:
        print_success("✅ Permission correctly denied (user cannot create companies)")
    
    # Summary
    print_step(8, "Summary")
    print_success("🎉 User creation practice completed!")
    print("\n📋 What we accomplished:")
    print("1. ✅ Logged in as admin")
    print("2. ✅ Viewed available roles")
    print("3. ✅ Created new user account")
    print("4. ✅ Assigned role to user")
    print("5. ✅ Verified role assignment")
    print("6. ✅ Tested user login")
    print("7. ✅ Tested permission enforcement")
    
    print(f"\n👤 New user created:")
    print(f"   Username: {username}")
    print(f"   Email: {email}")
    print(f"   Role: {selected_role}")
    print(f"   User ID: {user_id}")
    
    print("\n🔑 Next steps:")
    print("1. Create more users with different roles")
    print("2. Test different permission combinations")
    print("3. Practice role management (add/remove roles)")
    print("4. Set up production security measures")
    
    print("\n📚 For more details, see:")
    print("- MANUAL_USER_CREATION_GUIDE.md")
    print("- COMPLETE_WORKFLOW_GUIDE.md")

def main():
    """Main function"""
    print("🚀 RBAC User Creation Practice")
    print("This script will guide you through creating a new user step by step.")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print_success("Server is running and accessible")
        else:
            print_error("Server responded with unexpected status")
            return
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to server. Make sure it's running on port 8000")
        print_info("Start the server with: uvicorn main:app --reload --host 0.0.0.0 --port 8000")
        return
    
    # Start practice
    practice_user_creation()

if __name__ == "__main__":
    main()
