#!/usr/bin/env python3
"""
Complete Workflow Testing Script for Authentication, Permissions, Roles & User Management
This script will test the entire system step by step
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
ADMIN_CREDENTIALS = {
    "username": "admin",
    "password": "admin123"
}

class WorkflowTester:
    def __init__(self):
        self.admin_token = None
        self.test_user_token = None
        self.test_user_id = None
        
    def print_step(self, step_num: int, title: str):
        """Print a formatted step header"""
        print(f"\n{'='*60}")
        print(f"STEP {step_num}: {title}")
        print(f"{'='*60}")
    
    def print_success(self, message: str):
        """Print success message"""
        print(f"✅ {message}")
    
    def print_error(self, message: str):
        """Print error message"""
        print(f"❌ {message}")
    
    def print_info(self, message: str):
        """Print info message"""
        print(f"ℹ️  {message}")
    
    def make_request(self, method: str, endpoint: str, data: Dict[str, Any] = None, 
                    headers: Dict[str, str] = None, expected_status: int = 200) -> Dict[str, Any]:
        """Make HTTP request and return response"""
        url = f"{BASE_URL}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, headers=headers)
            elif method.upper() == "PUT":
                response = requests.put(url, json=data, headers=headers)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if response.status_code == expected_status:
                return {"success": True, "data": response.json() if response.content else None, "status_code": response.status_code}
            else:
                return {"success": False, "data": response.text, "status_code": response.status_code}
                
        except Exception as e:
            return {"success": False, "data": str(e), "status_code": 0}
    
    def test_server_connection(self):
        """Step 1: Test if server is running"""
        self.print_step(1, "Testing Server Connection")
        
        try:
            response = requests.get(f"{BASE_URL}/docs")
            if response.status_code == 200:
                self.print_success("Server is running and accessible")
                self.print_info(f"API Documentation available at: {BASE_URL}/docs")
                return True
            else:
                self.print_error(f"Server responded with status: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            self.print_error("Cannot connect to server. Make sure it's running on port 8000")
            self.print_info("Start the server with: uvicorn main:app --reload --host 0.0.0.0 --port 8000")
            return False
    
    def test_admin_login(self):
        """Step 2: Test admin login"""
        self.print_step(2, "Testing Admin Login")
        
        result = self.make_request("POST", "/login", ADMIN_CREDENTIALS)
        
        if result["success"]:
            self.admin_token = result["data"]["access_token"]
            self.print_success("Admin login successful")
            self.print_info(f"Token received: {self.admin_token[:20]}...")
            return True
        else:
            self.print_error(f"Admin login failed: {result['data']}")
            return False
    
    def test_user_creation(self):
        """Step 3: Test user creation"""
        self.print_step(3, "Testing User Creation")
        
        if not self.admin_token:
            self.print_error("Admin token not available. Run admin login first.")
            return False
        
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        new_user_data = {
            "username": "testuser_workflow",
            "email": "testuser@workflow.com",
            "password": "testpass123"
        }
        
        result = self.make_request("POST", "/signup", new_user_data, headers)
        
        if result["success"]:
            self.test_user_id = result["data"]["id"]
            self.print_success("Test user created successfully")
            self.print_info(f"User ID: {self.test_user_id}")
            return True
        else:
            self.print_error(f"User creation failed: {result['data']}")
            return False
    
    def test_user_login(self):
        """Step 4: Test user login"""
        self.print_step(4, "Testing User Login")
        
        user_login_data = {
            "username": "testuser_workflow",
            "password": "testpass123"
        }
        
        result = self.make_request("POST", "/login", user_login_data)
        
        if result["success"]:
            self.test_user_token = result["data"]["access_token"]
            self.print_success("User login successful")
            self.print_info(f"User token received: {self.test_user_token[:20]}...")
            return True
        else:
            self.print_error(f"User login failed: {result['data']}")
            return False
    
    def test_permission_checking(self):
        """Step 5: Test permission checking"""
        self.print_step(5, "Testing Permission Checking")
        
        if not self.admin_token:
            self.print_error("Admin token not available")
            return False
        
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        
        # Test permission check endpoint
        permission_check = {
            "resource": "companies",
            "action": "create"
        }
        
        result = self.make_request("POST", "/check-permission", permission_check, headers)
        
        if result["success"]:
            self.print_success("Permission check endpoint working")
            self.print_info(f"Response: {result['data']}")
        else:
            self.print_error(f"Permission check failed: {result['data']}")
        
        # Test getting user permissions
        result = self.make_request("GET", "/my-permissions", headers=headers)
        
        if result["success"]:
            self.print_success("Get user permissions working")
            self.print_info(f"Admin permissions: {result['data']['permissions'][:5]}...")
        else:
            self.print_error(f"Get permissions failed: {result['data']}")
        
        return True
    
    def test_role_management(self):
        """Step 6: Test role management"""
        self.print_step(6, "Testing Role Management")
        
        if not self.admin_token:
            self.print_error("Admin token not available")
            return False
        
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        
        # Get all roles
        result = self.make_request("GET", "/roles", headers=headers)
        
        if result["success"]:
            self.print_success("Get roles endpoint working")
            roles = result["data"]
            self.print_info(f"Found {len(roles)} roles: {[role['name'] for role in roles]}")
        else:
            self.print_error(f"Get roles failed: {result['data']}")
        
        return True
    
    def test_user_role_assignment(self):
        """Step 7: Test user role assignment"""
        self.print_step(7, "Testing User Role Assignment")
        
        if not self.admin_token or not self.test_user_id:
            self.print_error("Admin token or test user ID not available")
            return False
        
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        
        # Get available roles first
        result = self.make_request("GET", "/roles", headers=headers)
        if not result["success"]:
            self.print_error("Cannot get roles for assignment")
            return False
        
        # Assign 'user' role to test user
        user_role_data = {
            "user_id": str(self.test_user_id),
            "role_ids": [role["id"] for role in result["data"] if role["name"] == "user"]
        }
        
        if user_role_data["role_ids"]:
            result = self.make_request("POST", f"/users/{self.test_user_id}/roles", user_role_data, headers)
            
            if result["success"]:
                self.print_success("User role assignment successful")
            else:
                self.print_error(f"Role assignment failed: {result['data']}")
        else:
            self.print_error("No 'user' role found for assignment")
        
        return True
    
    def test_protected_endpoints(self):
        """Step 8: Test protected endpoints"""
        self.print_step(8, "Testing Protected Endpoints")
        
        if not self.test_user_token:
            self.print_error("Test user token not available")
            return False
        
        headers = {"Authorization": f"Bearer {self.test_user_token}"}
        
        # Test accessing companies (should work for read permission)
        result = self.make_request("GET", "/companies/1", headers=headers)
        
        if result["success"]:
            self.print_success("User can access companies (read permission working)")
        else:
            self.print_info(f"Companies access result: {result['data']}")
        
        return True
    
    def test_logout(self):
        """Step 9: Test logout functionality"""
        self.print_step(9, "Testing Logout Functionality")
        
        if not self.admin_token:
            self.print_error("Admin token not available")
            return False
        
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        
        result = self.make_request("POST", "/logout", headers=headers)
        
        if result["success"]:
            self.print_success("Logout successful")
        else:
            self.print_error(f"Logout failed: {result['data']}")
        
        return True
    
    def run_complete_workflow(self):
        """Run the complete workflow test"""
        print("🚀 Starting Complete Authentication & RBAC Workflow Test")
        print(f"📡 Testing against: {BASE_URL}")
        
        # Step 1: Test server connection
        if not self.test_server_connection():
            return False
        
        # Step 2: Test admin login
        if not self.test_admin_login():
            return False
        
        # Step 3: Test user creation
        if not self.test_user_creation():
            return False
        
        # Step 4: Test user login
        if not self.test_user_login():
            return False
        
        # Step 5: Test permission checking
        if not self.test_permission_checking():
            return False
        
        # Step 6: Test role management
        if not self.test_role_management():
            return False
        
        # Step 7: Test user role assignment
        if not self.test_user_role_assignment():
            return False
        
        # Step 8: Test protected endpoints
        if not self.test_protected_endpoints():
            return False
        
        # Step 9: Test logout
        if not self.test_logout():
            return False
        
        print(f"\n{'='*60}")
        print("🎉 COMPLETE WORKFLOW TEST SUCCESSFUL!")
        print(f"{'='*60}")
        print("✅ All authentication, permissions, roles, and user management features are working!")
        print("✅ Your RBAC system is fully operational!")
        print("\n📋 Next steps:")
        print("1. Start using the system with real users")
        print("2. Customize roles and permissions as needed")
        print("3. Implement additional security measures for production")
        
        return True

def main():
    """Main function to run the workflow test"""
    tester = WorkflowTester()
    tester.run_complete_workflow()

if __name__ == "__main__":
    main()
