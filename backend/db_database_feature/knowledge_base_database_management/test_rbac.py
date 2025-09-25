#!/usr/bin/env python3
"""
Test script for RBAC (Role-Based Access Control) functionality
"""

import requests
import json
import time
from typing import Dict, List

# API base URL - adjust this to match your server
BASE_URL = "http://localhost:8000"

class RBACTester:
    def __init__(self):
        self.admin_token = None
        self.user_token = None
        self.test_permission_id = None
        self.test_role_id = None
        self.test_user_id = None
        
    def print_section(self, title: str):
        """Print a section header"""
        print(f"\n{'='*50}")
        print(f" {title}")
        print(f"{'='*50}")
    
    def print_result(self, test_name: str, success: bool, message: str = ""):
        """Print test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if message:
            print(f"   {message}")
    
    def make_request(self, method: str, endpoint: str, data: dict = None, token: str = None) -> Dict:
        """Make HTTP request and return response"""
        url = f"{BASE_URL}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=headers, json=data)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            return {
                "status_code": response.status_code,
                "data": response.json() if response.content else None,
                "success": 200 <= response.status_code < 300
            }
        except Exception as e:
            return {
                "status_code": 0,
                "data": {"error": str(e)},
                "success": False
            }
    
    def test_admin_login(self):
        """Test admin login"""
        self.print_section("Testing Admin Login")
        
        login_data = {
            "username": "admin",
            "password": "admin123"
        }
        
        result = self.make_request("POST", "/login", login_data)
        
        if result["success"] and result["data"]:
            self.admin_token = result["data"]["access_token"]
            self.print_result("Admin Login", True, f"Token received: {self.admin_token[:20]}...")
            
            # Decode token to check roles and permissions
            import jwt
            try:
                payload = jwt.decode(self.admin_token, "esap-secret-key-kb-esap-secret-key-23334-KB", algorithms=["HS256"])
                roles = payload.get("roles", [])
                permissions = payload.get("permissions", [])
                self.print_result("Token Decode", True, f"Roles: {roles}, Permissions: {len(permissions)}")
            except Exception as e:
                self.print_result("Token Decode", False, str(e))
        else:
            self.print_result("Admin Login", False, f"Status: {result['status_code']}, Data: {result['data']}")
    
    def test_create_test_user(self):
        """Test creating a test user"""
        self.print_section("Creating Test User")
        
        user_data = {
            "username": "testuser_rbac",
            "email": "testuser_rbac@example.com",
            "password": "testpass123"
        }
        
        result = self.make_request("POST", "/signup", user_data)
        
        if result["success"] and result["data"]:
            self.test_user_id = result["data"]["id"]
            self.print_result("Create Test User", True, f"User ID: {self.test_user_id}")
        else:
            self.print_result("Create Test User", False, f"Status: {result['status_code']}, Data: {result['data']}")
    
    def test_user_login(self):
        """Test test user login"""
        self.print_section("Testing Test User Login")
        
        login_data = {
            "username": "testuser_rbac",
            "password": "testpass123"
        }
        
        result = self.make_request("POST", "/login", login_data)
        
        if result["success"] and result["data"]:
            self.user_token = result["data"]["access_token"]
            self.print_result("Test User Login", True, f"Token received: {self.user_token[:20]}...")
        else:
            self.print_result("Test User Login", False, f"Status: {result['status_code']}, Data: {result['data']}")
    
    def test_permission_management(self):
        """Test permission management"""
        self.print_section("Testing Permission Management")
        
        # Create permission
        permission_data = {
            "name": "test_permission",
            "description": "Test permission for RBAC testing",
            "resource": "test",
            "action": "read"
        }
        
        result = self.make_request("POST", "/permissions", permission_data, self.admin_token)
        
        if result["success"] and result["data"]:
            self.test_permission_id = result["data"]["id"]
            self.print_result("Create Permission", True, f"Permission ID: {self.test_permission_id}")
        else:
            self.print_result("Create Permission", False, f"Status: {result['status_code']}, Data: {result['data']}")
        
        # Get permissions
        result = self.make_request("GET", "/permissions", token=self.admin_token)
        self.print_result("Get Permissions", result["success"], f"Found {len(result['data']) if result['data'] else 0} permissions")
        
        # Get specific permission
        if self.test_permission_id:
            result = self.make_request("GET", f"/permissions/{self.test_permission_id}", token=self.admin_token)
            self.print_result("Get Specific Permission", result["success"])
    
    def test_role_management(self):
        """Test role management"""
        self.print_section("Testing Role Management")
        
        # Create role
        role_data = {
            "name": "test_role",
            "description": "Test role for RBAC testing",
            "permission_ids": [self.test_permission_id] if self.test_permission_id else []
        }
        
        result = self.make_request("POST", "/roles", role_data, self.admin_token)
        
        if result["success"] and result["data"]:
            self.test_role_id = result["data"]["id"]
            self.print_result("Create Role", True, f"Role ID: {self.test_role_id}")
        else:
            self.print_result("Create Role", False, f"Status: {result['status_code']}, Data: {result['data']}")
        
        # Get roles
        result = self.make_request("GET", "/roles", token=self.admin_token)
        self.print_result("Get Roles", result["success"], f"Found {len(result['data']) if result['data'] else 0} roles")
        
        # Get specific role
        if self.test_role_id:
            result = self.make_request("GET", f"/roles/{self.test_role_id}", token=self.admin_token)
            self.print_result("Get Specific Role", result["success"])
    
    def test_user_role_assignment(self):
        """Test user-role assignment"""
        self.print_section("Testing User-Role Assignment")
        
        if not self.test_user_id or not self.test_role_id:
            self.print_result("User-Role Assignment", False, "Missing user ID or role ID")
            return
        
        # Assign role to user
        assignment_data = {
            "user_id": self.test_user_id,
            "role_ids": [self.test_role_id]
        }
        
        result = self.make_request("POST", f"/users/{self.test_user_id}/roles", assignment_data, self.admin_token)
        self.print_result("Assign Role to User", result["success"])
        
        # Get user roles
        result = self.make_request("GET", f"/users/{self.test_user_id}/roles", token=self.admin_token)
        self.print_result("Get User Roles", result["success"], f"User has {len(result['data']) if result['data'] else 0} roles")
        
        # Get users by role
        result = self.make_request("GET", f"/roles/{self.test_role_id}/users", token=self.admin_token)
        self.print_result("Get Users by Role", result["success"], f"Role has {len(result['data']) if result['data'] else 0} users")
    
    def test_permission_checking(self):
        """Test permission checking"""
        self.print_section("Testing Permission Checking")
        
        # Check admin permissions
        result = self.make_request("GET", "/my-permissions", token=self.admin_token)
        self.print_result("Get Admin Permissions", result["success"], f"Admin has {len(result['data']['permissions']) if result['data'] and 'permissions' in result['data'] else 0} permissions")
        
        # Check user permissions
        result = self.make_request("GET", "/my-permissions", token=self.user_token)
        self.print_result("Get User Permissions", result["success"], f"User has {len(result['data']['permissions']) if result['data'] and 'permissions' in result['data'] else 0} permissions")
        
        # Check specific permission
        check_data = {
            "resource": "test",
            "action": "read"
        }
        
        result = self.make_request("POST", "/check-permission", check_data, self.admin_token)
        self.print_result("Check Admin Permission", result["success"], f"Has permission: {result['data']['has_permission'] if result['data'] else False}")
        
        result = self.make_request("POST", "/check-permission", check_data, self.user_token)
        self.print_result("Check User Permission", result["success"], f"Has permission: {result['data']['has_permission'] if result['data'] else False}")
    
    def test_access_control(self):
        """Test access control on protected endpoints"""
        self.print_section("Testing Access Control")
        
        # Test admin access to protected endpoints
        result = self.make_request("GET", "/protected", token=self.admin_token)
        self.print_result("Admin Access to Protected", result["success"])
        
        # Test user access to protected endpoints
        result = self.make_request("GET", "/protected", token=self.user_token)
        self.print_result("User Access to Protected", result["success"])
        
        # Test unauthorized access
        result = self.make_request("GET", "/protected")
        self.print_result("Unauthorized Access", not result["success"], "Correctly denied access")
    
    def test_cleanup(self):
        """Clean up test data"""
        self.print_section("Cleaning Up Test Data")
        
        # Remove role from user
        if self.test_user_id and self.test_role_id:
            result = self.make_request("DELETE", f"/users/{self.test_user_id}/roles?role_ids={self.test_role_id}", token=self.admin_token)
            self.print_result("Remove User Role", result["success"])
        
        # Delete test role
        if self.test_role_id:
            result = self.make_request("DELETE", f"/roles/{self.test_role_id}", token=self.admin_token)
            self.print_result("Delete Test Role", result["success"])
        
        # Delete test permission
        if self.test_permission_id:
            result = self.make_request("DELETE", f"/permissions/{self.test_permission_id}", token=self.admin_token)
            self.print_result("Delete Test Permission", result["success"])
    
    def run_all_tests(self):
        """Run all RBAC tests"""
        print("🚀 Starting RBAC System Tests")
        print(f"📡 Testing against: {BASE_URL}")
        
        try:
            self.test_admin_login()
            self.test_create_test_user()
            self.test_user_login()
            self.test_permission_management()
            self.test_role_management()
            self.test_user_role_assignment()
            self.test_permission_checking()
            self.test_access_control()
            self.test_cleanup()
            
            print("\n🎉 RBAC System Tests Completed!")
            
        except Exception as e:
            print(f"\n❌ Test execution failed: {e}")

def main():
    """Main test function"""
    tester = RBACTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()

