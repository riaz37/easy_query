#!/usr/bin/env python3
"""
Test script for authentication and sign out functionality
"""

import requests
import json

# API base URL - adjust this to match your server
BASE_URL = "http://localhost:8000"

def test_auth_flow():
    """Test the complete authentication flow including sign out"""
    
    print("=== Testing Authentication and Sign Out Flow ===\n")
    
    # Test data
    test_user = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123"
    }
    
    # 1. Sign up
    print("1. Testing sign up...")
    try:
        response = requests.post(f"{BASE_URL}/signup", json=test_user)
        if response.status_code == 200:
            print("✅ Sign up successful")
            user_data = response.json()
            print(f"   User ID: {user_data['id']}")
        elif response.status_code == 400 and "already registered" in response.text:
            print("⚠️  User already exists, continuing with login...")
        else:
            print(f"❌ Sign up failed: {response.status_code} - {response.text}")
            return
    except Exception as e:
        print(f"❌ Sign up error: {e}")
        return
    
    # 2. Login
    print("\n2. Testing login...")
    try:
        login_data = {
            "username": test_user["username"],
            "password": test_user["password"]
        }
        response = requests.post(f"{BASE_URL}/login", json=login_data)
        if response.status_code == 200:
            print("✅ Login successful")
            token_data = response.json()
            access_token = token_data["access_token"]
            print(f"   Token type: {token_data['token_type']}")
        else:
            print(f"❌ Login failed: {response.status_code} - {response.text}")
            return
    except Exception as e:
        print(f"❌ Login error: {e}")
        return
    
    # 3. Test protected route
    print("\n3. Testing protected route...")
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(f"{BASE_URL}/protected", headers=headers)
        if response.status_code == 200:
            print("✅ Protected route accessible")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Protected route failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Protected route error: {e}")
    
    # 4. Test user profile
    print("\n4. Testing user profile...")
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(f"{BASE_URL}/profile", headers=headers)
        if response.status_code == 200:
            print("✅ User profile accessible")
            profile_data = response.json()
            print(f"   Username: {profile_data['username']}")
            print(f"   Email: {profile_data['email']}")
        else:
            print(f"❌ User profile failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ User profile error: {e}")
    
    # 5. Change password
    print("\n5. Testing change password...")
    try:
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        new_password = "new_" + test_user["password"]
        change_body = {
            "current_password": test_user["password"],
            "new_password": new_password
        }
        response = requests.post(f"{BASE_URL}/change-password", headers=headers, data=json.dumps(change_body))
        if response.status_code == 200:
            print("✅ Password change successful")
            print(f"   Message: {response.json()['message']}")
        else:
            print(f"❌ Password change failed: {response.status_code} - {response.text}")
            return
    except Exception as e:
        print(f"❌ Change password error: {e}")

    # 6. Verify old token is revoked after password change
    print("\n6. Testing token revocation after password change...")
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(f"{BASE_URL}/protected", headers=headers)
        if response.status_code == 401:
            print("✅ Old token revoked after password change")
        else:
            print(f"❌ Old token not revoked: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Revocation check error: {e}")

    # 7. Login with new password
    print("\n7. Testing login with new password...")
    try:
        login_data = {
            "username": test_user["username"],
            "password": new_password
        }
        response = requests.post(f"{BASE_URL}/login", json=login_data)
        if response.status_code == 200:
            print("✅ Login with new password successful")
            token_data = response.json()
            new_access_token = token_data["access_token"]
        else:
            print(f"❌ Login with new password failed: {response.status_code} - {response.text}")
            return
    except Exception as e:
        print(f"❌ Login error: {e}")

    # 8. Sign out with new token
    print("\n8. Testing sign out with new token...")
    try:
        headers = {"Authorization": f"Bearer {new_access_token}"}
        response = requests.post(f"{BASE_URL}/logout", headers=headers)
        if response.status_code == 200:
            print("✅ Sign out successful")
        else:
            print(f"❌ Sign out failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Sign out error: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_auth_flow()
