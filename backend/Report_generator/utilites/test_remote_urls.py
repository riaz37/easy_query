#!/usr/bin/env python3
"""
Test script to verify remote URL generation for image files.
This script tests the URL conversion functions independently.
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root to the path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)

# Load environment variables
load_dotenv(override=True)

def test_url_generation():
    """Test URL generation functions."""
    print("🧪 Testing Remote URL Generation")
    print("="*50)
    
    # Check environment variables
    server_url = os.getenv("SERVER_URL")
    base_url = os.getenv("BASE_URL")
    
    print(f"🔍 Environment Variables:")
    print(f"   SERVER_URL: {server_url}")
    print(f"   BASE_URL: {base_url}")
    print()
    
    # Import the functions
    try:
        from Report_generator.utilites.graph_Generator import (
            get_image_server_url, 
            get_server_base_url, 
            convert_file_path_to_url
        )
        print("✅ Successfully imported URL functions")
    except ImportError as e:
        print(f"❌ Failed to import URL functions: {e}")
        return False
    
    # Test URL functions
    try:
        remote_url = get_image_server_url()
        local_url = get_server_base_url()
        
        print(f"✅ URL Functions Test:")
        print(f"   Remote Image URL: {remote_url}")
        print(f"   Local Server URL: {local_url}")
        print()
    except Exception as e:
        print(f"❌ URL Functions Test Failed: {e}")
        return False
    
    # Test URL conversion with different file types
    test_cases = [
        "/path/to/storage/graphs/images/test_graph_123456.png",
        "/path/to/storage/graphs/html/test_graph_123456.html",
        "/path/to/other/file.txt",
        None,
        ""
    ]
    
    print("🔄 Testing URL Conversion:")
    for i, test_path in enumerate(test_cases, 1):
        try:
            converted_url = convert_file_path_to_url(test_path)
            print(f"   Test {i}: {test_path}")
            print(f"   Result: {converted_url}")
            print()
        except Exception as e:
            print(f"   Test {i}: {test_path}")
            print(f"   Error: {e}")
            print()
    
    # Test with actual file path structure
    print("🔄 Testing with actual file path structure:")
    try:
        # Simulate the actual file path that would be generated
        timestamp = 1234567890
        filename = f"graph_bar_{timestamp}.png"
        actual_path = f"/Users/nilab/Desktop/projects/Esap-database_agent/Knowledge_base_backend/storage/graphs/images/{filename}"
        
        converted_url = convert_file_path_to_url(actual_path)
        print(f"   Actual path: {actual_path}")
        print(f"   Converted URL: {converted_url}")
        print()
        
        # Check if it's a remote URL
        if converted_url and converted_url.startswith("https://"):
            print("✅ SUCCESS: Generated remote URL correctly!")
            return True
        else:
            print("❌ FAILED: Generated URL is not remote!")
            return False
            
    except Exception as e:
        print(f"❌ Error testing actual path: {e}")
        return False

def test_environment_setup():
    """Test environment setup."""
    print("🔧 Testing Environment Setup")
    print("="*50)
    
    # Check if .env file exists
    env_file = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_file):
        print(f"✅ .env file found: {env_file}")
    else:
        print(f"⚠️  .env file not found: {env_file}")
    
    # Check environment variables
    required_vars = ["SERVER_URL", "BASE_URL"]
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
    
    print()

if __name__ == "__main__":
    print("🚀 Remote URL Generation Test")
    print("="*60)
    print()
    
    # Test environment setup
    test_environment_setup()
    
    # Test URL generation
    success = test_url_generation()
    
    print("="*60)
    if success:
        print("🎉 All tests passed! Remote URL generation is working correctly.")
    else:
        print("❌ Some tests failed. Please check the configuration.")
    
    print("="*60)
