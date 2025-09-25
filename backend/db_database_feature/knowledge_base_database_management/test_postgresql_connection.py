#!/usr/bin/env python3
"""
Test script to verify PostgreSQL connection
"""
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import engine, Base
from sqlalchemy import text

def test_connection():
    """Test the PostgreSQL connection"""
    try:
        # Test basic connection
        with engine.connect() as connection:
            result = connection.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"✅ Successfully connected to PostgreSQL!")
            print(f"📊 Database version: {version}")
            
            # Test if we can create tables
            print("\n🔧 Testing table creation...")
            Base.metadata.create_all(bind=engine)
            print("✅ Tables created successfully!")
            
            # Test a simple query
            result = connection.execute(text("SELECT current_database(), current_user;"))
            db_info = result.fetchone()
            print(f"📁 Current database: {db_info[0]}")
            print(f"👤 Current user: {db_info[1]}")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Testing PostgreSQL connection...")
    print("=" * 50)
    
    success = test_connection()
    
    if success:
        print("\n🎉 All tests passed! PostgreSQL connection is working correctly.")
    else:
        print("\n💥 Tests failed! Please check your PostgreSQL configuration.")
        sys.exit(1)
