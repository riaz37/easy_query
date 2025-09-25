#!/usr/bin/env python3
"""
Standalone test script to verify PostgreSQL connection
"""
import sys
import os
from sqlalchemy import create_engine, text

# PostgreSQL Configuration (matching your config.py)
DB_CONFIG = {
    "HOST": "localhost",
    "PORT": 5432,  # Updated to match the correct port
    "DATABASE": "postgres",  # Using default database for now
    "USERNAME": "postgres",
    "PASSWORD": "postgres"  # Updated to correct password
}

def test_connection():
    """Test the PostgreSQL connection"""
    try:
        # Create connection string
        DATABASE_URL = f"postgresql://{DB_CONFIG['USERNAME']}:{DB_CONFIG['PASSWORD']}@{DB_CONFIG['HOST']}:{DB_CONFIG['PORT']}/{DB_CONFIG['DATABASE']}"
        
        # Create engine
        engine = create_engine(
            DATABASE_URL, 
            echo=True,
            connect_args={
                "connect_timeout": 10,
                "application_name": "knowledge_base_backend"
            }
        )
        
        # Test basic connection
        with engine.connect() as connection:
            result = connection.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"✅ Successfully connected to PostgreSQL!")
            print(f"📊 Database version: {version}")
            
            # Test a simple query
            result = connection.execute(text("SELECT current_database(), current_user;"))
            db_info = result.fetchone()
            print(f"📁 Current database: {db_info[0]}")
            print(f"👤 Current user: {db_info[1]}")
            
            # Test if we can create a simple table
            print("\n🔧 Testing table creation capability...")
            connection.execute(text("CREATE TABLE IF NOT EXISTS test_connection (id SERIAL PRIMARY KEY, name VARCHAR(50));"))
            connection.commit()
            print("✅ Test table created successfully!")
            
            # Clean up test table
            connection.execute(text("DROP TABLE IF EXISTS test_connection;"))
            connection.commit()
            print("✅ Test table cleaned up!")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Testing PostgreSQL connection...")
    print("=" * 50)
    print(f"🔗 Connecting to: {DB_CONFIG['HOST']}:{DB_CONFIG['PORT']}/{DB_CONFIG['DATABASE']}")
    print(f"👤 User: {DB_CONFIG['USERNAME']}")
    print("=" * 50)
    
    success = test_connection()
    
    if success:
        print("\n🎉 All tests passed! PostgreSQL connection is working correctly.")
    else:
        print("\n💥 Tests failed! Please check your PostgreSQL configuration.")
        print("\n🔍 Troubleshooting tips:")
        print("   - Make sure PostgreSQL is running on localhost:5433")
        print("   - Verify the database 'main_db' exists")
        print("   - Check that user 'postgres' has password '1234'")
        print("   - Ensure the user has necessary permissions")
        sys.exit(1)
