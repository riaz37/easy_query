#!/usr/bin/env python3
"""
Database migration script to add missing columns to existing tables
"""

from sqlalchemy import text
from database import engine

def migrate_database():
    """Add missing columns to existing tables"""
    print("🔄 Starting database migration...")
    
    with engine.connect() as conn:
        try:
            # Check if columns exist and add them if they don't
            print("📋 Checking users table...")
            
            # Check if is_active column exists
            result = conn.execute(text("""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'dbo' 
                AND TABLE_NAME = 'users' 
                AND COLUMN_NAME = 'is_active'
            """))
            
            if not result.fetchone():
                print("➕ Adding is_active column to users table...")
                conn.execute(text("ALTER TABLE dbo.users ADD is_active BIT DEFAULT 1 NOT NULL"))
                print("✅ Added is_active column")
            else:
                print("✅ is_active column already exists")
            
            # Check if updated_at column exists
            result = conn.execute(text("""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'dbo' 
                AND TABLE_NAME = 'users' 
                AND COLUMN_NAME = 'updated_at'
            """))
            
            if not result.fetchone():
                print("➕ Adding updated_at column to users table...")
                conn.execute(text("ALTER TABLE dbo.users ADD updated_at DATETIME2"))
                print("✅ Added updated_at column")
            else:
                print("✅ updated_at column already exists")
            
            # Check roles table
            print("📋 Checking roles table...")
            
            # Check if is_active column exists in roles
            result = conn.execute(text("""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'dbo' 
                AND TABLE_NAME = 'roles' 
                AND COLUMN_NAME = 'is_active'
            """))
            
            if not result.fetchone():
                print("➕ Adding is_active column to roles table...")
                conn.execute(text("ALTER TABLE dbo.roles ADD is_active BIT DEFAULT 1 NOT NULL"))
                print("✅ Added is_active column")
            else:
                print("✅ is_active column already exists")
            
            # Check if updated_at column exists in roles
            result = conn.execute(text("""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'dbo' 
                AND TABLE_NAME = 'roles' 
                AND COLUMN_NAME = 'updated_at'
            """))
            
            if not result.fetchone():
                print("➕ Adding updated_at column to roles table...")
                conn.execute(text("ALTER TABLE dbo.roles ADD updated_at DATETIME2"))
                print("✅ Added updated_at column")
            else:
                print("✅ updated_at column already exists")
            
            # Check permissions table
            print("📋 Checking permissions table...")
            
            # Check if is_active column exists in permissions
            result = conn.execute(text("""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'dbo' 
                AND TABLE_NAME = 'permissions' 
                AND COLUMN_NAME = 'is_active'
            """))
            
            if not result.fetchone():
                print("➕ Adding is_active column to permissions table...")
                conn.execute(text("ALTER TABLE dbo.permissions ADD is_active BIT DEFAULT 1 NOT NULL"))
                print("✅ Added is_active column")
            else:
                print("✅ is_active column already exists")
            
            # Check if updated_at column exists in permissions
            result = conn.execute(text("""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'dbo' 
                AND TABLE_NAME = 'permissions' 
                AND COLUMN_NAME = 'updated_at'
            """))
            
            if not result.fetchone():
                print("➕ Adding updated_at column to permissions table...")
                conn.execute(text("ALTER TABLE dbo.permissions ADD updated_at DATETIME2"))
                print("✅ Added updated_at column")
            else:
                print("✅ updated_at column already exists")
            
            # Commit the changes
            conn.commit()
            print("\n🎉 Database migration completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during migration: {e}")
            conn.rollback()
            raise

if __name__ == "__main__":
    migrate_database()
