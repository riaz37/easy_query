#!/usr/bin/env python3
"""
Quick Start Script for RBAC System
This script will help you get started quickly with authentication, permissions, roles, and user management
"""

import os
import sys
import subprocess
import time

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")

def print_step(step_num, title):
    """Print a formatted step"""
    print(f"\n📋 Step {step_num}: {title}")
    print("-" * 40)

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def print_info(message):
    """Print info message"""
    print(f"ℹ️  {message}")

def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")
    
    required_packages = [
        "fastapi",
        "uvicorn", 
        "sqlalchemy",
        "pyodbc",
        "python-dotenv",
        "passlib",
        "python-jose",
        "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print_success(f"{package} is installed")
        except ImportError:
            print_error(f"{package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print_info("Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print_success("All packages installed successfully")
        except subprocess.CalledProcessError:
            print_error("Failed to install packages. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def check_database_connection():
    """Check database connection"""
    print_header("Checking Database Connection")
    
    try:
        from database import engine
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        print_success("Database connection successful")
        return True
    except Exception as e:
        print_error(f"Database connection failed: {e}")
        print_info("Please check your .env file and ensure SQL Server is running")
        return False

def run_migration():
    """Run database migration if needed"""
    print_header("Running Database Migration")
    
    try:
        result = subprocess.run([sys.executable, "migrate_db.py"], 
                              capture_output=True, text=True, check=True)
        print_success("Database migration completed")
        return True
    except subprocess.CalledProcessError as e:
        if "already exists" in e.stdout or "already exists" in e.stderr:
            print_success("Database migration not needed (columns already exist)")
            return True
        else:
            print_error(f"Migration failed: {e.stderr}")
            return False

def run_rbac_init():
    """Run RBAC initialization if needed"""
    print_header("Initializing RBAC System")
    
    try:
        result = subprocess.run([sys.executable, "init_rbac.py"], 
                              capture_output=True, text=True, check=True)
        print_success("RBAC system initialized successfully")
        return True
    except subprocess.CalledProcessError as e:
        if "already exists" in e.stdout or "already exists" in e.stderr:
            print_success("RBAC system already initialized")
            return True
        else:
            print_error(f"RBAC initialization failed: {e.stderr}")
            return False

def start_server():
    """Start the FastAPI server"""
    print_header("Starting FastAPI Server")
    
    print_info("Starting server on http://localhost:8000")
    print_info("Press Ctrl+C to stop the server")
    print_info("Open http://localhost:8000/docs in your browser for API documentation")
    
    try:
        subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])
    except KeyboardInterrupt:
        print_info("Server stopped by user")
    except Exception as e:
        print_error(f"Failed to start server: {e}")

def show_next_steps():
    """Show what to do next"""
    print_header("Next Steps")
    
    print("🎯 Your RBAC system is now ready! Here's what you can do:")
    
    print("\n1. 🌐 Access the API:")
    print("   - Open http://localhost:8000/docs in your browser")
    print("   - This shows the interactive API documentation")
    
    print("\n2. 🔐 Test Authentication:")
    print("   - Login with admin/admin123")
    print("   - Create new users")
    print("   - Assign roles and permissions")
    
    print("\n3. 🧪 Run Complete Tests:")
    print("   - In a new terminal, run: python workflow_test.py")
    print("   - This will test all functionality automatically")
    
    print("\n4. 📚 Read the Complete Guide:")
    print("   - Check COMPLETE_WORKFLOW_GUIDE.md for detailed steps")
    
    print("\n5. 🔒 Production Setup:")
    print("   - Change default admin password")
    print("   - Set up proper environment variables")
    print("   - Configure HTTPS and security measures")

def main():
    """Main function"""
    print_header("RBAC System Quick Start")
    print("This script will help you get your RBAC system up and running quickly!")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check database connection
    if not check_database_connection():
        return
    
    # Run migration
    if not run_migration():
        return
    
    # Run RBAC initialization
    if not run_rbac_init():
        return
    
    # Show next steps
    show_next_steps()
    
    # Ask if user wants to start server
    print("\n" + "="*60)
    response = input("🚀 Would you like to start the server now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        start_server()
    else:
        print_info("You can start the server later with:")
        print("uvicorn main:app --reload --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    main()
