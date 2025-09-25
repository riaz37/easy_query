#!/usr/bin/env python3
"""
Script to run the main FastAPI application
"""
import uvicorn
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Starting Knowledge Base Database Management API...")
    print("=" * 60)
    print("📊 Database: PostgreSQL (localhost:5432)")
    print("👤 User: postgres")
    print("🗄️  Database: postgres")
    print("=" * 60)
    
    # Run the FastAPI application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
