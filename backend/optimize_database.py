#!/usr/bin/env python3
"""
Database Optimization Script
============================

This script optimizes the report task database by:
1. Removing duplicate indexes
2. Cleaning up old completed tasks
3. Optimizing database performance
4. Adding the new results_summary column
"""

import sqlite3
import os
import sys
from datetime import datetime, timedelta

def optimize_report_task_database():
    """Optimize the report task database for better performance."""
    
    db_path = "report_task_tracking.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database file {db_path} not found")
        return False
    
    try:
        print("🔧 Starting database optimization...")
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        
        # Step 1: Add results_summary column if it doesn't exist
        print("📝 Adding results_summary column...")
        try:
            conn.execute("ALTER TABLE report_tasks ADD COLUMN results_summary TEXT")
            print("✅ Added results_summary column")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("✅ results_summary column already exists")
            else:
                print(f"⚠️  Could not add results_summary column: {e}")
        
        # Step 2: Add current_query column if it doesn't exist
        print("📝 Adding current_query column...")
        try:
            conn.execute("ALTER TABLE report_tasks ADD COLUMN current_query TEXT")
            print("✅ Added current_query column")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("✅ current_query column already exists")
            else:
                print(f"⚠️  Could not add current_query column: {e}")
        
        # Step 3: Remove duplicate indexes
        print("🗑️  Removing duplicate indexes...")
        
        # Get all indexes
        indexes = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='report_tasks'").fetchall()
        index_names = [idx[0] for idx in indexes]
        
        print(f"Found {len(index_names)} indexes: {index_names}")
        
        # Define the indexes we want to keep
        keep_indexes = [
            "idx_report_tasks_user_id",
            "idx_report_tasks_status", 
            "idx_report_tasks_created_at",
            "idx_report_tasks_user_status",
            "idx_report_tasks_user_created_at"
        ]
        
        # Remove duplicate indexes
        removed_count = 0
        for index_name in index_names:
            if index_name not in keep_indexes:
                try:
                    conn.execute(f"DROP INDEX {index_name}")
                    print(f"🗑️  Removed duplicate index: {index_name}")
                    removed_count += 1
                except Exception as e:
                    print(f"⚠️  Could not remove index {index_name}: {e}")
        
        print(f"✅ Removed {removed_count} duplicate indexes")
        
        # Step 4: Create optimized indexes
        print("🔨 Creating optimized indexes...")
        
        optimized_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_report_tasks_user_id ON report_tasks(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_report_tasks_status ON report_tasks(status)",
            "CREATE INDEX IF NOT EXISTS idx_report_tasks_created_at ON report_tasks(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_report_tasks_user_status ON report_tasks(user_id, status)",
            "CREATE INDEX IF NOT EXISTS idx_report_tasks_user_created_at ON report_tasks(user_id, created_at DESC)"
        ]
        
        for index_sql in optimized_indexes:
            try:
                conn.execute(index_sql)
            except Exception as e:
                print(f"⚠️  Could not create index: {e}")
        
        print("✅ Created optimized indexes")
        
        # Step 5: Clean up old completed tasks (older than 7 days)
        print("🧹 Cleaning up old completed tasks...")
        
        cutoff_date = (datetime.now() - timedelta(days=7)).isoformat()
        
        # Count old tasks
        old_tasks_count = conn.execute('''
            SELECT COUNT(*) FROM report_tasks 
            WHERE status = 'completed' AND created_at < ?
        ''', (cutoff_date,)).fetchone()[0]
        
        if old_tasks_count > 0:
            # Delete old completed tasks
            conn.execute('''
                DELETE FROM report_tasks 
                WHERE status = 'completed' AND created_at < ?
            ''', (cutoff_date,))
            
            print(f"✅ Cleaned up {old_tasks_count} old completed tasks")
        else:
            print("✅ No old tasks to clean up")
        
        # Step 6: Update results_summary for existing tasks
        print("📊 Updating results_summary for existing tasks...")
        
        # Get tasks with results but no summary
        tasks_to_update = conn.execute('''
            SELECT task_id, total_queries, successful_queries, failed_queries, results_summary
            FROM report_tasks 
            WHERE results IS NOT NULL AND (results_summary IS NULL OR results_summary = '')
        ''').fetchall()
        
        updated_count = 0
        for task in tasks_to_update:
            task_id, total_queries, successful_queries, failed_queries, results_summary = task
            
            if not results_summary:
                try:
                    # Create lightweight summary
                    summary_data = {
                        "total_queries": total_queries or 0,
                        "successful_queries": successful_queries or 0,
                        "failed_queries": failed_queries or 0,
                        "success_rate": (successful_queries / total_queries * 100) if total_queries and total_queries > 0 else 0
                    }
                    
                    import json
                    summary_json = json.dumps(summary_data)
                    
                    conn.execute('''
                        UPDATE report_tasks 
                        SET results_summary = ? 
                        WHERE task_id = ?
                    ''', (summary_json, task_id))
                    
                    updated_count += 1
                except Exception as e:
                    print(f"⚠️  Could not update summary for task {task_id}: {e}")
        
        print(f"✅ Updated results_summary for {updated_count} tasks")
        
        # Step 7: Optimize database
        print("🔧 Optimizing database...")
        
        # Commit current transaction before VACUUM
        conn.commit()
        
        # Analyze tables for better query planning
        conn.execute("ANALYZE")
        
        # Update statistics
        conn.execute("PRAGMA optimize")
        
        conn.close()
        
        # VACUUM needs to be done outside of a transaction
        print("🧹 Running VACUUM to reclaim space...")
        conn = sqlite3.connect(db_path)
        conn.execute("VACUUM")
        conn.close()
        
        print("✅ Database optimization completed successfully!")
        
        # Show final statistics
        conn = sqlite3.connect(db_path)
        total_tasks = conn.execute("SELECT COUNT(*) FROM report_tasks").fetchone()[0]
        completed_tasks = conn.execute("SELECT COUNT(*) FROM report_tasks WHERE status = 'completed'").fetchone()[0]
        running_tasks = conn.execute("SELECT COUNT(*) FROM report_tasks WHERE status = 'running'").fetchone()[0]
        failed_tasks = conn.execute("SELECT COUNT(*) FROM report_tasks WHERE status = 'failed'").fetchone()[0]
        
        print(f"\n📊 Database Statistics:")
        print(f"   Total tasks: {total_tasks}")
        print(f"   Completed: {completed_tasks}")
        print(f"   Running: {running_tasks}")
        print(f"   Failed: {failed_tasks}")
        
        # Check database size
        db_size = os.path.getsize(db_path) / (1024 * 1024)  # MB
        print(f"   Database size: {db_size:.2f} MB")
        
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error optimizing database: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("🔧 REPORT TASK DATABASE OPTIMIZATION")
    print("="*60)
    
    success = optimize_report_task_database()
    
    if success:
        print("\n🎉 Database optimization completed successfully!")
        print("📈 Performance should be significantly improved.")
        print("🚀 The APIs should now respond much faster.")
    else:
        print("\n❌ Database optimization failed!")
        sys.exit(1)