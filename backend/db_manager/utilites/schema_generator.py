import os
import json
import time
from decimal import Decimal
from datetime import datetime, date, time as dt_time
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# -------------------------------------------------
# CUSTOM JSON ENCODER FOR SQL DATA TYPES
# -------------------------------------------------
class SQLJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle SQL Server data types"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, dt_time):
            return obj.isoformat()
        elif isinstance(obj, (bytes, bytearray)):
            try:
                return obj.decode('utf-8')
            except UnicodeDecodeError:
                return obj.decode('utf-8', errors='ignore')
        elif hasattr(obj, 'isoformat'):  # Any other datetime-like objects
            return obj.isoformat()
        return super().default(obj)

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
DATABASE_URL = (
    "mssql+pyodbc://sa:Esap.12.Three@176.9.16.194,1433/"
    "JustForRestore?driver=ODBC+Driver+18+for+SQL+Server&encrypt=no"
)

OUTPUT_FILE = "all_tables_schema_and_data.json"
SAMPLE_ROW_COUNT = 50

# -------------------------------------------------
# OPTIMIZED SINGLE TABLE SCHEMA GENERATION
# -------------------------------------------------
def generate_single_table_schema(database_url: str, schema_name: str, table_name: str, sample_row_count: int = 10) -> dict:
    """
    Generate schema for a single table only - much faster than processing entire database.
    
    Args:
        database_url (str): The database connection URL
        schema_name (str): Schema name of the table
        table_name (str): Table name
        sample_row_count (int, optional): Number of sample rows to fetch. Default 10
    
    Returns:
        dict: Table schema information in the same format as the full schema generator
    """
    print(f"🔍 Generating schema for single table: {schema_name}.{table_name}")
    
    try:
        # Connect to database
        engine = create_engine(database_url, fast_executemany=True)
        conn = engine.connect()
        print(f"✅ Connected to database for table {schema_name}.{table_name}")
        
        # Helper functions for single table processing
        def get_columns_for_table(schema_name: str, table_name: str):
            """Get column information for the specific table"""
            sql = text("""
                SELECT 
                    COLUMN_NAME,
                    DATA_TYPE,
                    IS_NULLABLE,
                    CHARACTER_MAXIMUM_LENGTH
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = :schema
                  AND TABLE_NAME   = :table
                ORDER BY ORDINAL_POSITION
            """)
            result = conn.execute(sql, {"schema": schema_name, "table": table_name})
            columns = []
            for row in result.fetchall():
                col_name, data_type, is_nullable, max_len = row
                columns.append({
                    "name": col_name,
                    "type": data_type.upper(),
                    "is_nullable": (is_nullable == "YES"),
                    "max_length": max_len
                })
            return columns

        def get_primary_key_columns(schema_name: str, table_name: str):
            """Get primary key columns for the specific table"""
            full_obj_name = f"{schema_name}.{table_name}"
            sql = text("""
                SELECT c.name
                FROM sys.indexes i
                INNER JOIN sys.index_columns ic
                    ON i.object_id = ic.object_id
                   AND i.index_id  = ic.index_id
                INNER JOIN sys.columns c
                    ON c.object_id = ic.object_id
                   AND c.column_id = ic.column_id
                WHERE i.is_primary_key = 1
                  AND i.object_id = OBJECT_ID(:full_name)
                ORDER BY ic.key_ordinal
            """)
            result = conn.execute(sql, {"full_name": full_obj_name})
            return [row[0] for row in result.fetchall()]

        def get_foreign_keys_for_table(schema_name: str, table_name: str):
            """Get foreign key information for the specific table"""
            sql = text("""
                SELECT 
                    fk.name                            AS constraint_name,
                    parentCol.name                     AS fk_column,
                    schemaRef.name                     AS referenced_schema,
                    tabRef.name                        AS referenced_table,
                    refCol.name                        AS referenced_column
                FROM sys.foreign_keys AS fk
                INNER JOIN sys.foreign_key_columns AS fkc
                    ON fk.object_id = fkc.constraint_object_id
                INNER JOIN sys.tables AS tabParent
                    ON fk.parent_object_id = tabParent.object_id
                INNER JOIN sys.schemas AS schemaParent
                    ON tabParent.schema_id = schemaParent.schema_id
                INNER JOIN sys.columns AS parentCol
                    ON fkc.parent_object_id = parentCol.object_id
                   AND fkc.parent_column_id = parentCol.column_id
                INNER JOIN sys.tables AS tabRef
                    ON fk.referenced_object_id = tabRef.object_id
                INNER JOIN sys.schemas AS schemaRef
                    ON tabRef.schema_id = schemaRef.schema_id
                INNER JOIN sys.columns AS refCol
                    ON fkc.referenced_object_id = refCol.object_id
                   AND fkc.referenced_column_id = refCol.column_id
                WHERE schemaParent.name = :schema
                  AND tabParent.name   = :table
            """)
            result = conn.execute(sql, {"schema": schema_name, "table": table_name})
            fk_list = []
            for row in result.fetchall():
                constraint_name, fk_col, ref_schema, ref_table, ref_col = row
                fk_list.append({
                    "constraint_name": constraint_name,
                    "fk_column": fk_col,
                    "referenced_schema": ref_schema,
                    "referenced_table": ref_table,
                    "referenced_column": ref_col
                })
            return fk_list

        def get_tables_referencing_current(schema_name: str, table_name: str):
            """Get tables that reference the current table"""
            full_obj_name = f"{schema_name}.{table_name}"
            sql = text("""
                SELECT 
                    schemaParent.name AS parent_schema,
                    tabParent.name    AS parent_table,
                    parentCol.name    AS parent_column,
                    fk.name           AS constraint_name
                FROM sys.foreign_keys AS fk
                INNER JOIN sys.foreign_key_columns AS fkc
                    ON fk.object_id = fkc.constraint_object_id
                INNER JOIN sys.tables AS tabParent
                    ON fk.parent_object_id = tabParent.object_id
                INNER JOIN sys.schemas AS schemaParent
                    ON tabParent.schema_id = schemaParent.schema_id
                INNER JOIN sys.columns AS parentCol
                    ON fkc.parent_object_id = parentCol.object_id
                   AND fkc.parent_column_id = parentCol.column_id
                WHERE fk.referenced_object_id = OBJECT_ID(:full_name)
            """)
            result = conn.execute(sql, {"full_name": full_obj_name})
            refs = []
            for row in result.fetchall():
                parent_schema, parent_table, parent_column, constraint_name = row
                refs.append({
                    "parent_schema": parent_schema,
                    "parent_table": parent_table,
                    "parent_column": parent_column,
                    "constraint_name": constraint_name
                })
            return refs

        def fetch_first_n_rows(schema_name: str, table_name: str, n: int = sample_row_count):
            """Fetch sample data from the table"""
            qry = text(f"SELECT TOP {n} * FROM [{schema_name}].[{table_name}]")
            try:
                result = conn.execute(qry)
                columns = result.keys()
                rows = result.fetchall()

                row_dicts = []
                for row in rows:
                    row_data = {}
                    for idx, col_name in enumerate(columns):
                        val = row[idx]
                        row_data[col_name] = val
                    row_dicts.append(row_data)

                return row_dicts
            except SQLAlchemyError as e:
                print(f"⚠️  Warning: Could not fetch data from {schema_name}.{table_name}: {e}")
                return []

        # Process the single table
        fq_table_name = f"{schema_name}.{table_name}"
        print(f"📋 Processing table: {fq_table_name}")
        
        # 1) Column metadata
        columns_meta = get_columns_for_table(schema_name, table_name)
        
        # 2) Primary key columns
        pk_columns = get_primary_key_columns(schema_name, table_name)
        
        # 3) Foreign key details for this table
        fk_defs = get_foreign_keys_for_table(schema_name, table_name)
        
        # 4) Tables that reference this table
        incoming_refs = get_tables_referencing_current(schema_name, table_name)
        
        # 5) Sample data (first N rows)
        sample_rows = fetch_first_n_rows(schema_name, table_name, n=sample_row_count)
        
        # --- Build "columns" list with flags ---
        columns_list = []
        for col in columns_meta:
            col_name = col["name"]
            is_primary = col_name in pk_columns
            fk_def = next((fk for fk in fk_defs if fk["fk_column"] == col_name), None)
            is_foreign = fk_def is not None
            col_entry = {
                "name": col_name,
                "type": col["type"],
                "is_primary": is_primary,
                "is_foreign": is_foreign,
                "is_required": not col["is_nullable"],
                "max_length": col["max_length"]
            }
            if is_foreign:
                col_entry["references"] = {
                    "table": f"{fk_def['referenced_schema']}.{fk_def['referenced_table']}",
                    "column": fk_def["referenced_column"],
                    "constraint": fk_def["constraint_name"]
                }
            columns_list.append(col_entry)
        
        # --- Build "relationships" list ---
        relationships = []
        
        # a) Self-referential (if any FK references the same table within same schema)
        for fk in fk_defs:
            if (fk["referenced_table"] == table_name and 
                fk["referenced_schema"] == schema_name):
                relationships.append({
                    "related_table": f"{fk['referenced_schema']}.{fk['referenced_table']}",
                    "type": "self_referential",
                    "via_column": fk["fk_column"],
                    "via_related": fk["referenced_column"]
                })
        
        # b) Many-to-one: current table references other tables
        for fk in fk_defs:
            if not (fk["referenced_table"] == table_name and 
                   fk["referenced_schema"] == schema_name):  # Skip self-referential
                relationships.append({
                    "related_table": f"{fk['referenced_schema']}.{fk['referenced_table']}",
                    "type": "many_to_one",
                    "via_column": fk["fk_column"],
                    "via_related": fk["referenced_column"]
                })
        
        # c) One-to-many: other tables referencing this table
        for ref in incoming_refs:
            if not (ref["parent_table"] == table_name and 
                   ref["parent_schema"] == schema_name):  # Skip self-referential (already handled)
                relationships.append({
                    "related_table": f"{ref['parent_schema']}.{ref['parent_table']}",
                    "type": "one_to_many",
                    "via_column": ref["parent_column"],
                    "via_related": pk_columns[0] if pk_columns else None
                })
        
        # --- Assemble table info dict ---
        table_info = {
            "schema": schema_name,
            "table_name": table_name,
            "full_name": fq_table_name,
            "primary_keys": pk_columns,
            "columns": columns_list,
            "relationships": relationships,
            "sample_data": sample_rows,
            "row_count_sample": len(sample_rows),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"✅ Successfully generated schema for {fq_table_name}")
        return table_info
        
    except Exception as e:
        print(f"❌ Error generating schema for {schema_name}.{table_name}: {e}")
        return None
        
    finally:
        # Clean up database connection
        try:
            conn.close()
            engine.dispose()
            print(f"🔌 Database connection closed for {schema_name}.{table_name}")
        except Exception:
            pass

# -------------------------------------------------
# MAIN FUNCTION (ORIGINAL - KEPT FOR BACKWARD COMPATIBILITY)
# -------------------------------------------------
def generate_schema_and_data(database_url: str, output_file: str = None, sample_row_count: int = 50):
    """
    Generate schema and sample data for all tables in the database.
    
    Args:
        database_url (str): The database connection URL
        output_file (str, optional): Output file path. If None, uses default name
        sample_row_count (int, optional): Number of sample rows to fetch per table. Default 50
    
    Returns:
        str: Path to the generated output file
    """
    if output_file is None:
        output_file = "all_tables_schema_and_data.json"
    
    # -------------------------------------------------
    # PROGRESS TRACKING HELPERS
    # -------------------------------------------------
    def print_progress(current, total, table_name="", status="Processing"):
        """Print progress bar with current table info"""
        percent = (current / total) * 100
        bar_length = 40
        filled_length = int(bar_length * current / total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"\r{status}: [{bar}] {percent:.1f}% ({current}/{total}) - {table_name}", end='', flush=True)

    def print_step(step_name, table_name=""):
        """Print current processing step"""
        if table_name:
            print(f"  📋 {step_name} for {table_name}...")
        else:
            print(f"🔄 {step_name}...")

    # -------------------------------------------------
    # CONNECT TO DATABASE
    # -------------------------------------------------
    print("🔌 Connecting to database...")
    try:
        engine = create_engine(database_url, fast_executemany=True)
        conn = engine.connect()
        print("✅ Database connection successful!")
    except SQLAlchemyError as e:
        print("❌ Failed to connect to the database. Please verify your DATABASE_URL.")
        print("Error:", e)
        raise

    # -------------------------------------------------
    # HELPER FUNCTIONS / QUERIES
    # -------------------------------------------------

    def get_all_tables():
        """
        Returns a list of tuples: (schema_name, table_name)
        Only includes user-defined base tables.
        """
        sql = text("""
            SELECT TABLE_SCHEMA, TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE'
              AND OBJECTPROPERTY(
                    OBJECT_ID(QUOTENAME(TABLE_SCHEMA) + '.' + QUOTENAME(TABLE_NAME)),
                    'IsMSShipped'
                  ) = 0
            ORDER BY TABLE_SCHEMA, TABLE_NAME
        """)
        result = conn.execute(sql)
        return result.fetchall()  # List of (schema_name, table_name)


    def get_columns_for_table(schema_name: str, table_name: str):
        """
        Returns a list of dicts for each column of the given table:
        [{
            "name": ...,
            "type": ...,
            "is_nullable": ...,
            "max_length": ...
        }, ...]
        """
        sql = text("""
            SELECT 
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = :schema
              AND TABLE_NAME   = :table
            ORDER BY ORDINAL_POSITION
        """)
        result = conn.execute(sql, {"schema": schema_name, "table": table_name})
        columns = []
        for row in result.fetchall():
            col_name, data_type, is_nullable, max_len = row
            columns.append({
                "name": col_name,
                "type": data_type.upper(),
                "is_nullable": (is_nullable == "YES"),
                "max_length": max_len
            })
        return columns


    def get_primary_key_columns(schema_name: str, table_name: str):
        """
        Returns a list of column names that belong to the PRIMARY KEY
        of the given table.
        """
        full_obj_name = f"{schema_name}.{table_name}"
        sql = text("""
            SELECT c.name
            FROM sys.indexes i
            INNER JOIN sys.index_columns ic
                ON i.object_id = ic.object_id
               AND i.index_id  = ic.index_id
            INNER JOIN sys.columns c
                ON c.object_id = ic.object_id
               AND c.column_id = ic.column_id
            WHERE i.is_primary_key = 1
              AND i.object_id = OBJECT_ID(:full_name)
            ORDER BY ic.key_ordinal
        """)
        result = conn.execute(sql, {"full_name": full_obj_name})
        return [row[0] for row in result.fetchall()]


    def get_foreign_keys_for_table(schema_name: str, table_name: str):
        """
        Returns a list of foreign-key definitions for the given table.
        Each item is a dict:
          {
            "constraint_name": ...,
            "fk_column": ...,
            "referenced_schema": ...,
            "referenced_table": ...,
            "referenced_column": ...
          }
        """
        sql = text("""
            SELECT 
                fk.name                            AS constraint_name,
                parentCol.name                     AS fk_column,
                schemaRef.name                     AS referenced_schema,
                tabRef.name                        AS referenced_table,
                refCol.name                        AS referenced_column
            FROM sys.foreign_keys AS fk
            INNER JOIN sys.foreign_key_columns AS fkc
                ON fk.object_id = fkc.constraint_object_id
            INNER JOIN sys.tables AS tabParent
                ON fk.parent_object_id = tabParent.object_id
            INNER JOIN sys.schemas AS schemaParent
                ON tabParent.schema_id = schemaParent.schema_id
            INNER JOIN sys.columns AS parentCol
                ON fkc.parent_object_id = parentCol.object_id
               AND fkc.parent_column_id = parentCol.column_id
            INNER JOIN sys.tables AS tabRef
                ON fk.referenced_object_id = tabRef.object_id
            INNER JOIN sys.schemas AS schemaRef
                ON tabRef.schema_id = schemaRef.schema_id
            INNER JOIN sys.columns AS refCol
                ON fkc.referenced_object_id = refCol.object_id
               AND fkc.referenced_column_id = refCol.column_id
            WHERE schemaParent.name = :schema
              AND tabParent.name   = :table
        """)
        result = conn.execute(sql, {"schema": schema_name, "table": table_name})
        fk_list = []
        for row in result.fetchall():
            constraint_name, fk_col, ref_schema, ref_table, ref_col = row
            fk_list.append({
                "constraint_name": constraint_name,
                "fk_column": fk_col,
                "referenced_schema": ref_schema,
                "referenced_table": ref_table,
                "referenced_column": ref_col
            })
        return fk_list


    def get_tables_referencing_current(schema_name: str, table_name: str):
        """
        Returns a list of dicts for tables that have FKs pointing to the given table.
        Each dict:
          {
            "parent_schema": ...,
            "parent_table": ...,
            "parent_column": ...,
            "constraint_name": ...
          }
        """
        full_obj_name = f"{schema_name}.{table_name}"
        sql = text("""
            SELECT 
                schemaParent.name AS parent_schema,
                tabParent.name    AS parent_table,
                parentCol.name    AS parent_column,
                fk.name           AS constraint_name
            FROM sys.foreign_keys AS fk
            INNER JOIN sys.foreign_key_columns AS fkc
                ON fk.object_id = fkc.constraint_object_id
            INNER JOIN sys.tables AS tabParent
                ON fk.parent_object_id = tabParent.object_id
            INNER JOIN sys.schemas AS schemaParent
                ON tabParent.schema_id = schemaParent.schema_id
            INNER JOIN sys.columns AS parentCol
                ON fkc.parent_object_id = parentCol.object_id
               AND fkc.parent_column_id = parentCol.column_id
            WHERE fk.referenced_object_id = OBJECT_ID(:full_name)
        """)
        result = conn.execute(sql, {"full_name": full_obj_name})
        refs = []
        for row in result.fetchall():
            parent_schema, parent_table, parent_column, constraint_name = row
            refs.append({
                "parent_schema": parent_schema,
                "parent_table": parent_table,
                "parent_column": parent_column,
                "constraint_name": constraint_name
            })
        return refs


    def fetch_first_n_rows(schema_name: str, table_name: str, n: int = sample_row_count):
        """
        Returns a list of dicts representing the first n rows.
        Converts each value to a JSON-serializable form.
        """
        qry = text(f"SELECT TOP {n} * FROM [{schema_name}].[{table_name}]")
        try:
            result = conn.execute(qry)
            columns = result.keys()
            rows = result.fetchall()

            row_dicts = []
            for row in rows:
                row_data = {}
                for idx, col_name in enumerate(columns):
                    val = row[idx]
                    # Let the custom JSON encoder handle type conversion
                    row_data[col_name] = val
                row_dicts.append(row_data)

            return row_dicts
        except SQLAlchemyError as e:
            print(f"\n⚠️  Warning: Could not fetch data from {schema_name}.{table_name}: {e}")
            return []


    def process_single_table(schema_name: str, table_name: str, table_idx: int, total_tables: int):
        """Process a single table and return its info dict"""
        fq_table_name = f"{schema_name}.{table_name}"
        
        try:
            # Update progress
            print_progress(table_idx, total_tables, fq_table_name, "Processing")
            
            # 1) Column metadata
            columns_meta = get_columns_for_table(schema_name, table_name)
            
            # 2) Primary key columns
            pk_columns = get_primary_key_columns(schema_name, table_name)
            
            # 3) Foreign key details for this table
            fk_defs = get_foreign_keys_for_table(schema_name, table_name)
            
            # 4) Tables that reference this table
            incoming_refs = get_tables_referencing_current(schema_name, table_name)
            
            # 5) Sample data (first N rows)
            sample_rows = fetch_first_n_rows(schema_name, table_name, n=sample_row_count)
            
            # --- Build "columns" list with flags ---
            columns_list = []
            for col in columns_meta:
                col_name = col["name"]
                is_primary = col_name in pk_columns
                fk_def = next((fk for fk in fk_defs if fk["fk_column"] == col_name), None)
                is_foreign = fk_def is not None
                col_entry = {
                    "name": col_name,
                    "type": col["type"],
                    "is_primary": is_primary,
                    "is_foreign": is_foreign,
                    "is_required": not col["is_nullable"],
                    "max_length": col["max_length"]
                }
                if is_foreign:
                    col_entry["references"] = {
                        "table": f"{fk_def['referenced_schema']}.{fk_def['referenced_table']}",
                        "column": fk_def["referenced_column"],
                        "constraint": fk_def["constraint_name"]
                    }
                columns_list.append(col_entry)
            
            # --- Build "relationships" list ---
            relationships = []
            
            # a) Self-referential (if any FK references the same table within same schema)
            for fk in fk_defs:
                if (fk["referenced_table"] == table_name and 
                    fk["referenced_schema"] == schema_name):
                    relationships.append({
                        "related_table": f"{fk['referenced_schema']}.{fk['referenced_table']}",
                        "type": "self_referential",
                        "via_column": fk["fk_column"],
                        "via_related": fk["referenced_column"]
                    })
            
            # b) Many-to-one: current table references other tables
            for fk in fk_defs:
                if not (fk["referenced_table"] == table_name and 
                       fk["referenced_schema"] == schema_name):  # Skip self-referential
                    relationships.append({
                        "related_table": f"{fk['referenced_schema']}.{fk['referenced_table']}",
                        "type": "many_to_one",
                        "via_column": fk["fk_column"],
                        "via_related": fk["referenced_column"]
                    })
            
            # c) One-to-many: other tables referencing this table
            for ref in incoming_refs:
                if not (ref["parent_table"] == table_name and 
                       ref["parent_schema"] == schema_name):  # Skip self-referential (already handled)
                    relationships.append({
                        "related_table": f"{ref['parent_schema']}.{ref['parent_table']}",
                        "type": "one_to_many",
                        "via_column": ref["parent_column"],
                        "via_related": pk_columns[0] if pk_columns else None
                    })
            
            # --- Assemble per-table dict ---
            table_info = {
                "schema": schema_name,
                "table_name": table_name,
                "full_name": fq_table_name,
                "primary_keys": pk_columns,
                "columns": columns_list,
                "relationships": relationships,
                "sample_data": sample_rows,
                "row_count_sample": len(sample_rows)
            }
            
            return table_info
            
        except Exception as e:
            print(f"\n❌ Error processing table {fq_table_name}: {e}")
            return None

    try:
        # -------------------------------------------------
        # BUILD SCHEMA + DATA FOR ALL TABLES
        # -------------------------------------------------
        print_step("Discovering tables")
        all_tables = get_all_tables()
        total_tables = len(all_tables)

        if total_tables == 0:
            print("⚠️  No user-defined tables found in the database.")
            return output_file

        print(f"📊 Found {total_tables} tables to process")
        print("-" * 60)

        all_tables_info = []
        processed_count = 0
        failed_count = 0
        start_time = time.time()

        for idx, (schema_name, table_name) in enumerate(all_tables, 1):
            table_info = process_single_table(schema_name, table_name, idx, total_tables)
            
            if table_info:
                all_tables_info.append(table_info)
                processed_count += 1
            else:
                failed_count += 1

        # Final progress update
        print_progress(total_tables, total_tables, "Complete!", "Finished")
        print()  # New line after progress bar

        elapsed_time = time.time() - start_time
        print("-" * 60)
        print(f"📈 Processing Summary:")
        print(f"   ✅ Successfully processed: {processed_count} tables")
        if failed_count > 0:
            print(f"   ❌ Failed to process: {failed_count} tables")
        print(f"   ⏱️  Total time: {elapsed_time:.2f} seconds")
        print(f"   🔄 Average time per table: {elapsed_time/total_tables:.2f} seconds")

        # -------------------------------------------------
        # WRITE ALL TABLES INFO TO JSON
        # -------------------------------------------------
        print_step("Writing results to JSON file")
        try:
            with open(output_file, "w", encoding="utf-8") as fp:
                json.dump({
                    "metadata": {
                        "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "total_tables": total_tables,
                        "processed_tables": processed_count,
                        "failed_tables": failed_count,
                        "sample_row_count": sample_row_count,
                        "database_url": database_url.split('@')[0] + "@[REDACTED]"  # Hide credentials
                    },
                    "tables": all_tables_info
                }, fp, indent=4, ensure_ascii=False, cls=SQLJSONEncoder)

            print(f"✅ All tables schema + data written to '{output_file}'")
            print(f"📁 File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
            
        except Exception as e:
            print(f"❌ Failed to write output file: {e}")
            raise

        return output_file

    finally:
        # -------------------------------------------------
        # CLEAN UP
        # -------------------------------------------------
        print_step("Cleaning up database connection")
        conn.close()
        engine.dispose()
        print("🎉 Process completed successfully!")

# -------------------------------------------------
# PROGRESS TRACKING HELPERS (for backward compatibility)
# -------------------------------------------------
def print_progress(current, total, table_name="", status="Processing"):
    """Print progress bar with current table info"""
    percent = (current / total) * 100
    bar_length = 40
    filled_length = int(bar_length * current / total)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    print(f"\r{status}: [{bar}] {percent:.1f}% ({current}/{total}) - {table_name}", end='', flush=True)

def print_step(step_name, table_name=""):
    """Print current processing step"""
    if table_name:
        print(f"  📋 {step_name} for {table_name}...")
    else:
        print(f"🔄 {step_name}...")

# -------------------------------------------------
# MAIN EXECUTION (for backward compatibility)
# -------------------------------------------------
if __name__ == "__main__":
    # Use the new function with the default configuration
    try:
        output_file = generate_schema_and_data(DATABASE_URL, OUTPUT_FILE, SAMPLE_ROW_COUNT)
        print(f"🎉 Schema generation completed! Output file: {output_file}")
    except Exception as e:
        print(f"❌ Schema generation failed: {e}")
        exit(1)