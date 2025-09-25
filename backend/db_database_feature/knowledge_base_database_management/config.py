import os
from pathlib import Path

# Base directory for file storage
STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)

# Database configuration
# MSSQL Configuration (commented out)
# DB_CONFIG = {
#     "SERVER": os.getenv("DB_SERVER_mssql"),
#     "DATABASE": os.getenv("DB_NAME_mssql"),
#     "USERNAME": os.getenv("DB_USER_mssql"),
#     "PASSWORD": os.getenv("DB_PASSWORD_mssql")
# }

# # PostgreSQL Configuration
# DB_CONFIG = {
#     "HOST": "176.9.16.194",
#     "PORT": 5432,
#     "DATABASE": "postgres",  # Default database name, can be changed as needed
#     "USERNAME": "postgres",
#     "PASSWORD": "postgres"  # Changed from "postgre" to "postgres" (more common default)
# }


# PostgreSQL Configuration for raihan
# DB_CONFIG = {
#     "HOST": "localhost",
#     "PORT": 5432,
#     "DATABASE": "postgres",  # Using default database for now
#     "USERNAME": "postgres",
#     "PASSWORD": "postgres"  # Changed from "postgre" to "postgres" (more common default)
# }



# PostgreSQL Configuration for nabil
DB_CONFIG = {
    "HOST": "localhost",
    "PORT": 5433,
    "DATABASE": "main_db",  # Default database name, can be changed as needed
    "USERNAME": "postgres",
    "PASSWORD": "1234"  # Changed from "postgre" to "postgres" (more common default)
}
