from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
# import urllib  # Commented out - not needed for PostgreSQL
from dotenv import load_dotenv
import os
from .config import DB_CONFIG

load_dotenv()

# MSSQL Configuration (commented out)
# params = urllib.parse.quote_plus(
#     f"DRIVER={{ODBC Driver 17 for SQL Server}};"
#     f"SERVER={os.getenv('DB_SERVER_mssql')};"
#     f"DATABASE={os.getenv('DB_NAME_mssql')};"
#     f"UID={os.getenv('DB_USER_mssql')};"
#     f"PWD={os.getenv('DB_PASSWORD_mssql')};"
#     "Encrypt=yes;TrustServerCertificate=yes;"
# )
# engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}", echo=True)

# PostgreSQL Configuration
DATABASE_URL = f"postgresql://{DB_CONFIG['USERNAME']}:{DB_CONFIG['PASSWORD']}@{DB_CONFIG['HOST']}:{DB_CONFIG['PORT']}/{DB_CONFIG['DATABASE']}"
engine = create_engine(
    DATABASE_URL, 
    echo=True,
    connect_args={
        "connect_timeout": 10,
        "application_name": "knowledge_base_backend"
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()