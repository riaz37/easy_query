import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.getcwd())

# from data_sources.mssql.mssql_agent import router as mssql_router
# from data_sources.mssql.mssql_agent2 import router as mssql_router2
from data_sources.mssql.mssql_agent3 import router as mssql_router3
from data_sources.file_data.router import router as file_router
from db_database_feature.knowledge_base_database_management.router import router as router_company_management
from db_manager.data_base_config import router as router_data_base_config
from db_manager.mssql_config import router as router_mssql_config
from db_manager.utilites.semi_structured_To_table_db import router as excel_to_db_router
from db_manager.utilites.new_table_creation import router as new_table_router
from Report_generator.utilites.graph_Generator import router as graph_router
from Report_generator.utilites.report_agent import router as report_router
# Import voice agent router with error handling
try:
    from voice_agent.voice_agent_router import router as voice_agent_router
    VOICE_AGENT_AVAILABLE = True
    print("✅ Voice Agent Router imported successfully")
except Exception as e:
    print(f"⚠️ Warning: Voice Agent Router import failed: {e}")
    print("⚠️ Voice agent functionality will not be available")
    print("⚠️ Set GOOGLE_API_KEY environment variable to enable voice agent")
    VOICE_AGENT_AVAILABLE = False
    voice_agent_router = None

app = FastAPI(title="Knowledge base API", version="1.0")

# Create storage directories if they don't exist
storage_dir = os.path.join(BASE_DIR, "storage")
graphs_dir = os.path.join(storage_dir, "graphs")
images_dir = os.path.join(graphs_dir, "images")
html_dir = os.path.join(graphs_dir, "html")

os.makedirs(storage_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(html_dir, exist_ok=True)

# Middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip compression middleware for better performance
app.add_middleware(GZipMiddleware, minimum_size=1000)

# app.include_router(mssql_router, prefix="/mssql_old", tags=["MSSQL Agent"])
# app.include_router(file_router, prefix="/files", tags=["File Agent"])
app.include_router(mssql_router3, prefix="/mssql", tags=["MSSQL Agent Scalable"])
app.include_router(file_router, prefix="/files", tags=["File Agent"])
app.include_router(router_data_base_config, tags=["Data Base Config"])
app.include_router(router_mssql_config, prefix="/mssql-config")
app.include_router(excel_to_db_router, prefix="/excel-to-db", tags=["Excel to Database"])
app.include_router(new_table_router, prefix="/new-table", tags=["New Table Management"])
app.include_router(graph_router, prefix="/graph", tags=["Graph Generator"])
app.include_router(report_router, prefix="/reports", tags=["Report Generation"])
app.include_router(router_company_management, prefix="/Auth")

# Include voice agent router if available
if VOICE_AGENT_AVAILABLE and voice_agent_router:
    print("🔧 Including Voice Agent Router in main app...")
    app.include_router(voice_agent_router, prefix="/voice", tags=["Voice Agent"])
    print("🔧 Voice Agent Router included successfully with prefix /voice")
else:
    print("⚠️ Voice Agent Router not available - skipping inclusion")
    print("⚠️ Set GOOGLE_API_KEY environment variable to enable voice agent")

# Mount static file directories
app.mount("/uploads", StaticFiles(directory="./uploads/"), name="uploads")
app.mount("/storage/graphs/images", StaticFiles(directory=images_dir), name="graph_images")
app.mount("/storage/graphs/html", StaticFiles(directory=html_dir), name="graph_html")

if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Check if --http flag is provided for development
    if "--http" in sys.argv:
        print("🚀 Starting server in HTTP mode for development...")
        uvicorn.run("main:app", host="0.0.0.0", port=8200, reload=False)
    else:
        print("🔒 Starting server in HTTPS mode...")
        uvicorn.run("main:app", host="0.0.0.0", port=8200, reload=False, ssl_certfile="cert.pem", ssl_keyfile="key.pem")
    