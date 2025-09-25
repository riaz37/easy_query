# Getting Started with Easy Query

This guide will help you set up and run the Easy Query Knowledge Base project on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.11+** (recommended 3.11 or higher)
- **Node.js 18+** (recommended 18.x or higher)
- **pnpm** (preferred package manager)
- **PostgreSQL 13+** or **Microsoft SQL Server 2019+**
- **Redis** (optional, for caching)
- **API Keys**:
  - Google API Key (for Gemini models)
  - Groq API Key (for Llama models)
  - OpenAI API Key (optional, for voice agent)

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd easy_query
```

### 2. Backend Setup

Navigate to the backend directory:
```bash
cd backend
```

Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your actual configuration
```

Start the backend server:
```bash
# For development (HTTP)
python main.py --http

# For production (HTTPS)
python main.py
```

The backend will start on `https://localhost:8200` (HTTPS) or `http://localhost:8200` (HTTP)

### 3. Frontend Setup

In a new terminal, navigate to the frontend directory:
```bash
cd frontend
```

Install frontend dependencies using pnpm:
```bash
pnpm install
```

Start the development server:
```bash
pnpm dev
```

The frontend will start on `http://localhost:3000`

## Environment Configuration

Create the `backend/.env` file with your actual configuration:

```env
# ===========================================
# AI MODEL CONFIGURATION
# ===========================================
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Model Selection
GOOGLE_GEMINI_NAME=gemini-1.5-pro
GOOGLE_GEMINI_EMBEDDING_NAME=gemini-embedding-exp-03-07

# ===========================================
# DATABASE CONFIGURATION
# ===========================================
# PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=your_database_name

# Microsoft SQL Server (Alternative)
MSSQL_SERVER=your_server
MSSQL_DATABASE=your_database
MSSQL_USERNAME=your_username
MSSQL_PASSWORD=your_password

# ===========================================
# REDIS CACHING (OPTIONAL)
# ===========================================
REDIS_CLOUD=true
REDIS_HOST=your_redis_host
REDIS_PORT=6379
REDIS_USER=default
REDIS_PASSWORD=your_redis_password
REDIS_DB=0
CACHE_TTL=10

# ===========================================
# SERVER CONFIGURATION
# ===========================================
ENVIRONMENT=development
DEV_BACKEND_URL=https://localhost:8200
PROD_BACKEND_URL=https://your-production-url:8200

# Authentication Secret (change this in production)
SECRET_KEY=your_secret_key_here_change_this_in_production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Voice Agent Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8002
LOG_LEVEL=INFO
```

## First Time Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Sign up for a new account
3. Log in with your credentials
4. Configure your database connection in the settings
5. Start querying your database using natural language!

## Common Issues

### SSL Certificate Error
If you encounter SSL certificate issues, you can start the backend in HTTP mode:
```bash
python main.py --http
```

### Database Connection Issues
Ensure your database is running and the connection details in `.env` are correct.

### Missing API Keys
The application requires at least one AI API key to function. You can get these from:
- Google AI Studio (for Gemini)
- Groq Cloud (for Llama models)

## Next Steps

After successful setup:
1. Explore the dashboard and its various modules
2. Configure business rules for your database
3. Try different AI models for query generation
4. Set up user roles and permissions
5. Experiment with the voice agent functionality

## Documentation

For more detailed information:
- [Main README](README.md) - Complete project documentation
- [Project Structure](PROJECT_STRUCTURE.md) - Detailed code organization
- API documentation available at `https://localhost:8200/docs` when backend is running