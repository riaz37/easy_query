# Easy Query - AI-Powered Knowledge Base Solution

Easy Query is a comprehensive AI-powered knowledge base solution that enables natural language interaction with databases and documents. The system combines advanced AI models with intelligent document processing to provide a unified platform for data querying, analysis, and reporting.

## 🌟 Features

### Core Functionality
- **Natural Language to SQL**: Convert natural language questions into executable SQL queries using advanced AI models
- **Multi-Model AI Support**: Integration with Google Gemini, Groq (Llama), OpenAI, and other cutting-edge AI models
- **Database Agnostic**: Supports Microsoft SQL Server, PostgreSQL, and other enterprise databases
- **Advanced Document Processing**: Intelligent processing of PDFs, Excel files, Word documents, and more
- **Smart File Routing**: Automatic routing of files to appropriate processing pipelines (structured vs unstructured)

### User Experience
- **Interactive Dashboard**: Modern web interface with 3D visualizations and drag-and-drop components
- **Voice Agent Integration**: Real-time voice-controlled database querying with ultra-low latency
- **Conversation Memory**: Maintain context across multiple queries and sessions
- **Real-time Processing**: Background task processing with progress tracking

### Enterprise Features
- **Role-Based Access Control**: Comprehensive RBAC system with fine-grained permissions
- **Company Management**: Hierarchical company structure with multi-tenant support
- **Report Generation**: Create interactive visual reports, charts, and HTML exports
- **Redis Caching**: High-performance caching with cloud Redis support
- **HTTPS Security**: Secure communication with SSL/TLS encryption

## 🏗️ Architecture

The system is built with modern, scalable technologies and follows microservices architecture principles:

### Backend (FastAPI + Python 3.11+)
- **FastAPI 0.115.11**: High-performance async web framework with automatic API documentation
- **AI Integration**: Google Gemini, Groq (Llama), OpenAI with intelligent model selection
- **Database Layer**: SQLAlchemy ORM with PostgreSQL and Microsoft SQL Server support
- **Authentication & Security**: JWT-based authentication with comprehensive RBAC system
- **Caching & Performance**: Redis caching with cloud support and connection pooling
- **Voice Processing**: Pipecat framework for real-time voice interactions
- **Document Processing**: Advanced NLP pipelines for structured and unstructured data

### Frontend (Next.js 15 + TypeScript)
- **Next.js 15.3.3**: Latest React framework with App Router and Turbopack
- **React 19**: Latest React with concurrent features and improved performance
- **TypeScript**: Full type safety with strict configuration
- **Tailwind CSS v4**: Utility-first CSS with shadcn/ui component library
- **Three.js + React Three Fiber**: Advanced 3D visualizations and interactive components
- **State Management**: Zustand for lightweight, scalable state management
- **Data Visualization**: Recharts for interactive charts and graphs

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** (recommended 3.11 or higher)
- **Node.js 18+** (recommended 18.x or higher)
- **pnpm** (preferred package manager)
- **PostgreSQL 13+** or **Microsoft SQL Server 2019+**
- **Redis** (optional, for caching)
- **API Keys**:
  - Google API Key (for Gemini models)
  - Groq API Key (for Llama models)
  - OpenAI API Key (optional, for voice agent)

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create .env file with the following variables:
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional for voice agent

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=your_database

# Redis Configuration (Optional)
REDIS_CLOUD=true
REDIS_HOST=your_redis_host
REDIS_PORT=your_redis_port
REDIS_PASSWORD=your_redis_password

# Server Configuration
ENVIRONMENT=development
DEV_BACKEND_URL=https://localhost:8200
PROD_BACKEND_URL=https://your-production-url:8200
```

5. Start the backend server:
```bash
# For development (HTTP)
python main.py --http

# For production (HTTPS)
python main.py
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies using pnpm:
```bash
pnpm install
```

3. Start the development server:
```bash
pnpm dev
```

4. Open your browser to `http://localhost:3000`

### Docker Setup (Optional)

#### Backend Docker
```bash
cd backend
docker build -t easy-query-backend .
docker run -p 8200:8200 --env-file .env easy-query-backend
```

#### Frontend Docker
```bash
cd frontend
docker build -t easy-query-frontend .
docker run -p 3000:3000 easy-query-frontend
```

## 📊 Core Modules

### 1. Database Query Agent (`/mssql`)
Advanced SQL query generation and execution:
- **Natural Language Processing**: Multi-model AI converts questions to SQL
- **Business Rules Integration**: Company-specific query constraints and validation
- **Query Optimization**: Safe execution with SQL injection protection
- **Multiple Model Support**: Google Gemini, Groq, OpenAI with intelligent fallbacks
- **Conversation Memory**: Context-aware query building across sessions

### 2. File Processing System (`/files`)
Intelligent document and data processing:
- **Smart File Routing**: Automatic pipeline selection (structured vs unstructured)
- **Advanced NLP Pipelines**: Embedding generation, intent mapping, and semantic analysis
- **Multi-format Support**: PDFs, Excel, Word, CSV, and more
- **Background Processing**: Async task processing with progress tracking
- **Database Integration**: Automatic table creation and data ingestion

### 3. Authentication & RBAC (`/Auth`)
Enterprise-grade security system:
- **User Management**: Registration, login, profile management
- **Role-Based Access Control**: Hierarchical permissions and company structure
- **JWT Authentication**: Secure token-based sessions with revocation
- **Multi-tenant Support**: Company isolation and data separation

### 4. Report Generation (`/reports`, `/graph`)
Advanced analytics and visualization:
- **Interactive Charts**: Dynamic charts using Recharts and Three.js
- **HTML Report Generation**: Rich, exportable HTML reports
- **Real-time Processing**: Background report generation with status tracking
- **Export Capabilities**: Multiple format support (PDF, Excel, HTML)

### 5. Voice Agent (`/voice`)
Real-time voice interaction system:
- **Ultra-low Latency**: Pipecat framework for real-time voice processing
- **Multi-modal AI**: Speech-to-text, natural language processing, text-to-speech
- **WebSocket Communication**: Real-time bidirectional communication
- **Tool Integration**: Voice-controlled database queries and file operations

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory with the following variables:

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
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=your_database

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
REDIS_PORT=your_redis_port
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

# Authentication
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Voice Agent Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8002
LOG_LEVEL=INFO
```

## 📡 API Endpoints

### Database Query (`/mssql`)
- `POST /mssql/query` - Execute natural language to SQL queries
- `POST /mssql/reload-db` - Reload database schema and metadata
- `GET /mssql/conversation-history/{user_id}` - Retrieve user query history
- `POST /mssql/clear-history/{user_id}` - Clear user conversation history
- `GET /mssql/schema` - Get database schema information

### File Processing (`/files`)
- `POST /files/smart_file_system` - Smart file upload with automatic pipeline routing
- `POST /files/smart_file_system_backend` - Advanced file processing with custom parameters
- `POST /files/unstructured_file_system` - Process unstructured documents (PDFs, Word, etc.)
- `POST /files/semi_structured_file_system` - Process structured documents (Excel, CSV, etc.)
- `GET /files/task-status/{task_id}` - Check processing task status
- `GET /files/list` - List uploaded files and processing results

### Authentication & RBAC (`/Auth`)
- `POST /Auth/signup` - User registration with validation
- `POST /Auth/login` - User authentication with JWT tokens
- `POST /Auth/logout` - Secure user logout with token revocation
- `POST /Auth/change-password` - Password change functionality
- `GET /Auth/profile` - Get user profile information
- `POST /Auth/companies` - Create company hierarchy
- `GET /Auth/companies` - List companies and permissions

### Report Generation (`/reports`, `/graph`)
- `POST /reports/generate` - Generate comprehensive reports from queries
- `POST /graph/generate` - Create interactive charts and visualizations
- `GET /reports/{report_id}` - Retrieve generated report
- `GET /reports/list` - List available reports
- `POST /reports/export` - Export reports in various formats

### Voice Agent (`/voice`)
- `GET /voice/health` - Voice agent health check
- `POST /voice/connect` - Get WebSocket connection URL
- `WS /voice/ws` - Real-time voice conversation WebSocket
- `WS /voice/ws/tools` - Voice-controlled tool commands WebSocket
- `GET /voice/test-database-search` - Test database search functionality

### Database Configuration (`/mssql-config`, `/excel-to-db`)
- `POST /mssql-config/` - Configure MSSQL database connections
- `GET /mssql-config/` - List configured databases
- `POST /excel-to-db/convert` - Convert Excel files to database tables
- `POST /new-table/create` - Create new database tables

## 🧪 Testing

### Backend Testing
```bash
cd backend
# Run all tests
python -m pytest

# Run specific test modules
python -m pytest tests/test_auth.py
python -m pytest tests/test_mssql_agent.py

# Run with coverage
python -m pytest --cov=backend tests/
```

### Frontend Testing
```bash
cd frontend
# Run unit tests
pnpm test

# Run tests in watch mode
pnpm test:watch

# Run tests with coverage
pnpm test:coverage

# Run linting
pnpm lint

# Type checking
pnpm type-check
```

### Integration Testing
```bash
# Test API endpoints
curl -X POST http://localhost:8200/mssql/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me all users", "user_id": "test_user"}'

# Test voice agent health
curl http://localhost:8200/voice/health

# Test file upload
curl -X POST http://localhost:8200/files/smart_file_system \
  -F "files=@test_document.pdf" \
  -F "file_descriptions=Test document"
```

## 📈 Performance Optimization

### Backend Optimizations
- **Redis Caching**: Intelligent caching of queries, results, and embeddings
- **Connection Pooling**: Efficient database connection management with SQLAlchemy
- **Async Processing**: Non-blocking operations for better throughput
- **Background Tasks**: Heavy processing moved to background workers
- **GZip Compression**: Automatic response compression for faster transfers
- **Query Optimization**: AI-powered query optimization and caching

### Frontend Optimizations
- **Turbopack**: Fast bundling and hot reloading in development
- **Code Splitting**: Automatic route-based code splitting
- **Image Optimization**: Next.js automatic image optimization
- **Static Generation**: Pre-rendered pages for better performance
- **Three.js Optimization**: Efficient 3D rendering with React Three Fiber

## 🔐 Security

### Authentication & Authorization
- **JWT Authentication**: Secure token-based authentication with configurable expiration
- **Role-Based Access Control**: Hierarchical permissions with company-level isolation
- **Token Revocation**: Secure logout with token blacklisting
- **Password Security**: Bcrypt hashing with salt for password storage

### Data Protection
- **Input Validation**: Comprehensive validation against SQL injection and XSS attacks
- **HTTPS Support**: Encrypted communication with SSL/TLS certificates
- **CORS Configuration**: Configurable cross-origin resource sharing
- **Environment Isolation**: Separate development and production configurations

### API Security
- **Rate Limiting**: Protection against abuse and DDoS attacks
- **Request Validation**: Pydantic models for type-safe request validation
- **Secure Headers**: Security headers for enhanced protection
- **API Documentation**: Automatic OpenAPI documentation with security schemes

## 🛠️ Development

### Project Structure

```
easy_query/
├── backend/                           # FastAPI Backend
│   ├── data_sources/                 # Data source connectors
│   │   ├── mssql/                    # MSSQL database agents
│   │   ├── file_data/                # File processing system
│   │   └── db_query/                 # Query processing agents
│   ├── db_database_feature/          # Authentication & RBAC
│   │   └── knowledge_base_database_management/
│   ├── db_manager/                   # Database configuration
│   ├── Report_generator/             # Report generation tools
│   ├── voice_agent/                  # Voice interaction system
│   │   ├── frontend/                 # Voice agent UI components
│   │   ├── tools/                    # Voice agent tools
│   │   └── voice_agent_router.py     # Voice agent API router
│   ├── storage/                      # Generated reports and graphs
│   ├── uploads/                      # File upload directory
│   ├── requirements.txt              # Python dependencies
│   ├── main.py                       # Application entry point
│   └── Dockerfile                    # Backend container config
├── frontend/                         # Next.js Frontend
│   ├── src/
│   │   ├── app/                      # Next.js App Router pages
│   │   │   ├── database-query/       # Database query interface
│   │   │   ├── file-query/           # File processing interface
│   │   │   ├── ai-reports/           # AI report generation
│   │   │   ├── auth/                 # Authentication pages
│   │   │   ├── company-structure/    # Company management
│   │   │   ├── users/                # User management
│   │   │   └── tables/               # Table management
│   │   ├── components/               # React components
│   │   │   ├── ui/                   # shadcn/ui components
│   │   │   ├── dashboard/            # Dashboard components
│   │   │   ├── database-query/       # Query interface components
│   │   │   ├── file-query/           # File processing components
│   │   │   ├── voice-agent/          # Voice agent components
│   │   │   ├── 3d/                   # Three.js 3D components
│   │   │   └── auth/                 # Authentication components
│   │   ├── lib/                      # Utility libraries
│   │   │   ├── api/                  # API client functions
│   │   │   ├── hooks/                # Custom React hooks
│   │   │   └── utils/                # Utility functions
│   │   ├── store/                    # Zustand state management
│   │   └── types/                    # TypeScript type definitions
│   ├── public/                       # Static assets
│   ├── package.json                  # Node.js dependencies
│   ├── next.config.ts                # Next.js configuration
│   ├── tailwind.config.js            # Tailwind CSS config
│   ├── tsconfig.json                 # TypeScript configuration
│   └── Dockerfile                    # Frontend container config
└── README.md                         # Project documentation
```

### Adding New Features

#### Backend Development
1. **New API Routes**: Create routers in appropriate modules (`data_sources/`, `db_manager/`, etc.)
2. **Database Models**: Update SQLAlchemy models in `db_database_feature/knowledge_base_database_management/models.py`
3. **AI Integration**: Add new AI model integrations in `data_sources/mssql/mssql_agent3.py`
4. **Voice Agent Tools**: Extend voice capabilities in `voice_agent/tools/`

#### Frontend Development
1. **New Pages**: Add pages in `src/app/` following Next.js App Router conventions
2. **Components**: Create reusable components in `src/components/` with TypeScript
3. **State Management**: Add new stores in `src/store/` using Zustand
4. **API Integration**: Update API clients in `src/lib/api/`

#### Database Schema Changes
1. **Migrations**: Create database migrations for schema changes
2. **Models**: Update SQLAlchemy models and Pydantic schemas
3. **Validation**: Update input validation and business rules

#### Documentation Updates
1. **API Documentation**: Update endpoint documentation in this README
2. **Component Documentation**: Document new components with JSDoc comments
3. **Configuration**: Update environment variable documentation

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Workflow
1. **Fork the repository** and clone your fork
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Set up development environment**:
   ```bash
   # Backend
   cd backend && python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   
   # Frontend
   cd frontend && pnpm install
   ```
4. **Make your changes** following the coding standards
5. **Test your changes**:
   ```bash
   # Backend tests
   cd backend && python -m pytest
   
   # Frontend tests
   cd frontend && pnpm test && pnpm lint
   ```
6. **Commit your changes**: `git commit -m "feat: add new feature"`
7. **Push to your fork**: `git push origin feature/your-feature-name`
8. **Create a Pull Request** with a detailed description

### Coding Standards
- **Python**: Follow PEP 8, use type hints, and include docstrings
- **TypeScript**: Use strict TypeScript, follow ESLint rules, and include JSDoc comments
- **Commits**: Use conventional commit messages (`feat:`, `fix:`, `docs:`, etc.)
- **Testing**: Write tests for new functionality
- **Documentation**: Update relevant documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### Core Technologies
- **Google Gemini** - Advanced AI capabilities and multimodal processing
- **Groq** - High-performance LLM inference and processing
- **OpenAI** - Voice agent integration and real-time processing
- **FastAPI** - Modern, fast backend framework with automatic documentation
- **Next.js** - React framework with App Router and advanced optimization
- **Three.js & React Three Fiber** - 3D visualizations and interactive graphics

### Libraries & Tools
- **Pipecat** - Real-time voice agent framework
- **SQLAlchemy** - Python SQL toolkit and ORM
- **Tailwind CSS & shadcn/ui** - Modern UI framework and components
- **Zustand** - Lightweight state management
- **Redis** - High-performance caching and session storage
- **PostgreSQL & Microsoft SQL Server** - Robust database solutions

### Community
- All open-source contributors and maintainers
- The React, Python, and AI development communities
- Beta testers and early adopters who provided valuable feedback

## 📞 Support

### Getting Help
1. **Documentation**: Check this README and inline code documentation
2. **GitHub Issues**: Search existing issues or create a new one
3. **API Documentation**: Visit `/docs` endpoint when running the backend
4. **Community**: Join discussions in GitHub Discussions

### Reporting Issues
When reporting issues, please include:
- **Environment**: OS, Python version, Node.js version
- **Steps to reproduce**: Clear, numbered steps
- **Expected vs actual behavior**: What should happen vs what happens
- **Logs**: Relevant error messages and logs
- **Screenshots**: If applicable, include screenshots

### Feature Requests
We welcome feature requests! Please:
- Check existing issues first
- Provide a clear use case and description
- Include mockups or examples if applicable
- Consider contributing the feature yourself!