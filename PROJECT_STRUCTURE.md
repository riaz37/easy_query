# Project Structure

This document explains the organization of the Easy Query Knowledge Base codebase.

## Root Directory

```
easy_query/
├── backend/                    # FastAPI Backend
├── frontend/                   # Next.js Frontend
├── README.md                  # Main project documentation
├── API_DOCUMENTATION.md       # API endpoints documentation
├── DEPLOYMENT.md              # Deployment guide
├── FEATURES.md                # Features overview
├── GETTING_STARTED.md         # Quick start guide
├── PROJECT_STRUCTURE.md       # This file
├── docker-compose.yml         # Docker Compose configuration
└── LICENSE                    # MIT License
```

## Backend Structure

```
backend/
├── main.py                           # Application entry point
├── requirements.txt                  # Python dependencies (361 packages)
├── Dockerfile                        # Backend container configuration
├── data_sources/                     # Data source connectors and agents
│   ├── mssql/                        # Microsoft SQL Server integration
│   │   ├── mssql_agent3.py          # Main scalable MSSQL agent
│   │   ├── conversation_memory.json  # Conversation history
│   │   └── memory_manager.py        # Memory management
│   ├── file_data/                    # Advanced file processing system
│   │   ├── router.py                # File processing API router
│   │   ├── app/                     # File processing applications
│   │   │   ├── unstructured/        # Unstructured document processing
│   │   │   └── semi_structured/     # Semi-structured document processing
│   │   └── uploads/                 # File upload directory
│   └── db_query/                     # Database query utilities
│       └── agents/                  # AI agents for query processing
├── db_database_feature/              # Authentication and RBAC system
│   └── knowledge_base_database_management/
│       ├── router.py                # Authentication API router
│       ├── models.py                # Database models
│       ├── schemas.py               # Pydantic schemas
│       ├── services/                # Business logic services
│       └── README.md                # RBAC documentation
├── db_manager/                       # Database configuration management
│   ├── data_base_config.py          # Database configuration API
│   ├── mssql_config.py              # MSSQL configuration
│   └── utilites/                    # Database utilities
│       ├── schema_generator.py      # Schema generation
│       ├── table_info_generator.py  # Table information
│       └── semi_structured_To_table_db.py  # Data conversion
├── Report_generator/                 # Report and chart generation
│   ├── utilites/                    # Report utilities
│   │   ├── graph_Generator.py       # Chart generation
│   │   └── report_agent.py          # Report processing
│   └── storage/                     # Generated reports storage
├── voice_agent/                      # Voice interaction system
│   ├── voice_agent_router.py        # Voice agent API router
│   ├── agent_manager.py             # Agent management
│   ├── text_agent.py                # Text-based agent
│   ├── tools/                       # Voice agent tools
│   ├── frontend/                    # Voice agent UI components
│   └── memory/                      # Voice agent memory
├── storage/                          # Generated content storage
│   ├── graphs/                      # Generated charts and graphs
│   └── reports/                     # Generated reports
└── uploads/                          # File upload storage
```

### Key Backend Components

1. **MSSQL Agent (`data_sources/mssql/`)**
   - `mssql_agent3.py`: Main scalable database querying system
   - Natural language to SQL conversion with AI models
   - Business rules integration and validation
   - Conversation memory management with context
   - Multi-model AI support (Gemini, Groq, OpenAI)

2. **Authentication System (`db_database_feature/`)**
   - Comprehensive user signup/login with validation
   - JWT token management with revocation
   - Role-based access control (RBAC)
   - Company management with hierarchical structure
   - Permission management and isolation

3. **File Processing System (`data_sources/file_data/`)**
   - Smart file routing (structured vs unstructured)
   - Advanced document processing pipelines
   - Embedding generation and intent mapping
   - Background processing with progress tracking
   - Multi-format support (PDF, Excel, Word, CSV, JSON)

4. **Database Manager (`db_manager/`)**
   - Database connection configuration and management
   - Schema generation and validation
   - Table information utilities and metadata
   - Data conversion and transformation tools

5. **Report Generator (`Report_generator/`)**
   - Interactive chart creation with multiple formats
   - HTML report generation with rich formatting
   - Data visualization and export capabilities
   - Real-time report processing

6. **Voice Agent (`voice_agent/`)**
   - Real-time voice interaction with sub-500ms latency
   - Speech recognition and text-to-speech
   - Voice command processing and function calling
   - WebSocket communication for real-time interaction
   - Page-aware voice interactions

## Frontend Structure

```
frontend/
├── src/
│   ├── app/                      # Next.js App Router pages
│   │   ├── database-query/       # Database query interface
│   │   ├── file-query/           # File processing interface
│   │   ├── ai-reports/           # AI report generation
│   │   ├── auth/                 # Authentication pages
│   │   ├── company-structure/    # Company management
│   │   ├── users/                # User management
│   │   ├── tables/               # Table management
│   │   ├── user-configuration/   # User settings
│   │   ├── globals.css           # Global styles
│   │   └── layout.tsx            # Root layout
│   ├── components/               # React components
│   │   ├── ui/                   # shadcn/ui components (63 files)
│   │   ├── dashboard/            # Dashboard components
│   │   ├── database-query/       # Query interface components
│   │   ├── file-query/           # File processing components
│   │   ├── voice-agent/          # Voice agent components
│   │   ├── 3d/                   # Three.js 3D components
│   │   ├── auth/                 # Authentication components
│   │   ├── reports/              # Report generation components
│   │   └── shared/               # Shared components
│   ├── lib/                      # Utility libraries
│   │   ├── api/                  # API client functions (28 files)
│   │   ├── hooks/                # Custom React hooks (26 files)
│   │   ├── utils/                # Utility functions (5 files)
│   │   └── voice-agent/          # Voice agent utilities (16 files)
│   ├── store/                    # Zustand state management
│   │   ├── file-upload-store.ts  # File upload state
│   │   ├── query-store.ts        # Query state management
│   │   ├── task-store.ts         # Task tracking state
│   │   ├── theme-store.ts        # Theme management
│   │   └── uiStore.ts            # UI state management
│   └── types/                    # TypeScript type definitions
│       ├── api.ts                # API types
│       ├── auth.ts               # Authentication types
│       ├── reports.ts            # Report types
│       └── index.ts              # Type exports
├── public/                       # Static assets
│   ├── ai-results/               # AI result icons
│   ├── dashboard/                # Dashboard icons (16 files)
│   ├── file-query/               # File query icons
│   ├── tables/                   # Table icons (20 files)
│   └── user-configuration/       # User config icons
├── package.json                  # Frontend dependencies (72 packages)
├── next.config.ts                # Next.js configuration
├── tailwind.config.js            # Tailwind CSS configuration
├── tsconfig.json                 # TypeScript configuration
├── components.json               # shadcn/ui configuration
├── Dockerfile                    # Frontend container configuration
└── pnpm-lock.yaml                # Package lock file
```

### Key Frontend Components

1. **Dashboard (`src/components/dashboard/`)**
   - Main application interface with 3D visualizations
   - Drag-and-drop system cards with animations
   - Interactive background effects and neon styling
   - System status monitoring and health checks

2. **Database Query (`src/components/database-query/`)**
   - Advanced query input forms with AI assistance
   - Real-time results display with multiple formats
   - Query history management and favorites
   - Integrated report generation capabilities
   - Chart visualization with Recharts integration

3. **File Query (`src/components/file-query/`)**
   - Smart file upload with drag-and-drop
   - File processing progress tracking
   - Table generation and preview
   - Advanced file processing options

4. **Voice Agent (`src/components/voice-agent/`)**
   - Real-time voice interaction interface
   - Voice command visualization
   - Connection status and health monitoring
   - Voice agent configuration and settings

5. **Authentication (`src/components/auth/`)**
   - Modern login/signup forms with validation
   - Password change and profile management
   - Company structure and user management
   - Role-based access control interface

6. **Reports (`src/components/reports/`)**
   - Interactive chart visualization
   - Report generation interface with templates
   - Export capabilities (PDF, Excel, HTML)
   - Real-time processing status

7. **3D Components (`src/components/3d/`)**
   - Three.js-powered 3D visualizations
   - Interactive brain mesh and models
   - Scene lighting and animations
   - Performance-optimized rendering

## Data Flow

1. **User Interaction**: User asks a question through the web interface or voice
2. **Frontend Processing**: Question is sent to the backend API with context
3. **AI Processing**: Backend uses AI models (Gemini, Groq, OpenAI) to convert question to SQL
4. **Business Rules**: Query is validated against business rules and constraints
5. **Database Query**: Generated SQL is executed against the configured database
6. **Result Processing**: Results are formatted, cached, and processed
7. **Display**: Results are visualized in the web interface with charts and reports
8. **Voice Response**: For voice interactions, results are converted to speech

## Technology Stack

### Backend
- **FastAPI 0.115.11**: High-performance async web framework
- **Python 3.11+**: Latest Python features and performance
- **SQLAlchemy**: Database ORM with connection pooling
- **PostgreSQL/MSSQL**: Enterprise database support
- **Redis**: High-performance caching and session storage
- **JWT**: Secure authentication and authorization
- **Pipecat**: Voice agent framework for real-time interaction

### Frontend
- **Next.js 15.3.3**: Latest React framework with App Router
- **React 19**: Latest React with concurrent features
- **TypeScript**: Full type safety throughout
- **Tailwind CSS v4**: Utility-first styling
- **shadcn/ui**: Modern, accessible components
- **Three.js**: 3D visualizations and interactions
- **Zustand**: Lightweight state management
- **Recharts**: Data visualization and charts

### AI & ML
- **Google Gemini**: Advanced reasoning and multimodal capabilities
- **Groq**: High-performance LLM inference
- **OpenAI**: GPT models and Whisper for voice
- **Embeddings**: Vector embeddings for semantic search
- **LangChain**: AI agent framework and tool usage

## Configuration Files

- `backend/.env`: Environment variables (not in version control)
- `backend/requirements.txt`: Backend Python dependencies (361 packages)
- `frontend/package.json`: Frontend JavaScript dependencies (72 packages)
- `docker-compose.yml`: Docker Compose configuration with health checks
- `backend/Dockerfile`: Backend container configuration
- `frontend/Dockerfile`: Frontend container configuration