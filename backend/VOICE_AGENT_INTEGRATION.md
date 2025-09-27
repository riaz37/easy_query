# Voice Agent Integration

This document explains how the voice agent functionality has been integrated into the main Knowledge Base API application.

## Overview

The voice agent from `voice_agent/main_simplified.py` has been successfully integrated into the main FastAPI application (`main.py`) as a router module. This allows both applications to run on the same server and share resources.

## Integration Details

### 1. Router Creation
- Created `voice_agent/voice_agent_router.py` that extracts all voice agent functionality
- Uses FastAPI's `APIRouter` with prefix `/voice` and tag `["Voice Agent"]`
- All endpoints are now accessible under the `/voice` prefix

### 2. Main Application Updates
- Updated `main.py` to import and include the voice agent router
- Voice agent now runs on the same port (8200) as the main application
- All existing main application functionality remains unchanged

### 3. Import Path Updates
- Updated all import paths in the voice agent router to use relative imports
- Fixed dependencies to work within the main application context

## Available Endpoints

### Voice Agent Endpoints (under `/voice` prefix)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/voice/health` | Health check for voice agent |
| POST | `/voice/connect` | Get WebSocket URL for voice connection |
| GET | `/voice/test-database-search` | Test database search functionality |
| WS | `/voice/ws` | Voice conversation WebSocket |
| WS | `/voice/ws/tools` | Tool commands WebSocket |

### Main Application Endpoints (unchanged)

All existing endpoints remain available:
- `/mssql/*` - MSSQL Agent endpoints
- `/files/*` - File Agent endpoints
- `/Auth/*` - Authentication endpoints
- `/graph/*` - Graph Generator endpoints
- `/reports/*` - Report Generation endpoints
- And more...

## Usage

### Starting the Application

```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Start the application
python main.py
```

The application will start on port 8200 with both main API and voice agent functionality.

### Connecting to Voice Agent

1. **Get WebSocket URL:**
   ```bash
   curl -X POST "http://localhost:8200/voice/connect" \
        -H "Content-Type: application/json" \
        -d '{"user_id": "your_user_id"}'
   ```

2. **Connect to Voice WebSocket:**
   ```javascript
   const ws = new WebSocket('ws://localhost:8200/voice/ws?user_id=your_user_id');
   ```

3. **Connect to Tools WebSocket:**
   ```javascript
   const toolsWs = new WebSocket('ws://localhost:8200/voice/ws/tools?user_id=your_user_id');
   ```

### Testing Integration

Run the integration test script:

```bash
python test_voice_integration.py
```

## Configuration

### Environment Variables

Make sure these environment variables are set:

```bash
# Required for voice agent
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Override server port
SERVER_PORT=8200
```

### Dependencies

The following new dependencies have been added to `requirements.txt`:

```
# Voice Agent Dependencies
pipecat
loguru
```

## Architecture

```
Knowledge Base API (main.py)
├── Main Application Routers
│   ├── MSSQL Agent (/mssql)
│   ├── File Agent (/files)
│   ├── Authentication (/Auth)
│   ├── Graph Generator (/graph)
│   └── Report Generation (/reports)
└── Voice Agent Router (/voice)
    ├── Health Check (/voice/health)
    ├── Connect Endpoint (/voice/connect)
    ├── Voice WebSocket (/voice/ws)
    ├── Tools WebSocket (/voice/ws/tools)
    └── Test Endpoints (/voice/test-*)
```

## Benefits of Integration

1. **Single Server**: Both applications run on the same server, reducing infrastructure complexity
2. **Shared Resources**: Can share database connections, configurations, and utilities
3. **Unified API**: All endpoints accessible through a single API gateway
4. **Easier Deployment**: Only one application to deploy and manage
5. **Consistent Port**: All functionality available on port 8200

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all voice agent dependencies are installed
2. **WebSocket Connection Issues**: Verify the correct WebSocket URLs are being used
3. **API Key Issues**: Ensure `GOOGLE_API_KEY` is properly set
4. **Port Conflicts**: Make sure port 8200 is available

### Debugging

1. Check application logs for detailed error messages
2. Use the health check endpoint: `GET /voice/health`
3. Run the integration test: `python test_voice_integration.py`
4. Verify all dependencies are installed: `pip install -r requirements.txt`

## Migration from Standalone

If you were previously using the standalone voice agent (`main_simplified.py`):

1. **Update WebSocket URLs**: Change from `ws://localhost:8002/ws` to `ws://localhost:8200/voice/ws`
2. **Update API Endpoints**: Add `/voice` prefix to all voice agent endpoints
3. **Update Port**: Change from port 8002 to port 8200
4. **Test Integration**: Run the integration test to verify everything works

## Future Enhancements

- Add authentication integration between main app and voice agent
- Share database connections and configurations
- Implement unified logging and monitoring
- Add cross-service communication capabilities
