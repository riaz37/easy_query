#!/usr/bin/env python3
"""
Simplified Pipecat conversational AI bot using only Gemini Live Multimodal.
This version focuses on clean, maintainable code with function calling capabilities.
"""

import os
import sys
import asyncio
import uuid
from typing import Any, Dict
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.frames.frames import LLMMessagesAppendFrame, TranscriptionMessage
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

# Import our custom tools
from tools.navigation_tool import create_navigation_tool
from tool_websocket_registry import tool_websockets

# Import agent manager
from agent_manager import (
    AgentManager,
    MSSQL_SEARCH_AI_SYSTEM_INSTRUCTION
)

# Import the new user-specific registry
from tool_websocket_registry import (
    register_session_user,
    unregister_session_user,
    get_user_from_session,
    register_tool_websocket,
    unregister_tool_websocket,
    get_tool_websocket,
    get_all_users
)

load_dotenv(override=True)

# Session-based storage for navigation tools (supports multiple users)
session_navigation_tools = {}
session_websockets = {}
session_user_mapping = {}  # {session_id: user_id}

# Create all tools at the module level so they are accessible everywhere
rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
navigation_tool = create_navigation_tool(rtvi)

# Set .task property to None (or update as needed in your setup logic)
navigation_tool.task = None

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Use the MSSQL search system instruction as default
SYSTEM_INSTRUCTION = MSSQL_SEARCH_AI_SYSTEM_INSTRUCTION


def is_valid_api_key(key: str) -> bool:
    """Check if Google API key is valid."""
    if not key:
        return False
    
    # Check for common placeholder patterns
    placeholders = ["your_", "ai...", "example", "placeholder", "here", "key_here"]
    
    key_lower = key.lower()
    for placeholder in placeholders:
        if placeholder in key_lower:
            return False
    
    # Basic format validation for Google API key
    if not key.startswith("AI"):
        return False
    
    return True


async def create_gemini_live_llm(enable_function_calling: bool = True):
    """Create Gemini Live LLM service with optional function calling."""
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if not is_valid_api_key(google_key):
        raise ValueError("Invalid or missing GOOGLE_API_KEY. Please set a valid Google API key.")
    
    logger.info("🚀 Using Gemini Multimodal Live (native audio streaming)")
    
    # Create tools if function calling is enabled
    tools = None
    if enable_function_calling:
        # Create temporary RTVI processor for tool definitions
        rtvi_temp = RTVIProcessor(config=RTVIConfig(config=[]))
        
        # Create all tools
        navigation_tool_temp = create_navigation_tool(rtvi_temp)
        
        # Get function schemas
        tools = [
            navigation_tool_temp.get_tool_definition()
        ]
        
        # Log tool definitions for debugging
        for tool in tools:
            logger.info(f"🔧 Tool registered: {tool.name} - {tool.description}")
        
        # Create tools schema
        tools_schema = ToolsSchema(standard_tools=tools)
        logger.info(f"🔧 Function calling enabled with {len(tools)} tools")
    else:
        tools_schema = None
        logger.info("⚡ Function calling disabled")
    
    # Create Gemini Live service
    llm_service = GeminiMultimodalLiveLLMService(
        api_key=google_key,
        system_instruction=SYSTEM_INSTRUCTION,
        voice_id="Puck",  # Available voices: Zephyr, Aoede, Charon, Fenrir, Kore, Puck
        model="models/gemini-live-2.5-flash-preview",
        transcribe_model_audio=True,
        temperature=1,  # Moderate temperature for better function calling
        tools=tools_schema,
    )
    
    return llm_service


async def run_simplified_conversation_bot(websocket: WebSocket, session_id: str, user_id: str):
    """Run the simplified conversational AI bot with Gemini Live."""
    logger.info(f"🎤 Starting simplified conversation bot with Gemini Live - Session: {session_id}, User: {user_id}")
    
    # Create transport
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            serializer=ProtobufFrameSerializer(),
        ),
    )
    
    # Create LLM service with function calling
    llm_service = await create_gemini_live_llm(enable_function_calling=True)
    
    # Create proper context for function calling
    context = OpenAILLMContext([{
        "role": "user", 
        "content": "You are FunVoice, a helpful AI assistant. Use function calling when needed."
    }])
    context_aggregator = llm_service.create_context_aggregator(context)
    logger.info("✅ Context aggregator created for function calling")
    
    # Create RTVI processor for client communication
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
    
    # Create transcript processor for debugging
    transcript = TranscriptProcessor()
    
    # Create agent manager
    agent_manager = AgentManager()
    
    # Create session-specific tools
    session_navigation_tool = create_navigation_tool(rtvi)
    session_navigation_tools[session_id] = {
        "navigation": session_navigation_tool
    }
    logger.info(f"🔧 Session-specific tools created for session: {session_id}, user: {user_id}")
    
    # Build pipeline with context aggregator for function calling
    pipeline = Pipeline([
        transport.input(),
        context_aggregator.user(),
        rtvi,
        llm_service,
        transcript.user(),
        transport.output(),
        transcript.assistant(),
        context_aggregator.assistant(),
    ])
    
    # Create pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )
    
    # Set task reference for all tools
    navigation_tool.task = task
    session_navigation_tool.task = task
    agent_manager.task = task

    # Register tools with the agent manager
    agent_manager.register_tools(
        [
            navigation_tool.get_tool_definition()
        ],
        {
            "navigation": navigation_tool
        }
    )
    
    # Register function handlers for Gemini Live
    async def handle_function_call_gemini(params: FunctionCallParams):
        """Handle function calls from Gemini Live."""
        try:
            logger.info(f"🔧 FUNCTION CALL TRIGGERED: {params.function_name}")
            logger.info(f"🔧 Function arguments: {params.arguments}")
            logger.info(f"🔧 Function params type: {type(params)}")
            
            # Always override user_id with the correct session user_id
            params.arguments["user_id"] = user_id
            logger.info(f"🔧 Overrode user_id to correct session user: {user_id}")
            
            logger.info(f"🔧 Final function arguments: {params.arguments}")
            
            # Create a mock function call object for agent manager
            class MockFunctionCall:
                def __init__(self, name, arguments):
                    self.name = name
                    self.arguments = arguments
            
            mock_call = MockFunctionCall(params.function_name, params.arguments)
            logger.info(f"🔧 Mock call created: {mock_call.name} with {mock_call.arguments}")
            
            result = await agent_manager.handle_function_call(mock_call)
            logger.info(f"🔧 Function call result: {result}")
            await params.result_callback({"result": result})
            logger.info(f"🔧 Function call completed successfully")
        except Exception as e:
            logger.error(f"❌ Error in Gemini function handler: {e}")
            logger.error(f"❌ Error type: {type(e)}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            await params.result_callback({"error": str(e)})
    
    # Register all possible function names
    logger.info("🔧 Registering function handlers...")
    llm_service.register_function("navigate_page", handle_function_call_gemini)
    logger.info("🔧 Function handlers registered successfully")
    logger.info("🔧 Available functions: navigate_page")
    
    # Test function registration
    logger.info(f"🔧 LLM service type: {type(llm_service)}")
    logger.info(f"🔧 LLM service has register_function: {hasattr(llm_service, 'register_function')}")
    
    # Event handlers
    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("✅ Client ready - starting conversation")
        await rtvi.set_bot_ready()
        await task.queue_frames([context_aggregator.user().get_context_frame()])
    
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("🔗 Client connected to simplified conversation bot")
    
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("👋 Client disconnected from conversation bot")
        await task.cancel()
    
    # Transcript logging for debugging
    @transcript.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        for msg in frame.messages:
            if isinstance(msg, TranscriptionMessage):
                timestamp = f"[{msg.timestamp}] " if msg.timestamp else ""
                line = f"{timestamp}{msg.role}: {msg.content}"
                logger.info(f"📝 Transcript: {line}")
    
    # Run the pipeline
    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        await websocket.close(code=1000, reason="Pipeline error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle FastAPI startup and shutdown."""
    logger.info("🎤 Starting FunVoiceChat Simplified Server")
    logger.info("🚀 Using Gemini Live Multimodal for native audio streaming")
    yield
    logger.info("👋 Shutting down FunVoiceChat Simplified Server")


# Initialize FastAPI app
app = FastAPI(
    title="FunVoiceChat Simplified Server",
    description="Simplified conversational AI using Gemini Live Multimodal",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for RTVI conversation - accepts user_id parameter or sets default."""
    # Debug: Log all query parameters
    logger.info(f"🔍 WebSocket connection attempt - Query params: {websocket.query_params}")
    
    # Extract user_id from query parameters
    user_id = websocket.query_params.get("user_id")
    logger.info(f"🔍 Extracted user_id from query params: '{user_id}'")
    logger.info(f"🔍 All query params: {dict(websocket.query_params)}")
    
    if user_id:
        logger.info(f"🔍 Using user_id from query params: {user_id}")
    else:
        # Check if there's a recent tool WebSocket connection we can match
        from tool_websocket_registry import get_all_users
        all_users = get_all_users()
        if all_users:
            # Use the most recently connected user if no user_id provided
            user_id = all_users[-1]  # Get the last user
            logger.info(f"🔍 No user_id provided, using most recent tool connection: {user_id}")
        else:
            user_id = f"rtvi_user_{str(uuid.uuid4())[:8]}"
            logger.info(f"🔍 Generated default user_id: {user_id}")
    
    await websocket.accept()
    
    # Generate unique session ID for this connection
    session_id = str(uuid.uuid4())
    session_websockets[session_id] = websocket
    session_user_mapping[session_id] = user_id
    
    # Register session with user in the registry
    register_session_user(session_id, user_id)
    
    logger.info(f"🔗 WebSocket connection accepted for user {user_id} - Session: {session_id}")
    
    try:
        await run_simplified_conversation_bot(websocket, session_id, user_id)
    except Exception as e:
        logger.error(f"❌ Exception in conversation bot for user {user_id}, session {session_id}: {e}")
        try:
            await websocket.close(code=1000, reason="Bot error")
        except:
            pass
    finally:
        # Clean up session data
        if session_id in session_navigation_tools:
            del session_navigation_tools[session_id]
        if session_id in session_websockets:
            del session_websockets[session_id]
        if session_id in session_user_mapping:
            del session_user_mapping[session_id]
        unregister_session_user(session_id)
        logger.info(f"🧹 Cleaned up session: {session_id} for user: {user_id}")


@app.websocket("/ws/tools")
async def tools_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for tool commands - requires user_id parameter."""
    # Extract user_id from query parameters
    user_id = websocket.query_params.get("user_id")
    
    if not user_id:
        logger.error("❌ Tool WebSocket connection rejected: user_id parameter is required")
        await websocket.close(code=1008, reason="user_id parameter is required")
        return
    
    await websocket.accept()
    
    # Register the websocket for this user
    register_tool_websocket(user_id, websocket)
    
    logger.info(f"🔧 Tool WebSocket connection accepted for user {user_id}")
    
    try:
        # Keep connection alive and listen for any messages from client
        while True:
            try:
                # Wait for client messages (though we mainly send to client)
                message = await websocket.receive_text()
                logger.info(f"🔧 Received message from tool client {user_id}: {message}")
            except Exception as e:
                logger.info(f"🔧 Tool WebSocket {user_id} disconnected: {e}")
                break
    except Exception as e:
        logger.error(f"❌ Exception in tool WebSocket for user {user_id}: {e}")
    finally:
        # Clean up
        unregister_tool_websocket(user_id)
        try:
            await websocket.close()
        except:
            pass





@app.get("/health")
async def health_check():
    """Health check endpoint."""
    google_key = os.getenv("GOOGLE_API_KEY")
    gemini_available = is_valid_api_key(google_key)
    
    return {
        "status": "healthy",
        "llm_service": {
            "provider": "Gemini Live Multimodal",
            "available": gemini_available,
            "function_calling": True,
            "native_audio": True,
            "agent_type": "Navigation & Database Search AI"
        },
        "websockets": {
            "conversation": "/ws?user_id=your_user_id",
            "tools": "/ws/tools?user_id=your_user_id"
        },
        "endpoints": {
            "health": "/health",
            "connect": "/connect", 
            "set_user_id_post": "/set-user-id",
            "set_user_id_put": "/set-user-id/{session_id}/{user_id}",
            "get_user_id": "/get-user-id/{session_id}",
            "get_sessions": "/get-sessions",
            "test_database_search": "/test-database-search",
            "test_database_search_with_user": "/test-database-search-with-user",
            "test_navigation": "/test-navigation"
        },
        "sessions": {
            "active_sessions": len(session_navigation_tools),
            "total_users": len(get_all_users()),
            "users_with_tool_websockets": len([user_id for user_id in get_all_users() if get_tool_websocket(user_id)])
        },
        "expected_latency": "sub-500ms with native audio streaming"
    }


# Removed connect_user_ids - now using direct user_id in WebSocket URL

@app.post("/connect")
async def bot_connect(request: Request) -> Dict[Any, Any]:
    """RTVI connect endpoint for Pipecat client."""
    user_id = None
    
    # Try to get user_id from query parameters first
    user_id = request.query_params.get("user_id")
    if user_id:
        logger.info(f"🔗 Connect request with user_id from query: {user_id}")
    else:
        # Try to get from JSON body
        try:
            data = await request.json()
            user_id = data.get("user_id")
            logger.info(f"🔗 Connect request with user_id from body: {user_id}")
        except:
            logger.info("🔗 Connect request without user_id")
    
    if user_id:
        # Return URL with user_id parameter
        logger.info(f"🔗 Returning WebSocket URL with user_id: {user_id}")
        return {
            # "ws_url": f"wss://176.9.16.194:1294/ws?user_id={user_id}",
            "ws_url": f"ws://localhost:8002/ws?user_id={user_id}",
            "user_id": user_id,
            "note": "WebSocket connection will use the provided user_id"
        }
    else:
        # Return URL without user_id for backward compatibility
        logger.info("🔗 Connect request without user_id - using auto-generated ID")
        return {
            # "ws_url": f"wss://176.9.16.194:1294/ws",
            "ws_url": f"ws://localhost:8002/ws",
            "note": "WebSocket connection will auto-generate user_id"
        }


@app.get("/test-database-search")
async def test_database_search():
    """Test endpoint for database search via navigation tool."""
    try:
        # Create a test instance of the navigation tool
        rtvi_test = RTVIProcessor(config=RTVIConfig(config=[]))
        test_tool = create_navigation_tool(rtvi_test)
        
        # First navigate to database-query page
        user_id = "test_user"
        await test_tool.execute(user_id=user_id, target="database-query", action_type="navigate")
        
        # Then perform a database search
        test_query = "Show all the salary list of employee"
        result = await test_tool.execute(user_id=user_id, target=test_query, action_type="search")
        
        # Get user session info
        session_info = test_tool.get_user_session(user_id)
        
        return {
            "status": "success",
            "user_id": user_id,
            "test_query": test_query,
            "result": result,
            "session_info": session_info,
            "message": "Database search via navigation tool is working correctly",
            "websocket_sent": "Tool should have sent data via WebSocket to all connected clients"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Database search test failed"
        }




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
async def main():
    """Run the HTTPS-enabled server."""
    port = int(os.getenv("SERVEdR_PORT", 8200))
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        ssl_certfile="cert.pem",      # Path to your certificate
        ssl_keyfile="key.pem"         # Path to your private key
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())