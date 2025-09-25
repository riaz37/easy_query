#!/usr/bin/env python3
"""
Simplified Pipecat conversational AI bot using only Gemini Live Multimodal.
This version focuses on clean, maintainable code with function calling capabilities.
"""

import os
import sys
import asyncio
import uuid
import json
from typing import Any, Dict
from datetime import datetime
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request, APIRouter, HTTPException
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
from pipecat.transports.base_transport import TransportParams

# Import our custom tools
from voice_agent.tools.navigation_tool import create_navigation_tool
from voice_agent.tool_websocket_registry import tool_websockets

# Import agent manager
from voice_agent.agent_manager import (
    AgentManager,
    MSSQL_SEARCH_AI_SYSTEM_INSTRUCTION
)

# Import the new user-specific registry
from voice_agent.tool_websocket_registry import (
    register_session_user,
    unregister_session_user,
    get_user_from_session,
    register_tool_websocket,
    unregister_tool_websocket,
    get_tool_websocket,
    get_all_users
)

# Import text agent
from voice_agent.text_agent import LangChainTextAgent, MockLangChainAgent, create_text_agent

load_dotenv(override=True)

# Check if Google API key is available at module level
GOOGLE_API_KEY_AVAILABLE = bool(os.getenv("GOOGLE_API_KEY"))
if not GOOGLE_API_KEY_AVAILABLE:
    logger.warning("⚠️ GOOGLE_API_KEY not set. Voice and text agents will not be available.")
    logger.warning("⚠️ Set GOOGLE_API_KEY environment variable to enable Gemini services.")
else:
    logger.info("✅ GOOGLE_API_KEY is set and available")


class TextWebSocketTransport:
    """Text-only WebSocket transport that handles text input/output without audio."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self._connected = True
        self._event_handlers = {}

    def input(self):
        """Return self for input in pipeline."""
        return self

    def output(self):
        """Return self for output in pipeline."""
        return self

    async def process_input_frame(self, frame):
        """Process incoming frames (not used for text input)."""
        pass

    async def process_frame(self, frame):
        """Process outgoing frames by sending text responses."""
        # Handle different types of frames
        if hasattr(frame, 'text') and frame.text:
            try:
                await self.websocket.send_text(frame.text)
                logger.info("📤 Text response sent to client")
            except Exception as e:
                logger.error(f"❌ Error sending text response: {e}")
                self._connected = False
        elif hasattr(frame, 'content') and frame.content:
            # Handle LLM response frames
            try:
                await self.websocket.send_text(frame.content)
                logger.info("📤 LLM response sent to client")
            except Exception as e:
                logger.error(f"❌ Error sending LLM response: {e}")
                self._connected = False

    async def send_text(self, text: str):
        """Send text message to client."""
        try:
            if self._connected:
                await self.websocket.send_text(text)
                logger.info("📤 Text message sent to client")
        except Exception as e:
            logger.error(f"❌ Error sending text message: {e}")
            self._connected = False

    def event_handler(self, event_name: str):
        """Decorator for event handlers."""
        def decorator(func):
            self._event_handlers[event_name] = func
            return func
        return decorator

    async def trigger_event(self, event_name: str, *args, **kwargs):
        """Trigger an event if handler exists."""
        if event_name in self._event_handlers:
            await self._event_handlers[event_name](*args, **kwargs)

    async def close(self):
        """Close the transport."""
        self._connected = False
        try:
            await self.websocket.close()
        except:
            pass

# Environment configuration
def get_backend_url():
    """Get the appropriate backend URL based on environment."""
    environment = os.getenv("ENVIRONMENT", "development")

    # Debug: Print all relevant environment variables
    logger.info("🔧 === ENVIRONMENT DEBUG INFO ===")
    logger.info(f"🔧 ENVIRONMENT variable: '{environment}'")
    logger.info(f"🔧 DEV_BACKEND_URL: '{os.getenv('DEV_BACKEND_URL', 'NOT_SET')}'")
    logger.info(f"🔧 PROD_BACKEND_URL: '{os.getenv('PROD_BACKEND_URL', 'NOT_SET')}'")
    logger.info(f"🔧 GOOGLE_API_KEY set: {bool(os.getenv('GOOGLE_API_KEY'))}")
    logger.info(f"🔧 Current working directory: {os.getcwd()}")

    urls = {
        "development": os.getenv("DEV_BACKEND_URL", "https://localhost:8200"),
        "production": os.getenv("PROD_BACKEND_URL", "https://176.9.16.194:8200")
    }

    # Fix hardcoded fallback - if we're in production but no PROD_BACKEND_URL is set,
    # don't fall back to localhost, use the production default
    if environment == "production" and not os.getenv("PROD_BACKEND_URL"):
        selected_url = "https://176.9.16.194:8200"
        logger.warning("⚠️  PROD_BACKEND_URL not set, using default production URL")
    elif environment == "development" and not os.getenv("DEV_BACKEND_URL"):
        selected_url = "https://localhost:8200"
        logger.warning("⚠️  DEV_BACKEND_URL not set, using default development URL")
    else:
        selected_url = urls.get(environment, urls["development"])
    logger.info(f"🔧 Backend Environment: {environment}")
    logger.info(f"🌐 Backend URL: {selected_url}")
    logger.info("🔧 ================================")

    return selected_url

# Session-based storage for navigation tools (supports multiple users)
session_navigation_tools = {}
session_websockets = {}
session_user_mapping = {}  # {session_id: user_id}

# Text agent instances for each user
user_text_agents = {}  # {user_id: TextAgent}

# Create RTVI processor at the module level
rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

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


async def create_gemini_live_llm(enable_function_calling: bool = True, current_page: str = "dashboard"):
    """Create Gemini Live LLM service with optional function calling and current page context."""
    try:
        google_key = os.getenv("GOOGLE_API_KEY")
        
        if not is_valid_api_key(google_key):
            logger.warning("⚠️ GOOGLE_API_KEY not set or invalid. Voice agent will not be available.")
            return None
        
        logger.info("🚀 Using Gemini Multimodal Live (native audio streaming)")
        logger.info(f"🔧 Current page context: {current_page}")
        
        # Create tools if function calling is enabled
        tools = None
        if enable_function_calling:
            # Create temporary RTVI processor for tool definitions
            rtvi_temp = RTVIProcessor(config=RTVIConfig(config=[]))
            
            # Create all tools with current page context
            navigation_tool_temp = create_navigation_tool(rtvi_temp, initial_current_page=current_page)
            
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
        
        # Create agent manager to get system instruction with current page context
        agent_manager = AgentManager(current_page=current_page)
        system_instruction_with_context = agent_manager.get_system_instruction_with_page_context()
        
        # Create Gemini Live service
        llm_service = GeminiMultimodalLiveLLMService(
            api_key=google_key,
            system_instruction=system_instruction_with_context,
            voice_id="Puck",  # Available voices: Zephyr, Aoede, Charon, Fenrir, Kore, Puck
            model="models/gemini-live-2.5-flash-preview",
            transcribe_model_audio=True,
            temperature=1,  # Moderate temperature for better function calling
            tools=tools_schema,
        )
        
        return llm_service
        
    except Exception as e:
        logger.error(f"❌ Failed to create Gemini Live LLM service: {e}")
        logger.warning("⚠️ Voice agent will not be available. Text agent will still work.")
        return None


async def create_text_gemini_live_llm(enable_function_calling: bool = True, current_page: str = "dashboard"):
    """Create Gemini Live LLM service for text-only conversations."""
    try:
        google_key = os.getenv("GOOGLE_API_KEY")

        if not is_valid_api_key(google_key):
            logger.warning("⚠️ GOOGLE_API_KEY not set or invalid. Text agent will not be available.")
            return None

        logger.info("📝 Using Gemini Live Multimodal for text-only conversations")
        logger.info(f"🔧 Current page context: {current_page}")

        # Create tools if function calling is enabled
        tools = None
        if enable_function_calling:
            # Create temporary RTVI processor for tool definitions
            rtvi_temp = RTVIProcessor(config=RTVIConfig(config=[]))

            # Create all tools with current page context
            navigation_tool_temp = create_navigation_tool(rtvi_temp, initial_current_page=current_page)

            # Get function schemas
            tools = [
                navigation_tool_temp.get_tool_definition()
            ]

            # Log tool definitions for debugging
            for tool in tools:
                logger.info(f"🔧 Tool registered: {tool.name} - {tool.description}")

            # Create tools schema
            tools_schema = ToolsSchema(standard_tools=tools)
            logger.info(f"🔧 Text function calling enabled with {len(tools)} tools")
        else:
            tools_schema = None
            logger.info("⚡ Text function calling disabled")

        # Create Gemini Live service for text-only
        llm_service = GeminiMultimodalLiveLLMService(
            api_key=google_key,
            system_instruction=SYSTEM_INSTRUCTION,
            voice_id="Puck",  # Required but not used for text
            model="models/gemini-live-2.5-flash-preview",
            transcribe_model_audio=False,  # Disable audio transcription for text-only
            temperature=1,  # Moderate temperature for better function calling
            tools=tools_schema,
        )

        return llm_service
        
    except Exception as e:
        logger.error(f"❌ Failed to create Gemini Live LLM service for text: {e}")
        logger.warning("⚠️ Text agent will not be available.")
        return None


async def run_simplified_conversation_bot(websocket: WebSocket, session_id: str, user_id: str, current_page: str = "dashboard"):
    """Run the simplified conversational AI bot with Gemini Live."""
    logger.info(f"🎤 Starting simplified conversation bot with Gemini Live - Session: {session_id}, User: {user_id}, Current Page: {current_page}")
    
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
    
    # Create LLM service with function calling and current page context
    llm_service = await create_gemini_live_llm(enable_function_calling=True, current_page=current_page)
    
    # Check if LLM service was created successfully
    if not llm_service:
        logger.error("❌ Failed to create LLM service. Voice agent will not be available.")
        await websocket.close(code=1008, reason="LLM service unavailable")
        return
    
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
    
    # Create agent manager with current page context
    agent_manager = AgentManager(current_page=current_page)
    logger.info(f"🔧 Created agent manager with current page: {current_page}")
    
    # Create session-specific tools with current page context
    session_navigation_tool = create_navigation_tool(rtvi, initial_current_page=current_page)
    session_navigation_tools[session_id] = {
        "navigation": session_navigation_tool
    }
    logger.info(f"🔧 Session-specific tools created for session: {session_id}, user: {user_id}")
    
    # Navigation tool is now initialized with current page context
    logger.info(f"🔧 Navigation tool initialized with current page: {current_page}")
    
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
    session_navigation_tool.task = task
    agent_manager.task = task

    # Register tools with the agent manager (use session-specific navigation tool)
    agent_manager.register_tools(
        [
            session_navigation_tool.get_tool_definition()
        ],
        {
            "navigation": session_navigation_tool
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


async def run_text_conversation_bot(websocket: WebSocket, session_id: str, user_id: str, current_page: str = "dashboard"):
    """Run the text-only conversational AI bot with LangChain and Gemini."""
    logger.info(f"📝 Starting text conversation bot with LangChain + Gemini - Session: {session_id}, User: {user_id}, Current Page: {current_page}")

    # Import LangChain components
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.memory import ConversationBufferWindowMemory
        from langchain_core.messages import SystemMessage
        from langchain_core.runnables import RunnableConfig
        LANGCHAIN_AVAILABLE = True
        logger.info("✅ LangChain components loaded successfully")
    except ImportError as e:
        logger.error(f"❌ LangChain not available: {e}")
        LANGCHAIN_AVAILABLE = False

    if not LANGCHAIN_AVAILABLE:
        # Fallback if LangChain is not available
        await websocket.send_text("LangChain is not available. Please install required dependencies.")
        return

    # Create RTVI processor for navigation tool
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Create navigation tool with current page context
    navigation_tool_instance = create_navigation_tool(rtvi, initial_current_page=current_page)

    # Create comprehensive navigation tool adapter for LangChain
    class ComprehensiveTextNavigationTool:
        """Comprehensive navigation tool adapted for text conversations with full functionality."""

        def __init__(self, navigation_tool, user_id):
            self.navigation_tool = navigation_tool
            self.user_id = user_id
            self.name = "navigate_page"
            self.description = "Navigate between pages, interact with page elements, perform database searches, or file system operations. Supports all action types: 'navigate' for page changes, 'click'/'interact' for button interactions, 'search' for database queries, 'file_search' for file system searches, 'file_upload' for file uploads, 'view_report' for viewing reports, 'generate_report' for creating reports."

        async def __call__(self, target: str, action_type: str = "navigate", context: str = None):
            """Execute comprehensive navigation action."""
            try:
                logger.info(f"🧭 Comprehensive text navigation tool called: {action_type} -> {target}")
                logger.info(f"🧭 User ID: {self.user_id}, Context: {context}")

                # Execute the navigation with full functionality
                result = await self.navigation_tool.execute(
                    user_id=self.user_id,
                    target=target,
                    action_type=action_type,
                    context=context
                )

                logger.info(f"🧭 Comprehensive navigation result: {result}")
                return result
            except Exception as e:
                logger.error(f"❌ Comprehensive navigation tool error: {e}")
                import traceback
                logger.error(f"❌ Traceback: {traceback.format_exc()}")
                return f"Error executing navigation: {str(e)}"

    # Create comprehensive tool instance
    comprehensive_nav_tool = ComprehensiveTextNavigationTool(navigation_tool_instance, user_id)

    # Create LangChain tools with comprehensive functionality
    from langchain.tools import tool

    @tool
    async def navigate_page(target: str, action_type: str = "navigate", context: str = None) -> str:
        """Navigate between pages, interact with elements, search databases, handle files, and manage reports. 
        
        Args:
            target: Target page, element, query, or operation
            action_type: Type of action - 'navigate', 'click', 'interact', 'search', 'file_search', 'file_upload', 'view_report', 'generate_report'
            context: Additional context or parameters
        
        Examples:
            - navigate_page("dashboard", "navigate") - Go to dashboard
            - navigate_page("configure database", "click") - Click button
            - navigate_page("Show employee data", "search") - Database search
            - navigate_page("Find project docs", "file_search") - File search
            - navigate_page("Upload data", "file_upload", "file_descriptions:data,table_names:users") - File upload
            - navigate_page("Sales report", "view_report") - View existing report
            - navigate_page("Monthly report", "generate_report") - Generate new report
        """
        return await comprehensive_nav_tool(target, action_type, context)

    @tool
    async def click_button(element_name: str, context: str = None) -> str:
        """Click a button or interact with page elements.
        
        Args:
            element_name: Name of the button or element to click
            context: Additional context (e.g., db_id for 'set database' button)
        
        Examples:
            - click_button("load tables") - Click load tables button
            - click_button("set database", "db_id:123") - Set database with ID
            - click_button("configure database") - Click configure button
        """
        return await comprehensive_nav_tool(element_name, "click", context)

    @tool
    async def search_database(query: str, context: str = None) -> str:
        """Search database records with SQL-like queries.
        
        Args:
            query: Search query or SQL-like statement
            context: Additional search context
        
        Examples:
            - search_database("Show all users")
            - search_database("Find employees with salary > 50000")
            - search_database("Get customer data from Dubai")
        """
        return await comprehensive_nav_tool(query, "search", context)

    @tool
    async def search_files(query: str, table_specific: bool = False, tables: str = None) -> str:
        """Search file system and documents.
        
        Args:
            query: Search query for files
            table_specific: Whether search is table-specific
            tables: Comma-separated list of table names
        
        Examples:
            - search_files("Find project guidelines")
            - search_files("Financial documents", True, "finance,accounting")
        """
        context = None
        if table_specific and tables:
            context = f"table_specific:true,tables:{tables}"
        elif table_specific:
            context = "table_specific:true,tables:string"
        
        return await comprehensive_nav_tool(query, "file_search", context)

    @tool
    async def upload_files(description: str, table_names: str = None, file_descriptions: str = None) -> str:
        """Upload files to the system.
        
        Args:
            description: Description of the upload operation
            table_names: Comma-separated target table names
            file_descriptions: Comma-separated file descriptions
        
        Examples:
            - upload_files("Upload user data", "users,profiles")
            - upload_files("Upload reports", "reports", "monthly,quarterly")
        """
        context_parts = []
        if file_descriptions:
            context_parts.append(f"file_descriptions:{file_descriptions}")
        if table_names:
            context_parts.append(f"table_names:{table_names}")
        
        context = ",".join(context_parts) if context_parts else None
        return await comprehensive_nav_tool(description, "file_upload", context)

    @tool
    async def view_report(report_name: str, context: str = None) -> str:
        """View existing reports.
        
        Args:
            report_name: Name or description of the report to view
            context: Additional context for report viewing
        
        Examples:
            - view_report("Sales report for Q4")
            - view_report("User activity report")
        """
        return await comprehensive_nav_tool(report_name, "view_report", context)

    @tool
    async def generate_report(report_description: str, context: str = None) -> str:
        """Generate new reports.
        
        Args:
            report_description: Description of the report to generate
            context: Additional parameters for report generation
        
        Examples:
            - generate_report("Monthly sales report")
            - generate_report("User engagement analysis for Q1 2024")
        """
        return await comprehensive_nav_tool(report_description, "generate_report", context)

    # Get Google API key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        await websocket.send_text("Google API key not configured.")
        return

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0.7,
        max_output_tokens=2048,
    )
    
    # Test the LLM directly
    try:
        test_response = await llm.ainvoke("Hello, can you respond to this test message?")
        logger.info(f"🔍 LLM test response: {test_response.content}")
    except Exception as llm_error:
        logger.error(f"❌ LLM test failed: {llm_error}")
        # Continue anyway, might be a temporary issue

    # Initialize memory
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=20
    )

    # Create tools list with comprehensive functionality
    tools = [
        navigate_page, 
        click_button,
        search_database, 
        search_files,
        upload_files,
        view_report,
        generate_report
    ]

    # Create agent manager to get system instruction with current page context
    agent_manager = AgentManager(current_page=current_page)
    system_instruction_with_context = agent_manager.get_system_instruction_with_page_context()
    
    # Use the dynamic system instruction from agent manager instead of hardcoded one
    comprehensive_system_instruction = system_instruction_with_context
    
    # Use comprehensive system instruction instead of simplified one
    
    # Create prompt template with comprehensive instruction
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=comprehensive_system_instruction),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create the agent
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

    logger.info("✅ LangChain agent initialized for text conversation with COMPREHENSIVE functionality")
    logger.info(f"🔍 Agent type: {type(agent)}")
    logger.info(f"🔍 Agent executor type: {type(agent_executor)}")
    logger.info(f"🔍 Available comprehensive tools: {[tool.name for tool in tools]}")
    logger.info(f"🔍 Comprehensive system instruction length: {len(comprehensive_system_instruction)}")
    logger.info("🎆 Text agent now has FULL FUNCTIONALITY: navigate, click, interact, search, file operations, reports!")

    # Handle text input from client
    try:
        while True:
            logger.info(f"📝 Waiting for message from user {user_id}...")

            # Receive text message from client
            try:
                text_message = await websocket.receive_text()
                logger.info(f"📝 Received text message from user {user_id}: '{text_message}'")
                
                # Try to parse JSON message if it looks like JSON
                if text_message.strip().startswith('{') or '"message"' in text_message:
                    try:
                        # Extract JSON part if it's mixed with other text
                        import re
                        json_match = re.search(r'\{.*\}', text_message)
                        if json_match:
                            json_str = json_match.group(0)
                            parsed_message = json.loads(json_str)
                            if 'message' in parsed_message:
                                text_message = parsed_message['message']
                                logger.info(f"📝 Parsed JSON message: '{text_message}'")
                    except json.JSONDecodeError as json_error:
                        logger.warning(f"⚠️ Failed to parse JSON message: {json_error}")
                        # Continue with original message
                
            except Exception as receive_error:
                logger.info(f"📝 WebSocket connection closed for user {user_id}: {receive_error}")
                break

            try:
                # Process message through LangChain agent
                logger.info("🤖 Processing message through LangChain agent...")

                # Add a timeout for the agent execution
                try:
                    result = await asyncio.wait_for(
                        agent_executor.ainvoke(
                            {"input": text_message},
                            config=RunnableConfig(tags=[f"user_{user_id}"])
                        ),
                        timeout=30.0  # 30 second timeout
                    )
                except asyncio.TimeoutError:
                    logger.error("❌ Agent execution timed out")
                    await websocket.send_text("I'm taking too long to process your request. Please try again.")
                    continue
                except Exception as agent_error:
                    logger.error(f"❌ Agent execution failed: {agent_error}")
                    # Try direct LLM response as fallback
                    try:
                        direct_response = await llm.ainvoke(f"User is on {current_page} page and asks: {text_message}")
                        await websocket.send_text(direct_response.content)
                        logger.info(f"📤 Direct LLM response sent: {direct_response.content}")
                        continue
                    except Exception as direct_error:
                        logger.error(f"❌ Direct LLM response also failed: {direct_error}")
                        # Use fallback response
                        pass

                response_text = result["output"]
                intermediate_steps = result.get("intermediate_steps", [])

                # Debug: Log the full result
                logger.info(f"🔍 Full agent result: {result}")
                logger.info(f"🔍 Response text: '{response_text}'")
                logger.info(f"🔍 Response text type: {type(response_text)}")
                logger.info(f"🔍 Response text length: {len(response_text) if response_text else 0}")

                # Log tool usage
                if intermediate_steps:
                    logger.info(f"🔧 Agent used {len(intermediate_steps)} tools")
                    for i, (action, observation) in enumerate(intermediate_steps):
                        logger.debug(f"Step {i+1}: {action.tool} -> {observation[:100]}...")

                        # Navigation tool already sends its own results to WebSocket
                        # No need to send additional tool execution result

                # Send conversation response to text WebSocket
                if response_text and response_text.strip():
                    try:
                        await websocket.send_text(response_text)
                        logger.info(f"📤 Conversation response sent to user {user_id}: '{response_text[:100]}...'")
                    except Exception as send_error:
                        logger.error(f"❌ Error sending response to WebSocket: {send_error}")
                        # Try to send a fallback message
                        try:
                            await websocket.send_text("I processed your message but encountered an error sending the response.")
                        except:
                            logger.error("❌ Could not send fallback message either")
                else:
                    logger.warning(f"⚠️ Empty or null response from agent: '{response_text}'")
                    # Generate a fallback response based on current page
                    fallback_responses = {
                        "database-query": "On the database-query page, you can search the database, view existing reports, or generate new reports. What would you like to do?",
                        "file-query": "On the file-query page, you can search for files in the system or upload new files. What would you like to do?",
                        "user-configuration": "On the user-configuration page, you can configure database settings or business rules. What would you like to do?",
                        "tables": "On the tables page, you can load tables, view table visualizations, import Excel files, or get AI suggestions. What would you like to do?",
                        "users": "On the users page, you can manage database access or vector database access. What would you like to do?",
                        "dashboard": "On the dashboard page, you can navigate to other pages in the application. Where would you like to go?",
                        "company-structure": "On the company-structure page, you can navigate to other pages in the application. Where would you like to go?"
                    }
                    
                    fallback_response = fallback_responses.get(current_page, "I'm here to help you with database and file operations. What would you like to do?")
                    
                    # Send a fallback response
                    try:
                        await websocket.send_text(fallback_response)
                        logger.info(f"📤 Fallback response sent to user {user_id}: '{fallback_response}'")
                    except Exception as send_error:
                        logger.error(f"❌ Error sending fallback response: {send_error}")

            except Exception as e:
                logger.error(f"❌ Error processing message with agent: {e}")
                import traceback
                logger.error(f"❌ Traceback: {traceback.format_exc()}")

                error_msg = "I apologize, but I encountered an error processing your message."
                await websocket.send_text(error_msg)

    except Exception as e:
        logger.error(f"❌ Error in text conversation loop for user {user_id}: {e}")
        try:
            await websocket.close(code=1000, reason="Conversation error")
        except Exception as close_error:
            logger.debug(f"WebSocket already closed: {close_error}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle FastAPI startup and shutdown."""
    logger.info("🎤 Starting FunVoiceChat Simplified Server")
    logger.info("🚀 Using Gemini Live Multimodal for native audio streaming")
    yield
    logger.info("👋 Shutting down FunVoiceChat Simplified Server")



router = APIRouter(tags=["Voice Agent"])

# Debug: Log router creation
print("🔧 Voice Agent Router created with tags: Voice Agent")

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for RTVI conversation - accepts user_id and current_page parameters."""
    # Debug: Log all query parameters
    logger.info(f"🔍 WebSocket connection attempt - Query params: {websocket.query_params}")
    
    # Extract user_id from query parameters
    user_id = websocket.query_params.get("user_id")
    logger.info(f"🔍 Extracted user_id from query params: '{user_id}'")
    logger.info(f"🔍 All query params: {dict(websocket.query_params)}")
    
    if user_id:
        # Clean up user_id - remove any leading/trailing whitespace
        user_id = user_id.strip()
        logger.info(f"🔍 Using cleaned user_id from query params: '{user_id}' (length: {len(user_id)})")
    else:
        # Check if there's a recent tool WebSocket connection we can match
        from voice_agent.tool_websocket_registry import get_all_users
        all_users = get_all_users()
        if all_users:
            # Use the most recently connected user if no user_id provided
            user_id = all_users[-1]  # Get the last user
            logger.info(f"🔍 No user_id provided, using most recent tool connection: {user_id}")
        else:
            user_id = f"rtvi_user_{str(uuid.uuid4())[:8]}"
            logger.info(f"🔍 Generated default user_id: {user_id}")
    
    # Extract current_page from query parameters
    current_page = websocket.query_params.get("current_page")
    if current_page:
        # Clean up current_page - remove any leading/trailing whitespace and normalize
        current_page = current_page.strip().lower().replace(" ", "-").replace("_", "-")
        logger.info(f"🔍 Using current_page from query params: '{current_page}'")
    else:
        logger.error("❌ Voice WebSocket connection rejected: current_page parameter is required")
        await websocket.close(code=1008, reason="current_page parameter is required")
        return
    
    # Validate current_page is not null, None, or invalid
    if current_page == "none" or current_page == "null" or current_page == "xxxxxx":
        logger.error(f"❌ Voice WebSocket connection rejected: invalid current_page value: '{current_page}'")
        await websocket.close(code=1008, reason=f"invalid current_page value: {current_page}")
        return
    
    await websocket.accept()
    
    # Generate unique session ID for this connection
    session_id = str(uuid.uuid4())
    session_websockets[session_id] = websocket
    session_user_mapping[session_id] = user_id
    
    # Register session with user in the registry
    register_session_user(session_id, user_id)
    
    logger.info(f"🔗 WebSocket connection accepted for user {user_id} - Session: {session_id} - Current Page: {current_page}")
    
    try:
        await run_simplified_conversation_bot(websocket, session_id, user_id, current_page)
    except Exception as e:
        logger.error(f"❌ Exception in conversation bot for user {user_id}, session {session_id}: {e}")
        try:
            await websocket.close(code=1000, reason="Bot error")
        except Exception as close_error:
            logger.debug(f"WebSocket already closed in voice bot: {close_error}")
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


@router.websocket("/ws/tools")
async def tools_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for tool commands - requires user_id parameter, accepts current_page parameter."""
    # Debug: Log all query parameters
    logger.info(f"🔧 Tool WebSocket connection attempt - Query params: {websocket.query_params}")
    
    # Extract user_id from query parameters
    user_id = websocket.query_params.get("user_id")
    logger.info(f"🔧 Extracted user_id from query params: '{user_id}'")
    
    if not user_id:
        logger.error("❌ Tool WebSocket connection rejected: user_id parameter is required")
        await websocket.close(code=1008, reason="user_id parameter is required")
        return
    
    # Clean up user_id - remove any leading/trailing whitespace
    user_id = user_id.strip()
    logger.info(f"🔧 Cleaned user_id: '{user_id}' (length: {len(user_id)})")
    
    # Extract current_page from query parameters
    current_page = websocket.query_params.get("current_page", "dashboard")
    if current_page:
        # Clean up current_page - remove any leading/trailing whitespace and normalize
        current_page = current_page.strip().lower().replace(" ", "-").replace("_", "-")
        logger.info(f"🔧 Using current_page from query params: '{current_page}'")
    else:
        current_page = "dashboard"
        logger.info(f"🔧 No current_page provided, using default: {current_page}")
    
    await websocket.accept()
    
    # Register the websocket for this user
    register_tool_websocket(user_id, websocket)
    
    logger.info(f"🔧 Tool WebSocket connection accepted for user {user_id} - Current Page: {current_page}")
    
    # Send a confirmation message to the client
    try:
        confirmation_message = {
            "type": "connection_confirmation",
            "user_id": user_id,
            "current_page": current_page,
            "message": f"Tool WebSocket connected successfully for user {user_id} on {current_page} page",
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send_json(confirmation_message)
        logger.info(f"📤 Sent confirmation message to tool client {user_id}")
    except Exception as send_error:
        logger.error(f"❌ Error sending confirmation message: {send_error}")
    
    try:
        # Keep connection alive and listen for any messages from client
        while True:
            try:
                # Wait for client messages (though we mainly send to client)
                message = await websocket.receive_text()
                logger.info(f"🔧 Received message from tool client {user_id}: {message}")
                
                # Send an echo response
                try:
                    echo_response = {
                        "type": "echo_response",
                        "user_id": user_id,
                        "current_page": current_page,
                        "original_message": message,
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send_json(echo_response)
                    logger.info(f"📤 Sent echo response to tool client {user_id}")
                except Exception as echo_error:
                    logger.error(f"❌ Error sending echo response: {echo_error}")
                    
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



@router.websocket("/ws/text-conversation")
async def text_conversation_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for text conversation with Gemini Live LLM and tool calling."""
    logger.info("🔗 New connection attempt to /ws/text-conversation")

    # Extract user_id from query parameters
    user_id = websocket.query_params.get("user_id")
    logger.info(f"🔍 Query params: {dict(websocket.query_params)}")
    logger.info(f"🔍 Extracted user_id: '{user_id}'")

    if not user_id:
        logger.error("❌ Text conversation WebSocket connection rejected: user_id parameter is required")
        await websocket.close(code=1008, reason="user_id parameter is required")
        return

    # Clean up user_id - remove any leading/trailing whitespace
    user_id = user_id.strip()
    logger.info(f"📝 Cleaned user_id for text conversation WebSocket: '{user_id}' (length: {len(user_id)})")

    # Extract current_page from query parameters
    current_page = websocket.query_params.get("current_page")
    if current_page:
        # Clean up current_page - remove any leading/trailing whitespace and normalize
        current_page = current_page.strip().lower().replace(" ", "-").replace("_", "-")
        logger.info(f"🔍 Using current_page from query params: '{current_page}'")
    else:
        logger.error("❌ Text conversation WebSocket connection rejected: current_page parameter is required")
        await websocket.close(code=1008, reason="current_page parameter is required")
        return
    
    # Validate current_page is not null, None, or invalid
    if current_page == "none" or current_page == "null" or current_page == "xxxxxx":
        logger.error(f"❌ Text conversation WebSocket connection rejected: invalid current_page value: '{current_page}'")
        await websocket.close(code=1008, reason=f"invalid current_page value: {current_page}")
        return

    await websocket.accept()
    logger.info("✅ Text conversation WebSocket connection accepted")

    # Generate unique session ID for this connection
    session_id = str(uuid.uuid4())
    logger.info(f"📝 Text conversation WebSocket connection accepted for user {user_id} - Session: {session_id} - Current Page: {current_page}")

    try:
        # Run the text conversation bot with Gemini Live and tool calling
        await run_text_conversation_bot(websocket, session_id, user_id, current_page)
    except Exception as e:
        logger.error(f"❌ Exception in text conversation WebSocket for user {user_id}, session {session_id}: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        try:
            await websocket.close(code=1000, reason="Bot error")
        except Exception as close_error:
            logger.debug(f"WebSocket already closed in outer handler: {close_error}")


@router.get("/health")
async def voice_agent_health_check():
    """Health check endpoint for voice agent."""
    google_key = os.getenv("GOOGLE_API_KEY")
    gemini_available = is_valid_api_key(google_key)
    
    # Get current environment and backend URL
    backend_url = get_backend_url()
    environment = os.getenv("ENVIRONMENT", "development")
    
    return {
        "status": "healthy",
        "environment": {
            "mode": environment,
            "backend_url": backend_url,
            "websocket_base": backend_url.replace("https://", "wss://")
        },
        "llm_service": {
            "provider": "Gemini Live Multimodal + LangChain",
            "available": gemini_available,
            "function_calling": True,
            "native_audio": True,
            "text_agent": True,
            "text_agent_capabilities": "COMPREHENSIVE - Full navigation, click, interact, search, file operations, reports",
            "agent_type": "Comprehensive Navigation & Database Search AI with LangChain",
            "parity_with_voice_agent": True
        },
        "websockets": {
            "conversation": f"{backend_url.replace('https://', 'wss://')}/voice/ws?user_id=your_user_id&current_page=database-query",
            "tools": f"{backend_url.replace('https://', 'wss://')}/voice/ws/tools?user_id=your_user_id&current_page=database-query",
            "text_conversation": f"{backend_url.replace('https://', 'wss://')}/voice/ws/text?user_id=your_user_id",
            "text_conversation_langchain": f"{backend_url.replace('https://', 'wss://')}/voice/ws/text-conversation?user_id=your_user_id&current_page=database-query"
        },
        "endpoints": {
            "health": "/voice/health",
            "connect": "/voice/connect",
            "test_database_search": "/voice/test-database-search",
            "test_function_call": "/voice/test-function-call",
            "memory_get": "/voice/memory/{user_id}",
            "memory_clear": "/voice/memory/{user_id}",
            "memory_save": "/voice/memory/{user_id}/save",
            "memory_load": "/voice/memory/{user_id}/load",
            "cleanup_text_agent": "/voice/text-agent/{user_id}",
            "text_conversation_test": "/voice/ws/text-conversation"
        },
        "sessions": {
            "active_sessions": len(session_navigation_tools),
            "total_users": len(get_all_users()),
            "users_with_tool_websockets": len([user_id for user_id in get_all_users() if get_tool_websocket(user_id)]),
            "text_agents": len(user_text_agents),
            "users_with_text_agents": list(user_text_agents.keys())
        },
        "expected_latency": "sub-500ms with native audio streaming"
            }


# Memory management endpoints
@router.get("/memory/{user_id}")
async def get_user_memory(user_id: str):
    """Get the conversation memory for a specific user."""
    if user_id not in user_text_agents:
        return {"error": f"No text agent found for user {user_id}"}

    text_agent = user_text_agents[user_id]
    memory = await text_agent.get_memory()

    return {
        "user_id": user_id,
        "memory_length": len(memory),
        "memory": memory
    }


@router.delete("/memory/{user_id}")
async def clear_user_memory(user_id: str):
    """Clear the conversation memory for a specific user."""
    if user_id not in user_text_agents:
        return {"error": f"No text agent found for user {user_id}"}

    text_agent = user_text_agents[user_id]
    success = await text_agent.clear_memory()

    if success:
        return {"message": f"Memory cleared for user {user_id}"}
    else:
        return {"error": f"Failed to clear memory for user {user_id}"}


@router.post("/memory/{user_id}/save")
async def save_user_memory(user_id: str):
    """Save the current memory state for a specific user."""
    if user_id not in user_text_agents:
        return {"error": f"No text agent found for user {user_id}"}

    text_agent = user_text_agents[user_id]
    memory_data = await text_agent.save_memory()

    return memory_data


@router.post("/memory/{user_id}/load")
async def load_user_memory(user_id: str, memory_data: dict):
    """Load memory data for a specific user."""
    if user_id not in user_text_agents:
        # Use default dashboard page for new agents
        user_text_agents[user_id] = create_text_agent(user_id, "dashboard")

    text_agent = user_text_agents[user_id]
    success = await text_agent.load_memory(memory_data.get("memory", []))

    if success:
        return {"message": f"Memory loaded for user {user_id}"}
    else:
        return {"error": f"Failed to load memory for user {user_id}"}


@router.delete("/text-agent/{user_id}")
async def cleanup_text_agent(user_id: str):
    """Clean up a text agent instance for a specific user."""
    if user_id in user_text_agents:
        # Save memory before cleanup (optional)
        text_agent = user_text_agents[user_id]
        await text_agent.save_memory()

        del user_text_agents[user_id]
        return {"message": f"Text agent cleaned up for user {user_id}"}
    else:
        return {"error": f"No text agent found for user {user_id}"}


# Removed connect_user_ids - now using direct user_id in WebSocket URL

@router.post("/connect")
async def bot_connect(request: Request) -> Dict[Any, Any]:
    """RTVI connect endpoint for Pipecat client."""
    user_id = None
    current_page = None
    
    # Try to get user_id and current_page from query parameters first
    user_id = request.query_params.get("user_id")
    current_page = request.query_params.get("current_page")
    print(f"🔗 Connect request with user_id from query: {user_id}")
    print(f"🔗 Connect request with current_page from query: {current_page}")
    
    if user_id:
        logger.info(f"🔗 Connect request with user_id from query: {user_id}")
    if current_page:
        logger.info(f"🔗 Connect request with current_page from query: {current_page}")
    
    if not user_id or not current_page:
        # Try to get from JSON body
        try:
            data = await request.json()
            if not user_id:
                user_id = data.get("user_id")
            if not current_page:
                current_page = data.get("current_page")
            logger.info(f"🔗 Connect request with user_id from body: {user_id}")
            logger.info(f"🔗 Connect request with current_page from body: {current_page}")
        except:
            logger.info("🔗 Connect request without user_id or current_page")
    
    # CRITICAL: current_page is now REQUIRED - do not proceed without it
    if not current_page or current_page == "None" or current_page == "null":
        error_msg = "❌ current_page parameter is required for connection"
        logger.error(error_msg)
        logger.error(f"🔗 Received current_page: '{current_page}' (type: {type(current_page)})")
        raise HTTPException(
            status_code=400,
            detail={
                "error": error_msg,
                "message": "Frontend must provide current_page parameter",
                "required_parameters": ["current_page"],
                "optional_parameters": ["user_id"],
                "note": "current_page tells the AI which page the user is currently on",
                "received_current_page": current_page,
                "received_user_id": user_id
            }
        )
    
    # Get the appropriate backend URL based on environment
    backend_url = get_backend_url()
    ws_base_url = backend_url.replace("https://", "wss://")
    
    if user_id:
        # Return URL with user_id and current_page parameters
        ws_url = f"{ws_base_url}/voice/ws?user_id={user_id}&current_page={current_page}"
        logger.info(f"🔗 Returning WebSocket URL with user_id: {user_id} and current_page: {current_page}")
        logger.info(f"🔗 WebSocket URL: {ws_url}")
        return {
            "ws_url": ws_url,
            "user_id": user_id,
            "current_page": current_page,
            "note": "WebSocket connection will use the provided user_id and current_page",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "backend_url": backend_url
        }
    else:
        # Return URL without user_id for backward compatibility
        ws_url = f"{ws_base_url}/voice/ws?current_page={current_page}"
        logger.info("🔗 Connect request without user_id - using auto-generated ID")
        logger.info(f"🔗 WebSocket URL: {ws_url}")
        return {
            "ws_url": ws_url,
            "current_page": current_page,
            "note": "WebSocket connection will auto-generate user_id but use provided current_page",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "backend_url": backend_url
        }


@router.get("/test-function-call")
async def test_function_call():
    """Test endpoint for function calling capabilities."""
    try:
        # Test if we can create a navigation tool instance
        from voice_agent.tools.navigation_tool import create_navigation_tool
        from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIConfig
        
        # Create a test RTVI processor
        rtvi_test = RTVIProcessor(config=RTVIConfig(config=[]))
        test_tool = create_navigation_tool(rtvi_test, initial_current_page="tables")
        
        # Test a simple navigation function call
        user_id = "test_function_call_user"
        result = await test_tool.execute(
            user_id=user_id,
            target="dashboard",
            action_type="navigate"
        )
        
        return {
            "message": "Function calling test successful",
            "test_result": result,
            "user_id": user_id,
            "tool_available": True,
            "function_calling_enabled": True,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Function calling test failed: {e}")
        return {
            "message": "Function calling test failed",
            "error": str(e),
            "tool_available": False,
            "function_calling_enabled": False,
            "timestamp": datetime.now().isoformat()
        }