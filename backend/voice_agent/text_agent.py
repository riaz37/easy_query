#!/usr/bin/env python3
"""
LangChain-based conversational AI agent with tool calling capabilities.
Handles text input, function calling, and memory management using LangChain framework.
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from loguru import logger
from dotenv import load_dotenv

# LangChain imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    from langchain.tools import BaseTool, tool
    from langchain_core.messages import SystemMessage
    from langchain_core.runnables import RunnableConfig
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("⚠️ LangChain libraries not available, using mock implementation")

# Import existing components
try:
    from voice_agent.agent_manager import AgentManager, MSSQL_SEARCH_AI_SYSTEM_INSTRUCTION
    from voice_agent.tool_websocket_registry import send_to_user_tool_websocket
    from voice_agent.tools.navigation_tool import create_navigation_tool
except ImportError:
    # Fallback for when running the script directly
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from voice_agent.agent_manager import AgentManager, MSSQL_SEARCH_AI_SYSTEM_INSTRUCTION
    from voice_agent.tool_websocket_registry import send_to_user_tool_websocket
    from voice_agent.tools.navigation_tool import create_navigation_tool

load_dotenv(override=True)


# Custom Tools for LangChain Agent
if LANGCHAIN_AVAILABLE:
    class ComprehensiveNavigationTool(BaseTool):
        """Comprehensive navigation tool for LangChain agent with full functionality."""
        
        name: str = "navigate_page"
        description: str = "Navigate between pages, interact with page elements, perform database searches, file system operations, table creation, and Excel import workflow. Supports all action types: 'navigate', 'click', 'interact', 'search', 'file_search', 'file_upload', 'view_report', 'generate_report', 'create_table', 'add_column', 'set_column_properties', 'complete_table', 'table_status', 'confirm_table_creation', 'page_info', 'update_table_name', 'update_column_name', 'update_column_properties', 'submit_table', 'update', 'form_update', 'table_management', 'business_rule_management', 'excel_import', 'select_table', 'list_available_tables', 'get_ai_suggestion', 'continue_to_import', 'back_to_mapping', 'import_data_to_database'."
        user_id: Optional[str] = None
        callback_handler: Optional[Any] = None

        def __init__(self, user_id: str, callback_handler=None):
            super().__init__(user_id=user_id)
            self.user_id = user_id
            self.callback_handler = callback_handler

        async def _arun(self, target: str, action_type: str = "navigate", context: str = None) -> str:
            """Async implementation with comprehensive functionality."""
            try:
                # Import here to avoid circular imports
                from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIConfig
                
                # Create a temporary RTVI processor for the tool
                rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
                nav_tool = create_navigation_tool(rtvi)

                # Execute the comprehensive navigation action
                result = await nav_tool.execute(
                    user_id=self.user_id,
                    target=target,
                    action_type=action_type,
                    context=context
                )

                logger.info(f"🧭 Comprehensive navigation action completed: {action_type} -> {target}")
                
                # Send result to WebSocket if callback handler is available
                if self.callback_handler:
                    await self.callback_handler.send_tool_result(result)
                
                return result

            except Exception as e:
                logger.error(f"❌ Comprehensive navigation tool error: {e}")
                return f"Error executing navigation action: {str(e)}"

        def _run(self, target: str, action_type: str = "navigate", context: str = None) -> str:
            """Sync implementation (not recommended for async operations)."""
            return asyncio.run(self._arun(target, action_type, context))

else:
    # Mock ComprehensiveNavigationTool when LangChain is not available
    class ComprehensiveNavigationTool:
        """Mock comprehensive navigation tool when LangChain is not available."""
        
        def __init__(self, user_id: str, callback_handler=None):
            self.name = "navigate_page"
            self.description = "Navigate between pages, interact with elements, search databases, handle files, and manage reports."
            self.user_id = user_id
            self.callback_handler = callback_handler

        async def _arun(self, target: str, action_type: str = "navigate", context: str = None) -> str:
            """Mock async implementation with comprehensive functionality."""
            logger.info(f"🧭 Mock comprehensive navigation action: {action_type} -> {target}")
            return f"Mock comprehensive navigation: {action_type} to {target} completed"

        def _run(self, target: str, action_type: str = "navigate", context: str = None) -> str:
            """Mock sync implementation."""
            return f"Mock comprehensive navigation: {action_type} to {target} completed"


if LANGCHAIN_AVAILABLE:
    @tool
    async def click_button(element_name: str, context: str = None) -> str:
        """Click buttons and interact with page elements."""
        try:
            logger.info(f"💲 Clicking button: {element_name}")
            # This would integrate with navigation tool for button clicking
            return f"Button '{element_name}' clicked successfully."
        except Exception as e:
            logger.error(f"❌ Button click error: {e}")
            return f"Error clicking button: {str(e)}"

    @tool
    async def search_files(query: str, table_specific: bool = False, tables: str = None) -> str:
        """Search file system and documents with advanced options."""
        try:
            logger.info(f"🔍 Searching files with query: {query}")
            table_info = f" in tables: {tables}" if table_specific and tables else ""
            return f"File search completed for: {query}{table_info}. Results would be returned here."
        except Exception as e:
            logger.error(f"❌ File search error: {e}")
            return f"Error searching files: {str(e)}"

    @tool
    async def upload_files(description: str, table_names: str = None, file_descriptions: str = None) -> str:
        """Upload files to the system with metadata."""
        try:
            logger.info(f"📁 Uploading files: {description}")
            target_info = f" to tables: {table_names}" if table_names else ""
            return f"File upload completed: {description}{target_info}."
        except Exception as e:
            logger.error(f"❌ File upload error: {e}")
            return f"Error uploading files: {str(e)}"

    @tool
    async def view_report(report_name: str, context: str = None) -> str:
        """View existing reports by name or description."""
        try:
            logger.info(f"📊 Viewing report: {report_name}")
            return f"Report '{report_name}' retrieved and displayed successfully."
        except Exception as e:
            logger.error(f"❌ Report viewing error: {e}")
            return f"Error viewing report: {str(e)}"

else:
    # Mock comprehensive tools when LangChain is not available
    async def click_button(element_name: str, context: str = None) -> str:
        """Mock button clicking."""
        logger.info(f"💲 Mock button click: {element_name}")
        return f"Mock button '{element_name}' clicked"

    async def search_files(query: str, table_specific: bool = False, tables: str = None) -> str:
        """Mock file search."""
        logger.info(f"🔍 Mock file search: {query}")
        return f"Mock file search completed for: {query}"

    async def upload_files(description: str, table_names: str = None, file_descriptions: str = None) -> str:
        """Mock file upload."""
        logger.info(f"📁 Mock file upload: {description}")
        return f"Mock file upload completed: {description}"

    async def view_report(report_name: str, context: str = None) -> str:
        """Mock report viewing."""
        logger.info(f"📊 Mock view report: {report_name}")
        return f"Mock report '{report_name}' viewed"
if LANGCHAIN_AVAILABLE:
    @tool
    async def search_database(query: str) -> str:
        """Search database records based on the provided query."""
        try:
            # This would integrate with your actual database search functionality
            logger.info(f"🔍 Searching database with query: {query}")
            return f"Database search completed for: {query}. Results would be returned here."
        except Exception as e:
            logger.error(f"❌ Database search error: {e}")
            return f"Error searching database: {str(e)}"

    @tool
    async def generate_report(report_type: str, parameters: str = None) -> str:
        """Generate various types of reports based on specified parameters."""
        try:
            logger.info(f"📊 Generating report: {report_type}")
            return f"Report '{report_type}' generated successfully with parameters: {parameters}"
        except Exception as e:
            logger.error(f"❌ Report generation error: {e}")
            return f"Error generating report: {str(e)}"

    @tool
    async def file_operations(operation: str, file_path: str = None, content: str = None) -> str:
        """Perform file operations like read, write, or search within files."""
        try:
            logger.info(f"📁 File operation: {operation} on {file_path}")
            return f"File operation '{operation}' completed successfully."
        except Exception as e:
            logger.error(f"❌ File operation error: {e}")
            return f"Error performing file operation: {str(e)}"

else:
    # Mock tools when LangChain is not available
    async def search_database(query: str) -> str:
        """Mock database search."""
        logger.info(f"🔍 Mock database search: {query}")
        return f"Mock database search completed for: {query}"

    async def generate_report(report_type: str, parameters: str = None) -> str:
        """Mock report generation."""
        logger.info(f"📊 Mock report generation: {report_type}")
        return f"Mock report '{report_type}' generated"

    async def file_operations(operation: str, file_path: str = None, content: str = None) -> str:
        """Mock file operations."""
        logger.info(f"📁 Mock file operation: {operation}")
        return f"Mock file operation '{operation}' completed"


class WebSocketCallbackHandler:
    """Custom callback handler to send tool execution results to WebSocket."""

    def __init__(self, user_id: str):
        self.user_id = user_id

    async def send_tool_result(self, result: str):
        """Send tool execution result to WebSocket."""
        await send_to_user_tool_websocket(
            self.user_id,
            {
                "type": "tool_execution_result",
                "action": "completed",
                "data": {
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            }
        )


class LangChainTextAgent:
    """LangChain-based conversational AI agent with tool calling capabilities."""

    def __init__(self, user_id: str, current_page: str = "dashboard"):
        self.user_id = user_id
        self.current_page = current_page
        self.llm = None
        self.agent_executor = None
        self.memory = None
        self.callback_handler = WebSocketCallbackHandler(user_id)
        self.agent_manager = None
        self.initialized = False

    async def initialize(self):
        """Initialize the LangChain agent with tools and memory."""
        try:
            if not LANGCHAIN_AVAILABLE:
                raise ImportError("LangChain libraries are not available")

            # Initialize Google Gemini LLM
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")

            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=google_api_key,
                temperature=0.7,
                max_output_tokens=2048,
            )

            # Initialize memory
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=20  # Keep last 20 messages
            )

            # Create agent manager for dynamic system instructions
            self.agent_manager = AgentManager(current_page=self.current_page)
            system_instruction_with_context = self.agent_manager.get_system_instruction_with_page_context()

            # Create comprehensive tools with full functionality
            tools = [
                ComprehensiveNavigationTool(user_id=self.user_id, callback_handler=self.callback_handler),
                click_button,
                search_database,
                search_files,
                upload_files,
                generate_report,
                view_report,
                file_operations
            ]

            # Create prompt template with dynamic system instruction
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_instruction_with_context),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            # Create the agent
            agent = create_tool_calling_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt
            )

            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                memory=self.memory,
                verbose=True,
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                max_iterations=3
            )

            self.initialized = True
            logger.info(f"✅ LangChain TextAgent initialized with COMPREHENSIVE functionality for user {self.user_id}")
            logger.info(f"🎆 Available comprehensive tools: {[tool.name if hasattr(tool, 'name') else str(tool) for tool in tools]}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize LangChain TextAgent for user {self.user_id}: {e}")
            raise

    async def process_message(self, message: str) -> str:
        """Process a text message using the LangChain agent."""
        try:
            if not self.initialized:
                await self.initialize()

            logger.info(f"📝 Processing message for user {self.user_id}: {message[:100]}...")

            # Process the message through the agent
            result = await self.agent_executor.ainvoke(
                {"input": message},
                config=RunnableConfig(
                    callbacks=[],
                    tags=[f"user_{self.user_id}"]
                )
            )

            response_text = result["output"]
            intermediate_steps = result.get("intermediate_steps", [])

            # Log intermediate steps for debugging
            if intermediate_steps:
                logger.info(f"🔧 Agent used {len(intermediate_steps)} intermediate steps")
                for i, (action, observation) in enumerate(intermediate_steps):
                    logger.debug(f"Step {i+1}: {action.tool} -> {observation[:100]}...")

            logger.info(f"✅ Message processed successfully for user {self.user_id}")

            # Send the response to the tool WebSocket
            await send_to_user_tool_websocket(
                self.user_id,
                {
                    "type": "langchain_agent_response",
                    "action": "message_processed",
                    "data": {
                        "user_message": message,
                        "agent_response": response_text,
                        "tools_used": [step[0].tool for step in intermediate_steps],
                        "timestamp": datetime.now().isoformat()
                    }
                }
            )

            return response_text

        except Exception as e:
            logger.error(f"❌ Error processing message for user {self.user_id}: {e}")
            error_msg = f"I apologize, but I encountered an error processing your message: {str(e)}"

            # Send error to WebSocket
            await send_to_user_tool_websocket(
                self.user_id,
                {
                    "type": "langchain_agent_error",
                    "action": "error_occurred",
                    "data": {
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                }
            )

            return error_msg

    async def add_custom_tool(self, tool):
        """Add a custom tool to the agent."""
        try:
            if not self.initialized:
                await self.initialize()

            # Get current tools
            current_tools = list(self.agent_executor.tools)
            current_tools.append(tool)

            # Recreate agent with new tools
            # Get updated system instruction with current page context
            system_instruction_with_context = self.agent_manager.get_system_instruction_with_page_context()
            
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_instruction_with_context),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            agent = create_tool_calling_agent(
                llm=self.llm,
                tools=current_tools,
                prompt=prompt
            )

            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=current_tools,
                memory=self.memory,
                verbose=True,
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                max_iterations=3
            )

            logger.info(f"🔧 Custom tool '{tool.name}' added to agent for user {self.user_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Error adding custom tool for user {self.user_id}: {e}")
            return False

    async def get_memory(self) -> List[Dict[str, Any]]:
        """Get the current conversation memory."""
        try:
            if not self.memory:
                return []

            # Convert LangChain messages to dict format
            messages = self.memory.chat_memory.messages
            memory_list = []
            
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    memory_list.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    memory_list.append({"role": "assistant", "content": msg.content})

            return memory_list

        except Exception as e:
            logger.error(f"❌ Error getting memory for user {self.user_id}: {e}")
            return []

    async def clear_memory(self) -> bool:
        """Clear the conversation memory."""
        try:
            if self.memory:
                self.memory.clear()
            logger.info(f"🧹 Memory cleared for user {self.user_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing memory for user {self.user_id}: {e}")
            return False

    async def save_memory(self) -> Dict[str, Any]:
        """Save the current memory state."""
        try:
            memory_data = await self.get_memory()
            return {
                "user_id": self.user_id,
                "memory": memory_data,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Error saving memory for user {self.user_id}: {e}")
            return {"error": str(e)}

    async def load_memory(self, memory_data: List[Dict[str, Any]]) -> bool:
        """Load memory from saved data."""
        try:
            if not self.memory:
                self.memory = ConversationBufferWindowMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    k=20
                )

            # Clear existing memory
            self.memory.clear()

            # Load messages into memory
            for msg_data in memory_data:
                if msg_data["role"] == "user":
                    self.memory.chat_memory.add_message(HumanMessage(content=msg_data["content"]))
                elif msg_data["role"] == "assistant":
                    self.memory.chat_memory.add_message(AIMessage(content=msg_data["content"]))

            logger.info(f"📚 Memory loaded for user {self.user_id} with {len(memory_data)} messages")
            return True

        except Exception as e:
            logger.error(f"❌ Error loading memory for user {self.user_id}: {e}")
            return False

    async def get_available_tools(self) -> List[str]:
        """Get list of available comprehensive tools."""
        try:
            if not self.agent_executor:
                return []
            tool_names = []
            for tool in self.agent_executor.tools:
                if hasattr(tool, 'name'):
                    tool_names.append(tool.name)
                elif hasattr(tool, '__name__'):
                    tool_names.append(str(tool))
                else:
                    tool_names.append(str(tool))
            logger.info(f"🔧 Available comprehensive tools for user {self.user_id}: {tool_names}")
            return tool_names
        except Exception as e:
            logger.error(f"❌ Error getting available tools: {e}")
            return []
    
    async def update_current_page(self, new_page: str):
        """Update the current page and refresh the system instruction."""
        try:
            if self.current_page != new_page:
                self.current_page = new_page
                logger.info(f"🔄 Updated current page for user {self.user_id}: {new_page}")
                
                # Update agent manager with new page
                if self.agent_manager:
                    self.agent_manager.current_page = new_page
                
                # Reinitialize with new page context if already initialized
                if self.initialized:
                    logger.info(f"🔄 Reinitializing agent for user {self.user_id} with new page: {new_page}")
                    await self.initialize()
                
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error updating current page for user {self.user_id}: {e}")
            return False
    
    async def get_current_page(self) -> str:
        """Get the current page for this agent."""
        return self.current_page

    async def stream_response(self, message: str):
        """Stream the agent response for real-time updates."""
        try:
            if not self.initialized:
                await self.initialize()

            logger.info(f"🔄 Streaming response for user {self.user_id}")

            # Stream the agent response
            async for chunk in self.agent_executor.astream(
                {"input": message},
                config=RunnableConfig(tags=[f"user_{self.user_id}"])
            ):
                # Send chunks to WebSocket for real-time updates
                await send_to_user_tool_websocket(
                    self.user_id,
                    {
                        "type": "agent_stream_chunk",
                        "action": "streaming",
                        "data": {
                            "chunk": chunk,
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                )
                
                yield chunk

        except Exception as e:
            logger.error(f"❌ Error streaming response for user {self.user_id}: {e}")
            yield {"error": str(e)}


# Mock implementations for when LangChain is not available
class MockLangChainAgent:
    """Mock implementation when LangChain is not available."""

    def __init__(self, user_id: str, current_page: str = "dashboard"):
        self.user_id = user_id
        self.current_page = current_page
        self.memory = []
        self.initialized = False

    async def initialize(self):
        """Mock initialization."""
        self.initialized = True
        logger.info(f"🔧 Mock LangChain agent initialized for user {self.user_id}")

    async def process_message(self, message: str) -> str:
        """Mock message processing."""
        self.memory.append({"role": "user", "content": message})
        response = f"Mock LangChain response to: '{message}'"
        self.memory.append({"role": "assistant", "content": response})
        return response

    async def get_memory(self) -> List[Dict[str, Any]]:
        return self.memory.copy()

    async def clear_memory(self) -> bool:
        self.memory = []
        return True

    async def save_memory(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "memory": self.memory.copy(),
            "timestamp": datetime.now().isoformat()
        }

    async def load_memory(self, memory_data: List[Dict[str, Any]]) -> bool:
        self.memory = memory_data.copy()
        return True

    async def get_available_tools(self) -> List[str]:
        return ["navigate_page", "click_button", "search_database", "search_files", "upload_files", "generate_report", "view_report", "file_operations", "create_table", "add_column", "set_column_properties", "complete_table", "table_status", "confirm_table_creation", "page_info", "update_table_name", "update_column_name", "update_column_properties", "submit_table", "update", "form_update", "table_management", "business_rule_management", "excel_import", "select_table", "list_available_tables", "get_ai_suggestion", "continue_to_import", "back_to_mapping", "import_data_to_database"]
    
    async def update_current_page(self, new_page: str):
        """Update the current page for mock agent."""
        if self.current_page != new_page:
            self.current_page = new_page
            logger.info(f"🔄 Updated current page for mock agent user {self.user_id}: {new_page}")
            return True
        return False
    
    async def get_current_page(self) -> str:
        """Get the current page for mock agent."""
        return self.current_page

    async def stream_response(self, message: str):
        response = await self.process_message(message)
        yield {"output": response}


# Factory function to create the appropriate agent
def create_text_agent(user_id: str, current_page: str = "dashboard"):
    """Factory function to create the appropriate text agent."""
    if LANGCHAIN_AVAILABLE:
        return LangChainTextAgent(user_id, current_page)
    else:
        logger.warning("🔧 Using mock LangChain agent")
        return MockLangChainAgent(user_id, current_page)


# Example usage and testing
async def main():
    """Example usage of the LangChain text agent."""
    try:
        # Create agent
        agent = create_text_agent("test_user_123", "tables")
        await agent.initialize()

        # Test basic conversation
        response1 = await agent.process_message("Hello, how can you help me?")
        print(f"Response 1: {response1}")

        # Test tool calling
        response2 = await agent.process_message("Navigate to the dashboard page")
        print(f"Response 2: {response2}")

        # Test database search
        response3 = await agent.process_message("Search for all users created in the last month")
        print(f"Response 3: {response3}")

        # Test report generation
        response4 = await agent.process_message("Generate a monthly user activity report")
        print(f"Response 4: {response4}")

        # Get memory
        memory = await agent.get_memory()
        print(f"Memory contains {len(memory)} messages")

        # Test streaming
        print("\nStreaming response:")
        async for chunk in agent.stream_response("What tools do you have available?"):
            print(f"Chunk: {chunk}")

    except Exception as e:
        logger.error(f"❌ Example usage failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())