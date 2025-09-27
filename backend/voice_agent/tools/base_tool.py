#!/usr/bin/env python3
"""
Base tool class for all tools to follow the same WebSocket communication pattern.
"""

import json
from typing import Dict, Any, Optional
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIServerMessageFrame
from pipecat.frames.frames import Frame, LLMMessagesAppendFrame
from pipecat.adapters.schemas.function_schema import FunctionSchema
from loguru import logger


class BaseTool:
    """Base tool class with standard WebSocket communication."""
    
    def __init__(self, rtvi_processor: RTVIProcessor, task=None):
        self.rtvi = rtvi_processor
        self.task = task
        
    def get_tool_definition(self) -> FunctionSchema:
        """Get the tool definition for the LLM using Pipecat's FunctionSchema."""
        raise NotImplementedError("Subclasses must implement get_tool_definition")
    
    async def execute(self, **kwargs) -> str:
        """Execute the tool action."""
        raise NotImplementedError("Subclasses must implement execute")
    
    async def send_websocket_message(self, message_type: str, action: str, data: Dict[str, Any]) -> bool:
        """Send a message to all connected WebSocket clients."""
        try:
            # Create command data for the frontend
            command_data = {
                "type": message_type,
                "action": action,
                "data": data
            }
            
            # Log the command for debugging
            logger.info(f"🔧 Sending {message_type} message: {json.dumps(command_data)}")
            
            # Method 1: Send via tool WebSocket connections using the registry
            from voice_agent.tool_websocket_registry import get_tool_websocket, user_tool_websockets
            
            # Only send to users who actually have tool WebSocket connections
            tool_users = list(user_tool_websockets.keys())
            sent_count = 0
            
            logger.info(f"🔧 Available tool WebSocket users: {tool_users}")
            logger.info(f"🔧 Total tool WebSocket connections: {len(user_tool_websockets)}")
            logger.info(f"🔧 Tool WebSocket registry contents: {user_tool_websockets}")
            
            for user_id in tool_users:
                try:
                    websocket = get_tool_websocket(user_id)
                    if websocket:
                        await websocket.send_json(command_data)
                        sent_count += 1
                        logger.info(f"✅ {message_type} message sent to user {user_id}")
                    else:
                        logger.warning(f"⚠️ No WebSocket found for user {user_id}")
                except Exception as ws_error:
                    logger.error(f"❌ Failed to send {message_type} to user {user_id}: {ws_error}")
            
            if sent_count > 0:
                logger.info(f"✅ {message_type} message sent to {sent_count} users via WebSocket")
                return True
            else:
                logger.info(f"ℹ️ No tool WebSocket clients available for {message_type}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error sending {message_type} message: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def send_websocket_message_with_fallback(self, message_type: str, action: str, data: Dict[str, Any]) -> bool:
        """Send a message to WebSocket clients with RTVI fallback."""
        try:
            # Try WebSocket first
            success = await self.send_websocket_message(message_type, action, data)
            if success:
                return True
            
            # Fallback to RTVI
            command_data = {
                "type": message_type,
                "action": action,
                "data": data
            }
            
            rtvi_frame = RTVIServerMessageFrame(data=command_data)
            
            if self.task:
                await self.task.queue_frames([rtvi_frame])
                logger.info(f"✅ {message_type} message sent via task queue (fallback)")
                return True
            elif self.rtvi:
                await self.rtvi.push_frame(rtvi_frame)
                logger.info(f"✅ {message_type} message sent via RTVI processor (fallback)")
                return True
            else:
                logger.info(f"ℹ️ No task or RTVI processor available for {message_type}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending {message_type} message with fallback: {e}")
            return False


# Example usage for other tools:
"""
class MyCustomTool(BaseTool):
    def get_tool_definition(self) -> FunctionSchema:
        return FunctionSchema(
            name="my_custom_tool",
            description="My custom tool description",
            properties={
                "parameter": {
                    "type": "string",
                    "description": "Parameter description"
                }
            },
            required=["parameter"]
        )
    
    async def execute(self, parameter: str, **kwargs) -> str:
        # Your tool logic here
        result = f"Processed: {parameter}"
        
        # Send result via WebSocket
        await self.send_websocket_message(
            message_type="my_custom_result",
            action="process",
            data={
                "parameter": parameter,
                "result": result,
                "status": "completed"
            }
        )
        
        return result
"""
