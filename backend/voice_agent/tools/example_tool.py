#!/usr/bin/env python3
"""
Example tool for Pipecat demonstrating best practices.
This shows how to create a tool with function calling and WebSocket communication.
"""

import json
from typing import Dict, Any, Optional
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIServerMessageFrame
from pipecat.frames.frames import Frame, LLMMessagesAppendFrame
from pipecat.adapters.schemas.function_schema import FunctionSchema
from loguru import logger


class ExampleTool:
    """Example tool demonstrating Pipecat tool patterns."""
    
    def __init__(self, rtvi_processor: RTVIProcessor, task=None):
        self.rtvi = rtvi_processor
        self.task = task
        self.current_data = {}  # Store current state
        
    def get_tool_definition(self) -> FunctionSchema:
        """Get the tool definition for the LLM using Pipecat's FunctionSchema."""
        return FunctionSchema(
            name="manage_example",
            description="Create, update, or manage example data. Use when user wants to work with example information.",
            properties={
                "action": {
                    "type": "string",
                    "enum": ["create", "update", "delete", "get"],
                    "description": "The action to perform. ONLY use these exact values: create, update, delete, get"
                },
                "field": {
                    "type": "string",
                    "enum": ["name", "email", "phone", "address"],
                    "description": "The field to update (required when action is update). ONLY use these exact field names."
                },
                "value": {
                    "type": "string",
                    "description": "The value to set for the field (required when action is update)."
                }
            },
            required=["action"]
        )
    
    async def execute(self, action: str, **kwargs) -> str:
        """Execute the example tool action."""
        try:
            # Validate action first
            valid_actions = ["create", "update", "delete", "get"]
            if action not in valid_actions:
                return f"Invalid action '{action}'. Valid actions: {', '.join(valid_actions)}"
            
            # Route to specific handlers
            if action == "create":
                return await self._create_item(**kwargs)
            elif action == "update":
                field = kwargs.get("field")
                value = kwargs.get("value")
                
                # Validate field name
                valid_fields = ["name", "email", "phone", "address"]
                if not field:
                    return "Field name is required for update action"
                
                if field not in valid_fields:
                    return f"Invalid field '{field}'. Valid fields: {', '.join(valid_fields)}"
                
                if value is None:
                    return "Value is required for update action"
                
                return await self._update_field(field, value)
                
            elif action == "delete":
                return await self._delete_item(**kwargs)
            elif action == "get":
                return await self._get_item(**kwargs)
            else:
                return f"Invalid action '{action}'. Valid actions: {', '.join(valid_actions)}"
                
        except Exception as e:
            logger.error(f"Error executing example tool: {e}")
            return f"Error managing example: {str(e)}"
    
    async def _create_item(self, **initial_data) -> str:
        """Create a new item with optional initial data."""
        logger.info("Creating new example item")
        
        # Send command to frontend
        await self._send_client_command("create_example_item", initial_data)
        
        # Initialize current data
        self.current_data = {}
        if initial_data:
            for key, value in initial_data.items():
                if value is not None:
                    self.current_data[key] = value
                    await self._send_client_command("update_example_field", {
                        "field": key,
                        "value": value
                    })
        
        response_parts = ["I've created a new example item"]
        if self.current_data:
            filled_fields = ", ".join(self.current_data.keys())
            response_parts.append(f"with: {filled_fields}")
        response_parts.append(". What else would you like to add?")
        
        return "".join(response_parts)
    
    async def _update_field(self, field: str, value: Any) -> str:
        """Update a specific field."""
        if not field or value is None:
            return "Please specify both a field name and value to update."
        
        logger.info(f"Updating example field: {field} = {value}")
        
        # Store in current data
        self.current_data[field] = value
        
        # Send update to frontend
        await self._send_client_command("update_example_field", {
            "field": field,
            "value": value
        })
        
        return f"Updated {field} to {value}"
    
    async def _delete_item(self, **kwargs) -> str:
        """Delete the current item."""
        logger.info("Deleting example item")
        
        # Send delete command to frontend
        await self._send_client_command("delete_example_item", self.current_data)
        
        # Clear current data
        self.current_data = {}
        
        return "Example item has been deleted successfully!"
    
    async def _get_item(self, **kwargs) -> str:
        """Get the current item data."""
        if not self.current_data:
            return "No example item data available."
        
        data_summary = ", ".join([f"{k}: {v}" for k, v in self.current_data.items()])
        return f"Current example item data: {data_summary}"
    
    async def _send_client_command(self, command: str, data: Dict[str, Any]):
        """Send a command to the frontend client via WebSocket."""
        try:
            # Create command data for the frontend
            command_data = {
                "type": "example_command",
                "action": command,
                **data
            }
            
            # Log the command for debugging
            logger.info(f"🔧 Sending example command to client: {command} with data: {json.dumps(data)}")
            
            # Method 1: Send via tool WebSocket connections (preferred)
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from tool_websocket_registry import tool_websockets
            if tool_websockets:
                sent_count = 0
                for client_id, ws in tool_websockets.items():
                    try:
                        await ws.send_json(command_data)
                        sent_count += 1
                        logger.info(f"✅ Example command sent to tool client {client_id}")
                    except Exception as ws_error:
                        logger.error(f"❌ Failed to send to tool client {client_id}: {ws_error}")
                
                if sent_count > 0:
                    logger.info(f"✅ Example command sent to {sent_count} tool clients")
                    return
                else:
                    logger.error("❌ No tool WebSocket clients available")
            else:
                logger.error("❌ No tool WebSocket connections available")
            
            # Method 2: Fallback to RTVI (less reliable)
            rtvi_frame = RTVIServerMessageFrame(data=command_data)
            
            if self.task:
                await self.task.queue_frames([rtvi_frame])
                logger.info("✅ Example command sent via task queue (fallback)")
            elif self.rtvi:
                await self.rtvi.push_frame(rtvi_frame)
                logger.info("✅ Example command sent via RTVI processor (fallback)")
            else:
                logger.error("❌ No task or RTVI processor available to send command")
            
        except Exception as e:
            logger.error(f"❌ Error sending client command: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")


# Factory function to create the tool
def create_example_tool(rtvi_processor: RTVIProcessor, task=None) -> ExampleTool:
    """Create and return an ExampleTool instance."""
    return ExampleTool(rtvi_processor, task)


# Example usage in main.py:
"""
# In your main.py file:

# 1. Import the tool
from tools.example_tool import create_example_tool

# 2. Create tool instance
example_tool = create_example_tool(rtvi, task)

# 3. Get tool definition for LLM
tool_definition = example_tool.get_tool_definition()

# 4. Add to tools list
tools = [
    example_tool.get_tool_definition(),
    # ... other tools
]

# 5. Create tools schema
tools_schema = ToolsSchema(standard_tools=tools)

# 6. Register function handler
async def handle_function_call(params):
    if params.function_name == "manage_example":
        result = await example_tool.execute(**params.arguments)
        await params.result_callback({"result": result})

llm_service.register_function("manage_example", handle_function_call)

# 7. Set task reference
example_tool.task = task
""" 