#!/usr/bin/env python3
"""
Agent switcher tool for Pipecat.
Allows switching between different specialized agents (invoice, request money, quotation).
"""

import json
from typing import Dict, Any, Optional, Callable
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIServerMessageFrame
from pipecat.frames.frames import Frame, LLMMessagesAppendFrame
from pipecat.adapters.schemas.function_schema import FunctionSchema
from loguru import logger


class AgentSwitcherTool:
    """Tool for switching between different specialized agents."""
    
    def __init__(self, rtvi_processor: RTVIProcessor, agent_switch_callback: Optional[Callable] = None, task=None):
        self.rtvi = rtvi_processor
        self.agent_switch_callback = agent_switch_callback
        self.task = task
        
    def get_tool_definition(self) -> FunctionSchema:
        """Get the tool definition for the LLM using Pipecat's FunctionSchema."""
        return FunctionSchema(
            name="switch_agent",
            description="Switch to a different specialized agent when user requests a different form type. Use when user wants to work with a different form than the current agent handles.",
            properties={
                "agent_type": {
                    "type": "string",
                    "enum": ["invoice", "request_money", "quotation"],
                    "description": "The type of agent to switch to. ONLY use these exact values: invoice, request_money, quotation"
                },
                "user_request": {
                    "type": "string",
                    "description": "The original user request that triggered the agent switch"
                }
            },
            required=["agent_type", "user_request"]
        )
    
    async def execute(self, agent_type: str, user_request: str, **kwargs) -> str:
        """Execute the agent switch."""
        try:
            # Validate agent type
            valid_agents = ["invoice", "request_money", "quotation"]
            if agent_type not in valid_agents:
                return f"Invalid agent type '{agent_type}'. Only these agents are available: {', '.join(valid_agents)}"
            
            logger.info(f"Switching to {agent_type} agent with request: {user_request}")
            
            # Call the agent switch callback if provided
            if self.agent_switch_callback:
                await self.agent_switch_callback(agent_type, user_request)
                
                # Map agent types to friendly names
                agent_names = {
                    "invoice": "Invoice Management",
                    "request_money": "Payment Request",
                    "quotation": "Quotation Management"
                }
                
                return f"Switched to {agent_names.get(agent_type, agent_type)} agent. Processing your request: {user_request}"
            else:
                return f"Agent switching not configured. Cannot switch to {agent_type} agent."
                
        except Exception as e:
            logger.error(f"Error switching agent: {e}")
            return f"Error switching agent: {str(e)}"


# Factory function to create the tool
def create_agent_switcher_tool(rtvi_processor: RTVIProcessor, agent_switch_callback: Optional[Callable] = None, task=None) -> AgentSwitcherTool:
    """Create and return an AgentSwitcherTool instance."""
    return AgentSwitcherTool(rtvi_processor, agent_switch_callback, task)