#!/usr/bin/env python3
"""
User-specific registry for tool WebSocket connections.
Supports multiple users simultaneously.
"""

from typing import Dict, Optional
from fastapi import WebSocket
import asyncio
from loguru import logger

# User-specific store for tool WebSocket connections
# Structure: {user_id: websocket}
user_tool_websockets: Dict[str, WebSocket] = {}

# User-specific registry for product info websocket clients
# Structure: {user_id: {"websocket": websocket, "last_message": str}}
user_product_info_clients: Dict[str, Dict] = {}

# Session management for user tracking
# Structure: {session_id: user_id}
session_user_mapping: Dict[str, str] = {}


def register_tool_websocket(user_id: str, websocket: WebSocket) -> None:
    """Register a tool websocket for a specific user."""
    user_tool_websockets[user_id] = websocket
    logger.info(f"🔧 Registered tool websocket for user: {user_id}")


def unregister_tool_websocket(user_id: str) -> None:
    """Unregister a tool websocket for a specific user."""
    if user_id in user_tool_websockets:
        del user_tool_websockets[user_id]
        logger.info(f"🧹 Unregistered tool websocket for user: {user_id}")


def get_tool_websocket(user_id: str) -> Optional[WebSocket]:
    """Get the tool websocket for a specific user."""
    return user_tool_websockets.get(user_id)


def register_product_info_client(user_id: str, websocket: WebSocket) -> None:
    """Register a product info websocket client for a specific user."""
    # Close existing connection if any
    if user_id in user_product_info_clients:
        existing_ws = user_product_info_clients[user_id].get("websocket")
        if existing_ws:
            try:
                asyncio.create_task(existing_ws.close(code=1000, reason="New client connection"))
                logger.info(f"🔄 Closed existing product info connection for user: {user_id}")
            except Exception:
                logger.warning(f"⚠️ Could not close existing connection for user: {user_id}")
    
    user_product_info_clients[user_id] = {
        "websocket": websocket,
        "last_message": None
    }
    logger.info(f"🔗 Registered product info client for user: {user_id}")
    logger.info(f"🔗 Total product info clients: {len(user_product_info_clients)}")
    logger.info(f"🔗 Current product info clients: {list(user_product_info_clients.keys())}")


def unregister_product_info_client(user_id: str) -> None:
    """Unregister a product info websocket client for a specific user."""
    if user_id in user_product_info_clients:
        del user_product_info_clients[user_id]
        logger.info(f"🧹 Unregistered product info client for user: {user_id}")


def get_product_info_client(user_id: str) -> Optional[Dict]:
    """Get the product info client data for a specific user."""
    return user_product_info_clients.get(user_id)


def set_product_info_last_message(user_id: str, message: str) -> None:
    """Set the last message for a specific user's product info client."""
    if user_id in user_product_info_clients:
        user_product_info_clients[user_id]["last_message"] = message


def register_session_user(session_id: str, user_id: str) -> None:
    """Register a session with a user ID."""
    session_user_mapping[session_id] = user_id
    logger.info(f"🆔 Registered session {session_id} for user: {user_id}")


def unregister_session_user(session_id: str) -> None:
    """Unregister a session."""
    if session_id in session_user_mapping:
        user_id = session_user_mapping[session_id]
        del session_user_mapping[session_id]
        logger.info(f"🧹 Unregistered session {session_id} for user: {user_id}")


def get_user_from_session(session_id: str) -> Optional[str]:
    """Get the user ID from a session ID."""
    return session_user_mapping.get(session_id)


def get_all_users() -> list:
    """Get all active user IDs."""
    # Get users from all sources: sessions, tool websockets, and product info clients
    session_users = list(session_user_mapping.values())
    tool_users = list(user_tool_websockets.keys())
    product_info_users = list(user_product_info_clients.keys())
    
    all_users = list(set(session_users + tool_users + product_info_users))
    logger.debug(f"[Registry] All users - Sessions: {session_users}, Tools: {tool_users}, ProductInfo: {product_info_users}, Combined: {all_users}")
    return all_users


async def send_to_user_tool_websocket(user_id: str, data: dict) -> bool:
    """Send data to a specific user's tool websocket."""
    websocket = get_tool_websocket(user_id)
    if websocket:
        try:
            await websocket.send_json(data)
            logger.info(f"✅ Sent data to user {user_id} tool websocket")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to send data to user {user_id} tool websocket: {e}")
            unregister_tool_websocket(user_id)
            return False
    else:
        logger.warning(f"⚠️ No tool websocket found for user: {user_id}")
        return False


async def send_to_user_product_info(user_id: str, message: str) -> bool:
    """Send message to a specific user's product info websocket."""
    client_data = get_product_info_client(user_id)
    if client_data and client_data["websocket"]:
        try:
            await client_data["websocket"].send_text(message)
            set_product_info_last_message(user_id, message)
            logger.info(f"✅ Sent product info to user {user_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to send product info to user {user_id}: {e}")
            unregister_product_info_client(user_id)
            return False
    else:
        logger.warning(f"⚠️ No product info client found for user: {user_id}")
        return False


# Backward compatibility (deprecated - will be removed)
tool_websockets = user_tool_websockets  # For backward compatibility
product_info_ws_client = {"websocket": None}  # For backward compatibility