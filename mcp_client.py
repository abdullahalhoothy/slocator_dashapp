"""
MCP Client Module

Handles Model Context Protocol (MCP) client initialization and management
for the Dash application. Provides a clean interface for connecting to
the MCP SSE server and managing conversation state.
"""

import asyncio
from typing import Optional
from report_agent import SimpleMCPClient


# Global MCP client instance for memory persistence
_mcp_client: Optional[SimpleMCPClient] = None

# Thread ID for Dash conversation continuity
DASH_THREAD_ID = "dash_conversation_main"


def get_or_create_client() -> SimpleMCPClient:
    """
    Get existing MCP client or create a new one (lazy initialization)

    This function ensures we have a single persistent MCP client instance
    that maintains conversation memory across multiple requests.

    Returns:
        SimpleMCPClient: The global MCP client instance
    """
    global _mcp_client

    if _mcp_client is None:
        print("[>>] Creating new MCP client with memory...", flush=True)
        # Create client with specific session for Dash app
        _mcp_client = SimpleMCPClient(session_id="dash_session")
        print("[OK] MCP client created, will connect on first use", flush=True)

    return _mcp_client


async def ensure_client_connected() -> SimpleMCPClient:
    """
    Ensure the MCP client is connected to the SSE server

    This function checks if the client is connected and connects it if needed.
    It handles the async connection process transparently.

    Returns:
        SimpleMCPClient: Connected MCP client instance

    Raises:
        ConnectionError: If connection to MCP SSE server fails
    """
    client = get_or_create_client()

    if client and not client.agent:
        print("[>>] Connecting MCP client...", flush=True)
        await client.connect()
        print("[OK] MCP client connected with memory!", flush=True)

    return client


def reset_client():
    """
    Reset the global MCP client instance

    Useful for testing or when you need to force a fresh connection.
    The client will be recreated on next call to get_or_create_client().
    """
    global _mcp_client
    _mcp_client = None
    print("[INFO] MCP client reset", flush=True)


def get_thread_id() -> str:
    """
    Get the current thread ID for conversation continuity

    Returns:
        str: The thread ID used for maintaining conversation context
    """
    return DASH_THREAD_ID