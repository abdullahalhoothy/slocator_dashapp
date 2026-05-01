"""MCP client and per-browser client cache.

Combines the LangGraph-backed ``SimpleMCPClient`` with a process-wide cache
keyed by browser session id, so each user's conversation memory stays isolated.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, Optional

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from .config import Config
from .prompts import TERRITORY_OPTIMIZATION_PROMPT


class AnalysisOutput(BaseModel):
    """Structured output for analysis with file handles and data files."""

    report_file: str = Field(description="Path to the generated markdown report file")
    data_files: Optional[Dict[str, str]] = Field(
        description="Dictionary of data files for downstream processing",
        default_factory=dict,
    )
    response: str = Field(description="Human-readable response to be shown in chat box")
    metadata: Optional[Dict[str, Any]] = Field(
        description="Additional metadata about the analysis", default_factory=dict
    )


class SimpleMCPClient:
    """MCP client that orchestrates server-side tools and keeps conversation memory."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        config_override: Optional[dict] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        print("[>>] Initializing Simple MCP Client with Memory...", flush=True)

        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.default_thread_id = f"conversation_{self.session_id}"

        if not Config.validate_paths():
            raise ValueError("Required paths are missing. Please check configuration.")

        self.model = model or Config.DEFAULT_MODEL
        self.temperature = temperature if temperature is not None else Config.DEFAULT_TEMPERATURE

        self.mcp_config = Config.get_mcp_config()
        if config_override:
            self.mcp_config.update(config_override)

        self.checkpointer = MemorySaver()

        self.client: Optional[MultiServerMCPClient] = None
        self.agent = None
        self.tools = None
        self.secrets: Optional[dict] = None

        self.parser = PydanticOutputParser(pydantic_object=AnalysisOutput)

        print(f"[*] Session ID: {self.session_id}", flush=True)

    async def check_server_health(self) -> bool:
        try:
            server_url = self.mcp_config.get(Config.MCP_SERVER_NAME, {}).get("url")
            transport = self.mcp_config.get(Config.MCP_SERVER_NAME, {}).get("transport")

            if not server_url:
                print("[WARNING] No MCP server URL configured", flush=True)
                return False

            if transport != "sse":
                print(f"[WARNING] Transport mode '{transport}' is not SSE.", flush=True)
                return False

            async with httpx.AsyncClient(timeout=Config.MCP_HEALTH_CHECK_TIMEOUT) as client:
                try:
                    response = await client.options(server_url)
                    if response.status_code in (200, 204):
                        print(
                            f"[OK] MCP SSE server reachable: {response.status_code}",
                            flush=True,
                        )
                        return True

                    base_url = server_url.rsplit("/sse", 1)[0] if "/sse" in server_url else server_url
                    response = await client.get(base_url)
                    print(
                        f"[INFO] MCP server base URL reachable: {response.status_code}",
                        flush=True,
                    )
                    return response.status_code < 500
                except httpx.ConnectError:
                    print(f"[ERROR] Cannot connect to MCP SSE server at {server_url}", flush=True)
                    return False
                except httpx.TimeoutException:
                    print(f"[ERROR] MCP SSE server timeout at {server_url}", flush=True)
                    return False
        except Exception as e:
            print(f"[WARNING] MCP server health check failed: {str(e)}", flush=True)
            return False

    async def connect(
        self, max_retries: Optional[int] = None, retry_delay: Optional[float] = None
    ) -> None:
        max_retries = max_retries if max_retries is not None else Config.MCP_MAX_RETRIES
        retry_delay = retry_delay if retry_delay is not None else Config.MCP_RETRY_DELAY_SECONDS

        print("[>>] Connecting to MCP server...", flush=True)
        print(f"[INFO] Transport: {Config.MCP_TRANSPORT}", flush=True)
        print(
            f"[INFO] URL: {self.mcp_config.get(Config.MCP_SERVER_NAME, {}).get('url', 'N/A')}",
            flush=True,
        )

        if not await self.check_server_health():
            print("[WARNING] Health check failed; attempting connection anyway.", flush=True)

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    print(f"[RETRY] Connection attempt {attempt}/{max_retries}", flush=True)
                    await asyncio.sleep(retry_delay)

                self.client = MultiServerMCPClient(self.mcp_config)
                self.tools = await self.client.get_tools()
                print(f"[*] Available tools: {[t.name for t in self.tools]}", flush=True)

                if not self.secrets:
                    secrets_path = Config.SECRETS_DIR / "secrets_llm.json"
                    with open(secrets_path, "r") as f:
                        self.secrets = json.load(f)
                    api_key = self.secrets["gemini_api_key"]
                    print(f"[KEY] Loaded Gemini API key: {api_key[:20]}...{api_key[-4:]}", flush=True)

                llm = ChatGoogleGenerativeAI(
                    model=self.model,
                    temperature=self.temperature,
                    google_api_key=self.secrets["gemini_api_key"],
                )
                self.agent = create_react_agent(llm, self.tools, checkpointer=self.checkpointer)
                print("[OK] MCP client connected via SSE with memory", flush=True)
                return

            except Exception as e:
                last_error = e
                print(f"[ERROR] Connection attempt {attempt}/{max_retries} failed: {str(e)}", flush=True)

        server_url = self.mcp_config.get(Config.MCP_SERVER_NAME, {}).get("url", "unknown URL")
        raise ConnectionError(
            f"Failed to connect to MCP SSE server at {server_url} after "
            f"{max_retries} attempts: {last_error}"
        ) from last_error

    def _refresh_agent(self) -> None:
        """Recreate the LLM/agent (preserving the checkpointer) to dodge stale event loops."""
        if not self.tools or not self.secrets:
            raise ValueError("MCP client must be connected first. Call connect() method.")
        llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=self.temperature,
            google_api_key=self.secrets["gemini_api_key"],
        )
        self.agent = create_react_agent(llm, self.tools, checkpointer=self.checkpointer)

    async def analyze_territories(self, user_query: str, thread_id: Optional[str] = None) -> str:
        if not self.tools or not self.secrets:
            raise ValueError("Agent not connected. Please call connect() first.")

        self._refresh_agent()
        current_thread_id = thread_id or self.default_thread_id
        config = {"configurable": {"thread_id": current_thread_id}}

        messages = [
            SystemMessage(content=TERRITORY_OPTIMIZATION_PROMPT),
            HumanMessage(content=user_query),
        ]

        print(f"🔄 Processing query: {user_query[:100]}...")
        print(f"🧠 Using thread: {current_thread_id}")

        response = await self.agent.ainvoke({"messages": messages}, config=config)
        return self._extract_final_response(response)

    async def analyze_territories_with_file_handle(
        self, user_query: str, thread_id: Optional[str] = None
    ) -> dict:
        if not self.tools or not self.secrets:
            raise ValueError("Agent not connected. Please call connect() first.")

        self._refresh_agent()
        current_thread_id = thread_id or self.default_thread_id
        config = {"configurable": {"thread_id": current_thread_id}}

        enhanced_system_prompt = (
            f"{TERRITORY_OPTIMIZATION_PROMPT}\n\n"
            "IMPORTANT: You must respond with a valid JSON object in the exact format specified below.\n"
            f"{self.parser.get_format_instructions()}\n\n"
            "Make sure to include:\n"
            "- report_file: The actual file path to the generated report\n"
            "- data_files: Dictionary of any data files created for downstream processing\n"
            "- response: A clear, human-readable summary of what was accomplished\n"
            "- metadata: Any additional relevant information about the analysis"
        )

        messages = [
            SystemMessage(content=enhanced_system_prompt),
            HumanMessage(content=user_query),
        ]

        print(f"🔄 Processing query: {user_query[:100]}...")
        print(f"🧠 Using thread: {current_thread_id}")

        response = await self.agent.ainvoke({"messages": messages}, config=config)
        raw_response = self._extract_final_response(response)

        try:
            structured_output = self.parser.parse(raw_response)
            print("✅ Successfully parsed structured output")
            print(f"📄 Report file: {structured_output.report_file}")
            return {
                "response": structured_output.response,
                "raw_content": raw_response,
                "structured_output": structured_output,
            }
        except Exception as e:
            print(f"⚠️ Failed to parse structured output: {e}")
            return {
                "response": raw_response,
                "raw_content": raw_response,
                "structured_output": None,
            }

    def _extract_final_response(self, response) -> str:
        if not (isinstance(response, dict) and "messages" in response):
            return "[OK] Analysis completed successfully."

        for message in reversed(response["messages"]):
            if "AI" not in str(getattr(message, "__class__", "")):
                continue
            content = getattr(message, "content", None)
            if not content:
                continue
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                    elif isinstance(item, str):
                        text_parts.append(item)
                    elif item:
                        text_parts.append(str(item))
                content = " ".join(text_parts)
            if isinstance(content, str) and content.strip():
                return content

        return "[OK] Territory analysis completed! Reports have been generated and saved by the system."

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client and hasattr(self.client, "close"):
            try:
                await self.client.close()
            except Exception as e:
                print(f"[WARNING] Error during client cleanup: {e}", flush=True)


# ===== Per-browser client cache =====================================================

_mcp_clients: Dict[str, SimpleMCPClient] = {}


def get_or_create_client(browser_id: str) -> SimpleMCPClient:
    """Return the MCP client for this browser, creating it on first use."""
    client = _mcp_clients.get(browser_id)
    if client is None:
        print(f"[>>] Creating MCP client for browser {browser_id}", flush=True)
        client = SimpleMCPClient(session_id=browser_id)
        _mcp_clients[browser_id] = client
    return client


async def ensure_client_connected(browser_id: str) -> SimpleMCPClient:
    """Return a connected MCP client for this browser."""
    client = get_or_create_client(browser_id)
    if not client.agent:
        print(f"[>>] Connecting MCP client for browser {browser_id}", flush=True)
        await client.connect()
    return client


def reset_client(browser_id: Optional[str] = None) -> None:
    """Drop a single browser's client, or all clients when browser_id is None."""
    if browser_id is None:
        _mcp_clients.clear()
        print("[INFO] All MCP clients reset", flush=True)
    else:
        _mcp_clients.pop(browser_id, None)
        print(f"[INFO] MCP client reset for browser {browser_id}", flush=True)


def get_thread_id(browser_id: str) -> str:
    """Return the LangGraph thread id used for this browser's conversation memory."""
    return f"thread_{browser_id}"