"""Authentication helpers: Firebase login, token refresh, per-browser session files.

The browser session id is an opaque UUID stored in the Flask signed cookie.
Auth tokens themselves live on disk under ``.sessions/{browser_id}_auth.json``
so the cookie never carries refreshable credentials.
"""

import asyncio
import json
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
from flask import session as flask_session

from .config import Config
from .mcp_client import ensure_client_connected, reset_client


def get_browser_id() -> str:
    """Return the current browser's session UUID. Callable inside any Dash
    callback because Dash callbacks run inside a Flask request context."""
    return flask_session.get("browser_id", "anonymous")


def auth_file_path(browser_id: str) -> Path:
    return Path(Config.get_session_file_path(f"{browser_id}_auth.json"))


def _write_auth_file(browser_id: str, auth_data: dict) -> None:
    with open(auth_file_path(browser_id), "w", encoding="utf-8") as f:
        json.dump(auth_data, f, indent=2)


async def authenticate_user_direct(email: str, password: str) -> dict:
    """POST credentials to the FastAPI ``/login`` endpoint.

    Returns ``{"success": True, "data": ..., "user_email": ...}`` on success or
    ``{"success": False, "error": ...}`` on failure.
    """
    try:
        endpoint_url = Config.BACKEND_URL + Config.LOGIN_ENDPOINT
        payload = {
            "message": "login request from dash app",
            "request_info": {},
            "request_body": {"email": email, "password": password},
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint_url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return {"success": False, "error": f"Login failed: {error_text}"}
                response_json = await response.json()
                login_data = response_json.get("data")
                if not login_data:
                    return {"success": False, "error": "Invalid response format from server"}
                return {"success": True, "data": login_data, "user_email": email}

    except Exception as e:
        return {"success": False, "error": f"Network error: {str(e)}"}


async def refresh_id_token(refresh_token: str) -> dict:
    """Exchange a refresh token for a fresh Firebase id token via the backend.

    Returns ``{"success": True, "id_token": ..., "expires_in": ..., "refresh_token": ...}``
    on success, ``{"success": False, "error": ...}`` on failure.
    """
    try:
        endpoint_url = Config.BACKEND_URL + Config.REFRESH_ENDPOINT
        payload = {
            "message": "token refresh from dash app",
            "request_info": {},
            "request_body": {"grant_type": "refresh_token", "refresh_token": refresh_token},
        }
        async with aiohttp.ClientSession() as http:
            async with http.post(endpoint_url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return {"success": False, "error": f"Refresh failed: {error_text}"}
                response_json = await response.json()
                refresh_data = response_json.get("data") or {}
                new_id_token = refresh_data.get("id_token") or refresh_data.get("idToken")
                expires_in = refresh_data.get("expires_in") or refresh_data.get("expiresIn")
                if not new_id_token or not expires_in:
                    return {"success": False, "error": f"Unexpected refresh payload: {response_json}"}
                return {
                    "success": True,
                    "id_token": new_id_token,
                    "expires_in": int(expires_in),
                    "refresh_token": refresh_data.get("refresh_token")
                    or refresh_data.get("refreshToken")
                    or refresh_token,
                }
    except Exception as e:
        return {"success": False, "error": f"Refresh network error: {str(e)}"}


async def update_mcp_session_auth(
    user_id: str, id_token: str, refresh_token: str, expires_in: int
) -> bool:
    """Persist authentication tokens for the current browser and warm up its MCP client."""
    try:
        browser_id = get_browser_id()
        await ensure_client_connected(browser_id)

        print(f"[AUTH] Storing auth tokens for browser {browser_id} (user: {user_id})", flush=True)

        token_buffer = Config.TOKEN_REFRESH_BUFFER_SECONDS
        auth_data = {
            "session_id": browser_id,
            "user_id": user_id,
            "id_token": id_token,
            "refresh_token": refresh_token,
            "token_expires_at": (
                datetime.now() + timedelta(seconds=expires_in - token_buffer)
            ).isoformat(),
            "created_at": datetime.now().isoformat(),
            "expires_at": (
                datetime.now() + timedelta(hours=Config.SESSION_DURATION_HOURS)
            ).isoformat(),
        }

        _write_auth_file(browser_id, auth_data)
        print(f"[OK] Stored auth tokens for browser {browser_id}", flush=True)
        return True

    except Exception as e:
        print(f"[ERROR] Failed to store session auth: {str(e)}", flush=True)
        traceback.print_exc()
        return False


async def logout_user() -> bool:
    """Remove the current browser's auth file and drop its MCP client."""
    try:
        browser_id = get_browser_id()
        metadata_path = auth_file_path(browser_id)
        if metadata_path.exists():
            metadata_path.unlink()
            print(f"[LOGOUT] Removed auth tokens for browser {browser_id}", flush=True)
        else:
            print(f"[LOGOUT] No auth tokens for browser {browser_id}", flush=True)
        reset_client(browser_id)
        return True

    except Exception as e:
        print(f"[ERROR] Failed to logout user: {str(e)}", flush=True)
        return False


def _attempt_token_refresh(browser_id: str, auth_data: dict) -> bool:
    """Refresh the id token (in place) and rewrite the file. Returns True on success."""
    refresh_token = auth_data.get("refresh_token")
    if not refresh_token:
        return False

    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("event loop closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    result = loop.run_until_complete(refresh_id_token(refresh_token))
    if not result.get("success"):
        print(f"[AUTH] Token refresh failed: {result.get('error')}", flush=True)
        return False

    token_buffer = Config.TOKEN_REFRESH_BUFFER_SECONDS
    auth_data["id_token"] = result["id_token"]
    auth_data["refresh_token"] = result["refresh_token"]
    auth_data["token_expires_at"] = (
        datetime.now() + timedelta(seconds=result["expires_in"] - token_buffer)
    ).isoformat()
    _write_auth_file(browser_id, auth_data)
    print(f"[AUTH] Refreshed id token for browser {browser_id}", flush=True)
    return True


def get_current_auth_status() -> dict:
    """Return ``{"authenticated": bool, ...}`` for the current browser session.

    Refreshes the id token transparently when it has expired but the configured
    session window is still open.
    """
    try:
        browser_id = get_browser_id()
        metadata_path = auth_file_path(browser_id)
        if not metadata_path.exists():
            return {"authenticated": False}

        with open(metadata_path, "r", encoding="utf-8") as f:
            auth_data = json.load(f)

        user_id = auth_data.get("user_id")
        token_expires_at = auth_data.get("token_expires_at")
        session_expires_at = auth_data.get("expires_at")
        if not (user_id and token_expires_at and session_expires_at):
            return {"authenticated": False}

        now = datetime.now()
        if now >= datetime.fromisoformat(session_expires_at):
            # Hard session window elapsed — user must re-login.
            return {"authenticated": False}

        if now >= datetime.fromisoformat(token_expires_at):
            if not _attempt_token_refresh(browser_id, auth_data):
                return {"authenticated": False}
            token_expires_at = auth_data["token_expires_at"]

        return {
            "authenticated": True,
            "user_id": user_id,
            "expires_at": token_expires_at,
        }

    except Exception as e:
        print(f"[ERROR] Failed to get auth status: {str(e)}", flush=True)
        return {"authenticated": False}