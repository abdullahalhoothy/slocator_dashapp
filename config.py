import json
from pathlib import Path

_config_path = Path(__file__).resolve().parent / "config.json"
with open(_config_path, "r") as _f:
    _cfg = json.load(_f)


class AgentConfig:
    """Configuration for the Geospatial Intelligence Analyst Agent, loaded from config.json"""

    PROJECT_ROOT = Path(__file__).resolve().parent

    # ===== DIRECTORY PATHS =====
    REPORTS_DIR       = PROJECT_ROOT / _cfg["directories"]["reports_dir"]
    STATIC_DIR        = PROJECT_ROOT / _cfg["directories"]["static_dir"]
    STATIC_PLOTS_DIR  = PROJECT_ROOT / _cfg["directories"]["static_plots_dir"]
    STATIC_DATA_DIR   = PROJECT_ROOT / _cfg["directories"]["static_data_dir"]
    SESSIONS_DIR      = PROJECT_ROOT / _cfg["directories"]["sessions_dir"]
    SECRETS_DIR       = PROJECT_ROOT / _cfg["directories"]["secrets_dir"]

    # ===== URL PATHS =====
    STATIC_URL_PATH  = _cfg["url_paths"]["static_url_path"]
    STATIC_PLOTS_URL = _cfg["url_paths"]["static_plots_url"]
    STATIC_DATA_URL  = _cfg["url_paths"]["static_data_url"]

    # ===== MODEL SETTINGS =====
    DEFAULT_MODEL       = _cfg["llm"]["default_model"]
    DEFAULT_TEMPERATURE = _cfg["llm"]["default_temperature"]

    # ===== MCP SETTINGS =====
    MCP_SERVER_NAME = _cfg["mcp"]["server_name"]
    MCP_TRANSPORT   = _cfg["mcp"]["transport"]
    MCP_SERVER_URL  = _cfg["mcp"]["server_url"]
    MCP_MAX_RETRIES         = _cfg["mcp"]["max_retries"]
    MCP_RETRY_DELAY_SECONDS = _cfg["mcp"]["retry_delay_seconds"]
    MCP_HEALTH_CHECK_TIMEOUT = _cfg["mcp"]["health_check_timeout_seconds"]

    # ===== BACKEND SETTINGS =====
    BACKEND_URL     = _cfg["backend"]["url"]
    LOGIN_ENDPOINT  = _cfg["backend"]["login_endpoint"]

    # ===== APP SETTINGS =====
    APP_HOST             = _cfg["app"]["host"]
    APP_PORT             = _cfg["app"]["port"]
    APP_DEBUG            = _cfg["app"]["debug"]
    FLASK_SECRET_KEY     = _cfg["app"]["flask_secret_key"]

    # ===== SESSION SETTINGS =====
    DASH_SESSION_ID              = _cfg["session"]["dash_session_id"]
    DASH_THREAD_ID               = _cfg["session"]["dash_thread_id"]
    SESSION_DURATION_HOURS       = _cfg["session"]["session_duration_hours"]
    TOKEN_REFRESH_BUFFER_SECONDS = _cfg["session"]["token_refresh_buffer_seconds"]

    # ===== FILE PATTERNS =====
    SUPPORTED_REPORT_FORMATS = _cfg["file_patterns"]["supported_report_formats"]
    DEFAULT_REPORT_TYPE      = _cfg["file_patterns"]["default_report_type"]
    REPORT_FILE_PATTERNS     = {fmt.lstrip("."): f"*{fmt}" for fmt in _cfg["file_patterns"]["supported_report_formats"]}
    GEOJSON_PATTERN          = _cfg["file_patterns"]["geojson_pattern"]
    PLOT_IMAGE_EXTENSIONS    = _cfg["file_patterns"]["plot_image_extensions"]

    @classmethod
    def get_mcp_config(cls) -> dict:
        """Get MCP client configuration for SSE transport"""
        config = {
            cls.MCP_SERVER_NAME: {
                "url": cls.MCP_SERVER_URL,
                "transport": cls.MCP_TRANSPORT,
            }
        }
        if cls.MCP_TRANSPORT == "sse":
            if not cls.MCP_SERVER_URL.startswith(("http://", "https://")):
                raise ValueError(f"SSE transport requires HTTP(S) URL, got: {cls.MCP_SERVER_URL}")
            print(f"[MCP] SSE transport configured: {cls.MCP_SERVER_URL}", flush=True)
        return config

    @classmethod
    def validate_paths(cls) -> bool:
        if not cls.PROJECT_ROOT.exists():
            print(f"[X] Project root does not exist: {cls.PROJECT_ROOT}", flush=True)
            return False

        for directory, name in [
            (cls.REPORTS_DIR,      "Reports"),
            (cls.STATIC_DIR,       "Static files"),
            (cls.STATIC_PLOTS_DIR, "Static plots"),
            (cls.STATIC_DATA_DIR,  "Static data"),
            (cls.SESSIONS_DIR,     "Sessions"),
        ]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"[OK] {name} directory ready: {directory}", flush=True)
            except Exception as e:
                print(f"[X] Failed to create {name} directory: {e}", flush=True)
                return False

        print("[OK] All required paths validated successfully", flush=True)
        print(f"[INFO] MCP Server URL: {cls.MCP_SERVER_URL}", flush=True)
        return True

    @classmethod
    def get_reports_path(cls) -> str:
        return str(cls.REPORTS_DIR)

    @classmethod
    def get_static_dir(cls) -> str:
        return str(cls.STATIC_DIR)

    @classmethod
    def get_static_plots_dir(cls) -> str:
        return str(cls.STATIC_PLOTS_DIR)

    @classmethod
    def get_static_data_dir(cls) -> str:
        return str(cls.STATIC_DATA_DIR)

    @classmethod
    def get_sessions_dir(cls) -> str:
        return str(cls.SESSIONS_DIR)

    @classmethod
    def get_secrets_dir(cls) -> str:
        return str(cls.SECRETS_DIR)

    @classmethod
    def get_report_file_path(cls, filename: str) -> str:
        return str(cls.REPORTS_DIR / filename)

    @classmethod
    def get_session_file_path(cls, filename: str) -> str:
        return str(cls.SESSIONS_DIR / filename)

    @classmethod
    def is_valid_report_file(cls, filename: str) -> bool:
        return any(filename.endswith(ext) for ext in cls.SUPPORTED_REPORT_FORMATS)


Config = AgentConfig