"""
Configuration file for Geospatial Intelligence Analyst Agent

Centralized configuration for all paths, directories, and settings
used across the Dash application.
"""

import os
from pathlib import Path

class AgentConfig:
    """Configuration class for the Geospatial Analysis Agent"""

    # ===== BASE PATHS =====
    # Base project path (DashApp directory itself)
    PROJECT_ROOT = Path(__file__).resolve().parent

    # ===== DIRECTORY PATHS =====
    # Reports directory - reads MCP markdown reports
    # When running in Docker: ./reports -> /app/reports (DashApp container)
    # When running on host: D:\slocator\reports
    REPORTS_DIR = PROJECT_ROOT / "reports"

    # Static files directory - reads backend static files (plots, data)
    # When running in Docker: ./static -> /app/static (DashApp container)
    # When running on host: D:\slocator\static
    STATIC_DIR = PROJECT_ROOT / "static"
    STATIC_PLOTS_DIR = STATIC_DIR / "plots"
    STATIC_DATA_DIR = STATIC_DIR / "reports"  # Backend HTML reports are in static/reports/

    # Sessions directory (for local auth storage)
    SESSIONS_DIR = PROJECT_ROOT / ".sessions"

    # Secrets directory (API keys, credentials)
    SECRETS_DIR = PROJECT_ROOT / "secrets"

    # ===== URL PATHS (for web serving) =====
    STATIC_URL_PATH = "/static"
    STATIC_PLOTS_URL = "/static/plots"
    STATIC_DATA_URL = "/static/data"

    # ===== MODEL SETTINGS =====
    DEFAULT_MODEL = "gemini-2.5-flash"
    DEFAULT_TEMPERATURE = 0

    # ===== MCP SETTINGS =====
    MCP_SERVER_NAME = "saudi-location-intelligence"
    MCP_TRANSPORT = "sse"
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001/sse")
    
    
    
    # ===== FILE PATTERNS =====
    # Report file patterns
    SUPPORTED_REPORT_FORMATS = ['.md', '.html']
    DEFAULT_REPORT_TYPE = 'md'
    REPORT_FILE_PATTERNS = {
        'md': '*.md',
        'html': '*.html'
    }

    # GeoJSON and data file patterns
    GEOJSON_PATTERN = "*.geojson"
    PLOT_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.svg', '.gif']
    
    @classmethod
    def get_mcp_config(cls) -> dict:
        """Get MCP client configuration for SSE transport"""
        config = {
            cls.MCP_SERVER_NAME: {
                "url": cls.MCP_SERVER_URL,
                "transport": cls.MCP_TRANSPORT
            }
        }

        # Validate SSE configuration
        if cls.MCP_TRANSPORT == "sse":
            if not cls.MCP_SERVER_URL.startswith(("http://", "https://")):
                raise ValueError(f"SSE transport requires HTTP(S) URL, got: {cls.MCP_SERVER_URL}")
            print(f"[MCP] SSE transport configured: {cls.MCP_SERVER_URL}", flush=True)

        return config
    
    @classmethod
    def validate_paths(cls) -> bool:
        """
        Validate that all required paths exist and create directories as needed

        Returns:
            bool: True if validation successful, False otherwise
        """
        # Check that project root exists
        if not cls.PROJECT_ROOT.exists():
            print(f"[X] Project root does not exist: {cls.PROJECT_ROOT}", flush=True)
            return False

        # Create directories if they don't exist
        directories_to_create = [
            (cls.REPORTS_DIR, "Reports"),
            (cls.STATIC_DIR, "Static files"),
            (cls.STATIC_PLOTS_DIR, "Static plots"),
            (cls.STATIC_DATA_DIR, "Static data"),
            (cls.SESSIONS_DIR, "Sessions"),
        ]

        for directory, name in directories_to_create:
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
        """Get the reports directory path as string"""
        return str(cls.REPORTS_DIR)

    @classmethod
    def get_static_dir(cls) -> str:
        """Get the static directory path as string"""
        return str(cls.STATIC_DIR)

    @classmethod
    def get_static_plots_dir(cls) -> str:
        """Get the static plots directory path as string"""
        return str(cls.STATIC_PLOTS_DIR)

    @classmethod
    def get_static_data_dir(cls) -> str:
        """Get the static data directory path as string"""
        return str(cls.STATIC_DATA_DIR)

    @classmethod
    def get_sessions_dir(cls) -> str:
        """Get the sessions directory path as string"""
        return str(cls.SESSIONS_DIR)

    @classmethod
    def get_secrets_dir(cls) -> str:
        """Get the secrets directory path as string"""
        return str(cls.SECRETS_DIR)

    @classmethod
    def get_report_file_path(cls, filename: str) -> str:
        """Get full path for a report file"""
        return str(cls.REPORTS_DIR / filename)

    @classmethod
    def get_session_file_path(cls, filename: str) -> str:
        """Get full path for a session file"""
        return str(cls.SESSIONS_DIR / filename)

    @classmethod
    def is_valid_report_file(cls, filename: str) -> bool:
        """Check if filename has a valid report format"""
        return any(filename.endswith(ext) for ext in cls.SUPPORTED_REPORT_FORMATS)
    

# Environment-specific configurations
# Use AgentConfig directly as Config
Config = AgentConfig