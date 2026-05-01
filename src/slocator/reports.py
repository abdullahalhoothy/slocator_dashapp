"""Markdown report file I/O and metadata extraction."""

import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config


class ReportDataManager:
    """Read and inspect markdown reports written by the MCP server."""

    def __init__(self):
        self.reports_dir = Path(Config.get_reports_path())
        self.supported_formats = Config.SUPPORTED_REPORT_FORMATS
        self.default_type = Config.DEFAULT_REPORT_TYPE
        self.last_data_files: Dict[str, str] = {}

    def get_data_files(self, structured_output=None) -> Dict[str, str]:
        if structured_output and hasattr(structured_output, "data_files"):
            data_files = structured_output.data_files or {}
            print(f"[ReportHandler] Using structured output data files: {list(data_files.keys())}")
            return data_files
        print(f"[ReportHandler] Using cached data files: {list(self.last_data_files.keys())}")
        return self.last_data_files.copy()

    def _translate_mcp_path_to_dashapp_path(self, file_handle: str) -> Path:
        """When running in Docker, the MCP server writes to ``/app/MCP_Server/reports/``
        but the Dash app reads from ``/app/reports/`` — the same host volume."""
        if file_handle.startswith("/app/MCP_Server/reports/"):
            return self.reports_dir / file_handle.replace("/app/MCP_Server/reports/", "")
        if os.path.isabs(file_handle):
            return Path(file_handle)
        return self.reports_dir / file_handle

    def read_md_report(self, file_handle: str) -> Optional[str]:
        try:
            file_path = self._translate_mcp_path_to_dashapp_path(file_handle)
            if not file_path.exists():
                print(f"Report file not found: {file_path}")
                return None
            if file_path.suffix != ".md":
                print(f"File is not a markdown file: {file_path}")
                return None
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                print(f"Successfully read report: {file_path.name}")
                return content
        except Exception as e:
            print(f"Error reading report file {file_handle}: {str(e)}")
            return None

    def validate_report_exists(self, file_path: str) -> bool:
        try:
            path = self._translate_mcp_path_to_dashapp_path(file_path)
            return path.exists() and path.is_file()
        except Exception as e:
            print(f"Error validating report path {file_path}: {str(e)}")
            return False

    def extract_report_metadata(self, file_path: str) -> Dict[str, Any]:
        try:
            path = self._translate_mcp_path_to_dashapp_path(file_path)
            if not path.exists():
                return {}

            filename = path.stem
            metadata: Dict[str, Any] = {
                "filename": path.name,
                "file_path": str(path),
                "file_size": path.stat().st_size,
                "created_time": datetime.fromtimestamp(path.stat().st_ctime),
                "modified_time": datetime.fromtimestamp(path.stat().st_mtime),
                "format": path.suffix[1:],
            }

            # Expected filename pattern: City_territory_report_type_YYYYMMDD_HHMMSS
            parts = filename.split("_")
            if len(parts) >= 4:
                metadata.update({
                    "city": parts[0],
                    "report_type": "_".join(parts[1:-2]) if len(parts) > 4 else parts[1],
                    "date_part": parts[-2],
                    "time_part": parts[-1],
                })
                try:
                    if len(metadata["date_part"]) == 8 and len(metadata["time_part"]) == 6:
                        metadata["parsed_datetime"] = datetime.strptime(
                            f"{metadata['date_part']}_{metadata['time_part']}", "%Y%m%d_%H%M%S"
                        )
                except ValueError:
                    pass

            return metadata
        except Exception as e:
            print(f"Error extracting metadata from {file_path}: {str(e)}")
            return {}

    def list_available_reports(self, report_type: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            pattern = (
                Config.REPORT_FILE_PATTERNS[report_type]
                if report_type and report_type in Config.REPORT_FILE_PATTERNS
                else "*.*"
            )
            files = glob.glob(str(self.reports_dir / pattern))
            reports = []
            for file_path in files:
                if Config.is_valid_report_file(Path(file_path).name):
                    metadata = self.extract_report_metadata(file_path)
                    if metadata:
                        reports.append(metadata)
            reports.sort(key=lambda x: x.get("created_time", datetime.min), reverse=True)
            return reports
        except Exception as e:
            print(f"Error listing reports: {str(e)}")
            return []

    def get_latest_report(self, report_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        report_type = report_type or self.default_type
        reports = self.list_available_reports(report_type)
        return reports[0] if reports else None

    def parse_file_handle_from_response(self, structured_output) -> Optional[str]:
        if structured_output is None or not hasattr(structured_output, "report_file"):
            return None

        report_file = structured_output.report_file
        if not report_file:
            return None

        print(f"[ReportHandler] Found report file: {report_file}")

        if hasattr(structured_output, "data_files") and structured_output.data_files:
            self.last_data_files = structured_output.data_files
            print(f"[ReportHandler] Cached data files: {list(self.last_data_files.keys())}")
        else:
            self.last_data_files = {}

        return report_file


report_data_manager = ReportDataManager()
