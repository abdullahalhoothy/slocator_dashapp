"""Dash UI components for displaying reports."""

import re
from typing import Any, Dict, Optional

import dash_bootstrap_components as dbc
import markdown
from dash import dcc, html

from .config import Config


class ReportAppUIBuilder:
    """Build the Dash components for report rendering and chat status."""

    def __init__(self):
        self.markdown_extensions = ["tables", "fenced_code", "codehilite"]

    def create_report_layout(self) -> html.Div:
        return html.Div(
            [
                html.Div(
                    [
                        html.H5(
                            "📊 Territory Analysis Report",
                            style={"margin-bottom": "10px", "color": "#495057"},
                        ),
                        html.Div(
                            id="report-status",
                            children=[
                                html.Small(
                                    "No report loaded",
                                    style={"color": "#6c757d", "font-style": "italic"},
                                )
                            ],
                        ),
                    ],
                    style={"margin-bottom": "20px", "text-align": "center"},
                ),
                dcc.Store(id="interactive-data-available", data=False),
                html.Div(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(label="📄 Report", tab_id="static-report",
                                        active_tab_style={"font-weight": "bold"}),
                                dbc.Tab(label="📊 Interactive Maps", tab_id="interactive-plots",
                                        disabled=True, active_tab_style={"font-weight": "bold"}),
                            ],
                            id="report-tabs",
                            active_tab="static-report",
                            style={"margin-bottom": "15px"},
                        ),
                        html.Div(
                            id="report-content",
                            children=[self._create_empty_state()],
                            style={
                                "height": "calc(100vh - 200px)",
                                "overflow-y": "auto",
                                "padding": "20px",
                                "background-color": "white",
                                "border": "1px solid #dee2e6",
                                "border-radius": "8px",
                                "box-shadow": "0 2px 4px rgba(0,0,0,0.1)",
                            },
                        ),
                        html.Div(
                            id="interactive-plots-content",
                            children=[self._create_interactive_plots_placeholder()],
                            style={"display": "none"},
                        ),
                    ]
                ),
            ],
            id="report-display-container",
        )

    def _create_empty_state(self) -> html.Div:
        return html.Div(
            [
                html.Div(
                    [
                        html.I(
                            className="fas fa-file-alt",
                            style={"font-size": "48px", "color": "#dee2e6", "margin-bottom": "20px"},
                        ),
                        html.H6("No Report Available",
                                style={"color": "#6c757d", "margin-bottom": "10px"}),
                        html.P(
                            "Start a conversation with the AI assistant to generate territory analysis reports. "
                            "Reports will appear here automatically when generated.",
                            style={"color": "#6c757d", "font-style": "italic", "text-align": "center"},
                        ),
                    ],
                    style={"text-align": "center", "margin-top": "100px", "padding": "40px"},
                )
            ]
        )

    def _create_interactive_plots_placeholder(self) -> html.Div:
        return html.Div(
            [
                html.Div(
                    [
                        html.I(
                            className="fas fa-map",
                            style={"font-size": "48px", "color": "#dee2e6", "margin-bottom": "20px"},
                        ),
                        html.H6("Interactive Maps Available",
                                style={"color": "#6c757d", "margin-bottom": "10px"}),
                        html.P(
                            "Interactive territory visualization maps will be loaded here when data is available.",
                            style={"color": "#6c757d", "font-style": "italic", "text-align": "center"},
                        ),
                    ],
                    style={"text-align": "center", "margin-top": "100px", "padding": "40px"},
                )
            ]
        )

    def create_interactive_plots_layout(
        self, available_variables: Dict[str, str], default_variable: Optional[str] = None
    ) -> html.Div:
        if not available_variables:
            return self._create_interactive_plots_placeholder()

        default_var = default_variable or next(iter(available_variables))

        return html.Div(
            [
                html.Div(
                    [
                        html.H5(
                            "🗺️ Interactive Territory Visualization",
                            style={"margin-bottom": "15px", "color": "#495057", "text-align": "center"},
                        ),
                        html.P(
                            "Explore your territory optimization results with interactive maps. "
                            "Select different variables to visualize population, purchasing power, and facility distribution.",
                            style={"color": "#6c757d", "text-align": "center", "margin-bottom": "20px"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label(
                            "Select Variable to Visualize:",
                            style={"font-weight": "bold", "margin-bottom": "10px", "display": "block"},
                        ),
                        dcc.Dropdown(
                            id="interactive-variable-dropdown",
                            options=[
                                {"label": display_name, "value": var_key}
                                for var_key, display_name in available_variables.items()
                            ],
                            value=default_var,
                            clearable=False,
                            style={"width": "100%", "margin-bottom": "20px"},
                        ),
                    ],
                    style={"margin-bottom": "25px"},
                ),
                html.Div(
                    [
                        html.H6(
                            "Population and Economic Data by Territory",
                            style={"margin-bottom": "15px", "color": "#495057"},
                        ),
                        dcc.Graph(
                            id="interactive-choropleth-map",
                            style={
                                "height": "500px",
                                "border": "1px solid #dee2e6",
                                "border-radius": "8px",
                                "margin-bottom": "30px",
                            },
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.H6("Facility Locations",
                                style={"margin-bottom": "15px", "color": "#495057"}),
                        dcc.Graph(
                            id="interactive-scatter-map",
                            style={
                                "height": "500px",
                                "border": "1px solid #dee2e6",
                                "border-radius": "8px",
                            },
                        ),
                    ]
                ),
            ],
            style={"padding": "0 10px"},
        )

    def _fix_image_paths(self, content: str) -> str:
        """Resize report images to 500x500px and route them via Config.STATIC_PLOTS_URL."""
        plots_url = Config.STATIC_PLOTS_URL

        patterns = [
            (
                r'<img\s+src="(https?://[^"]+/static/plots/[^"]+)"([^>]*?)>',
                r'<div style="text-align: center; margin: 20px 0;"><img src="\1" width="500" height="500" style="object-fit: contain; display: block; margin: 0 auto;"\2></div>',
            ),
            (
                r'!\[([^\]]*)\]\((https?://[^)]+/static/plots/[^)]+)\)',
                r'<div style="text-align: center; margin: 20px 0;"><img src="\2" alt="\1" width="500" height="500" style="object-fit: contain; display: block; margin: 0 auto;"></div>',
            ),
            (
                r'<img\s+src="\.\.\/static\/plots\/([^"]+)"([^>]*?)>',
                rf'<div style="text-align: center; margin: 20px 0;"><img src="{plots_url}/\1" width="500" height="500" style="object-fit: contain; display: block; margin: 0 auto;"\2></div>',
            ),
            (
                r'!\[([^\]]*)\]\(\.\.\/static\/plots\/([^)]+)\)',
                rf'<div style="text-align: center; margin: 20px 0;"><img src="{plots_url}/\2" alt="\1" width="500" height="500" style="object-fit: contain; display: block; margin: 0 auto;"></div>',
            ),
            (
                r'<img\s+src="\/static\/plots\/([^"]+)"([^>]*?)>',
                rf'<div style="text-align: center; margin: 20px 0;"><img src="{plots_url}/\1" width="500" height="500" style="object-fit: contain; display: block; margin: 0 auto;"\2></div>',
            ),
            (
                r'!\[([^\]]*)\]\(\/static\/plots\/([^)]+)\)',
                rf'<div style="text-align: center; margin: 20px 0;"><img src="{plots_url}/\2" alt="\1" width="500" height="500" style="object-fit: contain; display: block; margin: 0 auto;"></div>',
            ),
        ]

        fixed_content = content
        for pattern, replacement in patterns:
            fixed_content = re.sub(pattern, replacement, fixed_content)
        return fixed_content

    def format_markdown_for_dash(self, content: str) -> html.Div:
        if not content or not content.strip():
            return self._create_empty_state()

        try:
            content = self._fix_image_paths(content)
            markdown.markdown(
                content,
                extensions=self.markdown_extensions,
                extension_configs={"codehilite": {"css_class": "highlight"}},
            )
            return html.Div(
                [
                    dcc.Markdown(
                        content,
                        dangerously_allow_html=True,
                        style={
                            "font-family": "system-ui, -apple-system, sans-serif",
                            "line-height": "1.6",
                            "color": "#333",
                        },
                    )
                ],
                style={"padding": "10px"},
            )
        except Exception as e:
            print(f"❌ Error formatting markdown: {str(e)}")
            return html.Div(
                [
                    dbc.Alert(
                        [
                            html.H6("Error Displaying Report", className="alert-heading"),
                            html.P(f"Could not format the report content: {str(e)}"),
                            html.Hr(),
                            html.P("Please try regenerating the report.", className="mb-0"),
                        ],
                        color="warning",
                    )
                ]
            )

    def create_report_status_indicator(
        self, status: str, report_info: Optional[Dict[str, Any]] = None
    ) -> html.Div:
        if status == "loading":
            return html.Div(
                [
                    dbc.Spinner(size="sm", color="primary"),
                    html.Small(" Loading report...",
                               style={"margin-left": "10px", "color": "#007bff"}),
                ],
                style={"display": "flex", "align-items": "center", "justify-content": "center"},
            )

        if status == "loaded" and report_info:
            return html.Div(
                [
                    html.Small("📄 ", style={"color": "#28a745"}),
                    html.Small(
                        f"Report loaded: {report_info.get('filename', 'Unknown')}",
                        style={"color": "#28a745", "font-weight": "bold"},
                    ),
                    html.Br(),
                    html.Small(
                        f"Generated: {report_info.get('created_time', 'Unknown')}",
                        style={"color": "#6c757d", "font-size": "0.8em"},
                    ),
                ],
                style={"text-align": "center"},
            )

        if status == "error":
            return html.Div(
                [
                    html.Small("❌ Error loading report",
                               style={"color": "#dc3545", "font-weight": "bold"})
                ],
                style={"text-align": "center"},
            )

        return html.Div(
            [
                html.Small("No report loaded",
                           style={"color": "#6c757d", "font-style": "italic"})
            ],
            style={"text-align": "center"},
        )

    def create_error_display(self, error_message: str) -> html.Div:
        return html.Div(
            [
                dbc.Alert(
                    [
                        html.H6("Report Error", className="alert-heading"),
                        html.P(error_message),
                        html.Hr(),
                        html.P("Please try the following:", className="mb-2"),
                        html.Ul(
                            [
                                html.Li("Check that your query includes login credentials"),
                                html.Li("Ensure you've requested a specific analysis type"),
                                html.Li("Try rephrasing your request"),
                                html.Li("Contact support if the problem persists"),
                            ]
                        ),
                    ],
                    color="danger",
                )
            ],
            style={"margin": "20px"},
        )


report_app_ui_builder = ReportAppUIBuilder()