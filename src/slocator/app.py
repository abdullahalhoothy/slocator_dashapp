"""Dash app — layout and callbacks.

The module-level ``app`` is the single Dash instance for the process. Auth
helpers live in ``slocator.auth`` and the per-browser MCP client cache lives
in ``slocator.mcp_client``.
"""

import asyncio
import uuid
from datetime import timedelta

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from flask import session as flask_session

from .auth import (
    authenticate_user_direct,
    get_browser_id,
    get_current_auth_status,
    logout_user,
    update_mcp_session_auth,
)
from .config import Config
from .mcp_client import ensure_client_connected, get_thread_id
from .plots import load_and_create_plots, plotter
from .reports import report_data_manager
from .ui import report_app_ui_builder

# ===== Dash app initialization ======================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

# Flask signed-cookie session: holds an opaque per-browser UUID, never tokens.
app.server.secret_key = Config.FLASK_SECRET_KEY
app.server.permanent_session_lifetime = timedelta(hours=Config.SESSION_DURATION_HOURS)

Config.validate_paths()
app.server.static_folder = Config.get_static_dir()
app.server.static_url_path = Config.STATIC_URL_PATH


@app.server.before_request
def _ensure_browser_id():
    """Assign a stable per-browser UUID on first request so every user gets
    their own auth file and MCP client."""
    if "browser_id" not in flask_session:
        flask_session["browser_id"] = uuid.uuid4().hex
        flask_session.permanent = True


def _run_async(coro):
    """Run an async coroutine from a sync Dash callback."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("event loop closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ===== Layout =======================================================================

app.layout = html.Div(
    [
        dcc.Store(id="auth-state-store", data={"authenticated": False}),
        dcc.Store(id="user-data-store", data={}),

        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Login to Access AI Assistant")),
                dbc.ModalBody(
                    [
                        dbc.Form(
                            [
                                dbc.Row(
                                    [
                                        dbc.Label("Email", html_for="login-email", width=2),
                                        dbc.Col(
                                            [
                                                dbc.Input(
                                                    type="email",
                                                    id="login-email",
                                                    placeholder="Enter your email",
                                                    required=True,
                                                ),
                                            ],
                                            width=10,
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Label("Password", html_for="login-password", width=2),
                                        dbc.Col(
                                            [
                                                dbc.Input(
                                                    type="password",
                                                    id="login-password",
                                                    placeholder="Enter your password",
                                                    required=True,
                                                ),
                                            ],
                                            width=10,
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                            ]
                        ),
                        html.Div(id="login-error-message", style={"color": "red", "margin-top": "10px"}),
                    ]
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button("Cancel", id="login-cancel-btn", className="me-2", color="secondary"),
                        dbc.Button("Login", id="login-submit-btn", color="primary", type="submit"),
                    ]
                ),
            ],
            id="login-modal",
            is_open=False,
            backdrop="static",
            keyboard=False,
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            id="left-column-content",
                            children=[report_app_ui_builder.create_report_layout()],
                            style={
                                "height": "100vh",
                                "overflow-y": "auto",
                                "padding": "20px",
                                "background-color": "#f8f9fa",
                            },
                        )
                    ],
                    id="left-column",
                    width=8,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H4(
                                            "AI Assistant",
                                            style={"margin": "0", "text-align": "center"},
                                        ),
                                        html.Div(
                                            id="auth-status-indicator",
                                            children=[
                                                html.Small(
                                                    "🔒 Not logged in",
                                                    style={"color": "#dc3545", "font-weight": "bold"},
                                                ),
                                                html.Br(),
                                                dbc.ButtonGroup(
                                                    [
                                                        dbc.Button("Login", id="login-btn",
                                                                   size="sm", color="primary"),
                                                        dbc.Button(
                                                            "Logout",
                                                            id="logout-btn",
                                                            size="sm",
                                                            color="secondary",
                                                            style={"display": "none"},
                                                        ),
                                                    ],
                                                    size="sm",
                                                    style={"margin-top": "5px"},
                                                ),
                                            ],
                                            style={
                                                "margin-top": "10px",
                                                "padding": "8px",
                                                "background-color": "#f8f9fa",
                                                "border-radius": "5px",
                                                "text-align": "center",
                                            },
                                        ),
                                    ],
                                    style={"margin-bottom": "20px"},
                                ),
                                html.Div(
                                    id="conversation-div",
                                    children=[],
                                    style={
                                        "height": "calc(100vh - 250px)",
                                        "overflow-y": "auto",
                                        "padding": "15px",
                                        "border": "1px solid #dee2e6",
                                        "border-radius": "5px",
                                        "background-color": "white",
                                        "margin-bottom": "15px",
                                        "display": "flex",
                                        "flex-direction": "column-reverse",
                                    },
                                ),
                                html.Div(
                                    [
                                        dbc.InputGroup(
                                            [
                                                dbc.Input(
                                                    id="query-input",
                                                    placeholder="Enter your query here...",
                                                    type="text",
                                                    style={"border-radius": "20px 0 0 20px"},
                                                ),
                                                dbc.Button(
                                                    "Send",
                                                    id="send-button",
                                                    color="primary",
                                                    n_clicks=0,
                                                    style={"border-radius": "0 20px 20px 0"},
                                                ),
                                            ]
                                        )
                                    ],
                                    style={"position": "sticky", "bottom": "0"},
                                ),
                            ],
                            style={
                                "height": "100vh",
                                "padding": "20px",
                                "display": "flex",
                                "flex-direction": "column",
                            },
                        )
                    ],
                    id="right-column",
                    width=4,
                ),
            ],
            style={"margin": "0", "height": "100vh"},
        ),

        dbc.Button(
            "−",
            id="minimize-button",
            style={
                "position": "fixed",
                "top": "20px",
                "right": "20px",
                "width": "50px",
                "height": "50px",
                "border-radius": "50%",
                "background-color": "#28a745",
                "border": "none",
                "color": "white",
                "font-size": "24px",
                "font-weight": "bold",
                "box-shadow": "0 4px 8px rgba(0,0,0,0.3)",
                "z-index": "1000",
                "display": "flex",
                "align-items": "center",
                "justify-content": "center",
            },
            n_clicks=0,
        ),
    ],
    style={"height": "100vh", "overflow": "hidden"},
)


# ===== Authentication callbacks ====================================================

@app.callback(
    Output("login-modal", "is_open"),
    [Input("login-btn", "n_clicks"), Input("login-cancel-btn", "n_clicks")],
    [State("login-modal", "is_open")],
)
def toggle_login_modal(login_clicks, cancel_clicks, is_open):
    if login_clicks or cancel_clicks:
        return not is_open
    return is_open


@app.callback(
    [
        Output("auth-state-store", "data"),
        Output("user-data-store", "data"),
        Output("login-error-message", "children"),
        Output("login-email", "value"),
        Output("login-password", "value"),
    ],
    [Input("login-submit-btn", "n_clicks")],
    [State("login-email", "value"), State("login-password", "value")],
)
def handle_login(n_clicks, email, password):
    if not (n_clicks and email and password):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    try:
        auth_result = _run_async(authenticate_user_direct(email, password))
        if not auth_result["success"]:
            return ({"authenticated": False}, {},
                    f"Login failed: {auth_result['error']}", email, "")

        login_data = auth_result["data"]
        success = _run_async(
            update_mcp_session_auth(
                login_data["localId"],
                login_data["idToken"],
                login_data["refreshToken"],
                int(login_data["expiresIn"]),
            )
        )
        if not success:
            return ({"authenticated": False}, {},
                    "Failed to update session. Please try again.", email, "")

        return (
            {"authenticated": True},
            {"user_id": login_data["localId"], "email": email},
            "",
            "",
            "",
        )

    except Exception as e:
        return ({"authenticated": False}, {}, f"An error occurred: {str(e)}", email, "")


@app.callback(
    [
        Output("auth-state-store", "data", allow_duplicate=True),
        Output("user-data-store", "data", allow_duplicate=True),
    ],
    [Input("logout-btn", "n_clicks")],
    prevent_initial_call=True,
)
def handle_logout(n_clicks):
    if not n_clicks:
        return dash.no_update, dash.no_update
    try:
        _run_async(logout_user())
    except Exception as e:
        print(f"Logout error: {str(e)}")
    return {"authenticated": False}, {}


@app.callback(
    [
        Output("auth-status-indicator", "children"),
        Output("login-modal", "is_open", allow_duplicate=True),
    ],
    [Input("auth-state-store", "data"), Input("user-data-store", "data")],
    prevent_initial_call=True,
)
def update_auth_status_display(auth_state, user_data):
    if auth_state.get("authenticated", False):
        user_email = user_data.get("email", "Unknown User")
        return (
            [
                html.Small(
                    f"✅ Logged in as {user_email}",
                    style={"color": "#28a745", "font-weight": "bold"},
                ),
                html.Br(),
                dbc.ButtonGroup(
                    [
                        dbc.Button("Login", id="login-btn", size="sm",
                                   color="primary", style={"display": "none"}),
                        dbc.Button("Logout", id="logout-btn", size="sm", color="secondary"),
                    ],
                    size="sm",
                    style={"margin-top": "5px"},
                ),
            ],
            False,
        )

    return (
        [
            html.Small(
                "🔒 Not logged in",
                style={"color": "#dc3545", "font-weight": "bold"},
            ),
            html.Br(),
            dbc.ButtonGroup(
                [
                    dbc.Button("Login", id="login-btn", size="sm", color="primary"),
                    dbc.Button("Logout", id="logout-btn", size="sm",
                               color="secondary", style={"display": "none"}),
                ],
                size="sm",
                style={"margin-top": "5px"},
            ),
        ],
        dash.no_update,
    )


@app.callback(
    [
        Output("auth-state-store", "data", allow_duplicate=True),
        Output("user-data-store", "data", allow_duplicate=True),
    ],
    [Input("auth-state-store", "id")],  # static input fires once on layout init
    prevent_initial_call="initial_duplicate",
)
def check_auth_on_load(_):
    try:
        auth_status = get_current_auth_status()
        if auth_status.get("authenticated", False):
            return (
                {"authenticated": True},
                {"user_id": auth_status.get("user_id", ""), "email": ""},
            )
        return {"authenticated": False}, {}
    except Exception as e:
        print(f"Auth check error: {str(e)}")
        return {"authenticated": False}, {}


# ===== Layout toggle ===============================================================

@app.callback(
    [
        Output("left-column", "width"),
        Output("right-column", "width"),
        Output("minimize-button", "children"),
    ],
    [Input("minimize-button", "n_clicks")],
)
def toggle_right_panel(n_clicks):
    if n_clicks % 2 == 1:
        return 12, 0, "+"
    return 8, 4, "−"


# ===== Chat (main) callback ========================================================

def _user_message(query: str) -> html.Div:
    return html.Div(
        [
            html.Div(
                "Me:",
                style={"font-weight": "bold", "color": "#007bff", "margin-bottom": "5px"},
            ),
            html.Div(
                query,
                style={
                    "background-color": "#e3f2fd",
                    "padding": "10px",
                    "border-radius": "10px",
                    "margin-bottom": "10px",
                },
            ),
        ],
        style={"margin-bottom": "15px"},
    )


def _system_login_required_message() -> html.Div:
    return html.Div(
        [
            html.Div(
                "System:",
                style={"font-weight": "bold", "color": "#dc3545", "margin-bottom": "5px"},
            ),
            html.Div(
                "🔒 Please log in first to use the AI assistant.",
                style={
                    "background-color": "#f8d7da",
                    "padding": "10px",
                    "border-radius": "10px",
                    "color": "#721c24",
                },
            ),
        ],
        style={"margin-bottom": "15px"},
    )


def _agent_message(text: str, color: str = "#28a745", bg: str = "#f8f9fa", text_color: str = None) -> html.Div:
    inner_style = {
        "background-color": bg,
        "padding": "10px",
        "border-radius": "10px",
        "white-space": "pre-wrap",
    }
    if text_color:
        inner_style["color"] = text_color
    return html.Div(
        [
            html.Div(
                "Agent:",
                style={"font-weight": "bold", "color": color, "margin-bottom": "5px"},
            ),
            html.Div(text, style=inner_style),
        ],
        style={"margin-bottom": "15px"},
    )


def _scan_geojson_fallback(static_data_dir):
    """Find the most recent set of grid_data/places_data/boundaries geojson files."""
    if not static_data_dir.exists():
        return None
    geojson_files = list(static_data_dir.glob("*_*.geojson"))
    if len(geojson_files) < 3:
        return None

    file_groups: dict = {}
    for file_path in geojson_files:
        parts = file_path.stem.split("_", 1)
        if len(parts) == 2:
            request_id, file_type = parts
            file_groups.setdefault(request_id, {})[file_type] = str(file_path)

    for _, files in sorted(file_groups.items(), reverse=True):
        if all(key in files for key in ("grid_data", "places_data", "boundaries")):
            return {
                "grid_data": files["grid_data"],
                "places_data": files["places_data"],
                "boundaries": files["boundaries"],
            }
    return None


@app.callback(
    [
        Output("conversation-div", "children"),
        Output("query-input", "value"),
        Output("report-content", "children"),
        Output("report-status", "children"),
        Output("interactive-plots-content", "children"),
        Output("interactive-data-available", "data"),
    ],
    [Input("send-button", "n_clicks"), Input("query-input", "n_submit")],
    [
        State("query-input", "value"),
        State("conversation-div", "children"),
        State("report-content", "children"),
        State("report-status", "children"),
        State("interactive-plots-content", "children"),
        State("interactive-data-available", "data"),
    ],
)
def process_query(
    n_clicks,
    n_submit,
    query,
    current_conversation,
    current_report_content,
    current_report_status,
    current_interactive_plots,
    current_data_available,
):
    has_input = (n_clicks and n_clicks > 0) or n_submit
    has_query = bool(query and query.strip())

    if not (has_input and has_query):
        preserved_report = current_report_content or report_app_ui_builder._create_empty_state()
        preserved_status = current_report_status or report_app_ui_builder.create_report_status_indicator("empty")
        preserved_plots = current_interactive_plots or report_app_ui_builder._create_interactive_plots_placeholder()
        preserved_available = current_data_available or False
        return (
            current_conversation or [],
            query or "",
            preserved_report,
            preserved_status,
            preserved_plots,
            preserved_available,
        )

    auth_status = get_current_auth_status()
    if not auth_status.get("authenticated", False):
        conversation = [_system_login_required_message(), _user_message(query)] + (current_conversation or [])
        preserved_report = current_report_content or report_app_ui_builder._create_empty_state()
        preserved_status = current_report_status or report_app_ui_builder.create_report_status_indicator("empty")
        preserved_plots = current_interactive_plots or report_app_ui_builder._create_interactive_plots_placeholder()
        preserved_available = current_data_available or False
        return conversation, "", preserved_report, preserved_status, preserved_plots, preserved_available

    try:
        browser_id = get_browser_id()

        async def run_query_with_memory():
            client = await ensure_client_connected(browser_id)
            if not client:
                return {"response": "Error: Could not connect to MCP client", "raw_content": ""}
            try:
                return await client.analyze_territories_with_file_handle(
                    query, thread_id=get_thread_id(browser_id)
                )
            except Exception as e:
                return {"response": f"Error processing query: {str(e)}", "raw_content": ""}

        result = _run_async(run_query_with_memory())

        agent_response = str(result.get("response", "")) if isinstance(result, dict) else str(result)

        conversation = [_agent_message(agent_response), _user_message(query)] + (current_conversation or [])

        report_content = current_report_content or report_app_ui_builder._create_empty_state()
        report_status = current_report_status or report_app_ui_builder.create_report_status_indicator("empty")
        interactive_plots_content = current_interactive_plots or report_app_ui_builder._create_interactive_plots_placeholder()
        interactive_data_available = current_data_available or False

        structured_output = result.get("structured_output") if isinstance(result, dict) else None
        if structured_output:
            file_handle = report_data_manager.parse_file_handle_from_response(structured_output)
            if file_handle:
                print(f"📄 Found file handle: {file_handle}")
                md_content = report_data_manager.read_md_report(file_handle)
                if md_content:
                    report_content = report_app_ui_builder.format_markdown_for_dash(md_content)
                    metadata = report_data_manager.extract_report_metadata(file_handle)
                    report_status = report_app_ui_builder.create_report_status_indicator("loaded", metadata)

                    data_files = report_data_manager.get_data_files(structured_output)
                    if not data_files:
                        try:
                            data_files = _scan_geojson_fallback(Config.STATIC_DATA_DIR) or {}
                        except Exception as e:
                            print(f"❌ Error in fallback data file detection: {str(e)}")
                            data_files = {}

                    if data_files:
                        success, plot_info = load_and_create_plots(data_files)
                        if success:
                            interactive_plots_content = report_app_ui_builder.create_interactive_plots_layout(
                                plot_info["variables"], plot_info["default_variable"]
                            )
                            interactive_data_available = True
                else:
                    report_status = report_app_ui_builder.create_report_status_indicator("error")

        return (
            conversation,
            "",
            report_content,
            report_status,
            interactive_plots_content,
            interactive_data_available,
        )

    except Exception as e:
        conversation = [
            _agent_message(
                f"Error: {str(e)}", color="#dc3545", bg="#f8d7da", text_color="#721c24"
            ),
            _user_message(query),
        ] + (current_conversation or [])
        return (
            conversation,
            "",
            report_app_ui_builder.create_error_display(str(e)),
            report_app_ui_builder.create_report_status_indicator("error"),
            report_app_ui_builder._create_interactive_plots_placeholder(),
            False,
        )


# ===== Tab + plot callbacks ========================================================

@app.callback(
    Output("report-tabs", "children"),
    [Input("interactive-data-available", "data")],
)
def update_tab_state(data_available):
    if data_available:
        return [
            dbc.Tab(label="📄 Report", tab_id="static-report",
                    active_tab_style={"font-weight": "bold"}),
            dbc.Tab(label="📊 Interactive Maps", tab_id="interactive-plots",
                    active_tab_style={"font-weight": "bold"}),
        ]
    return [
        dbc.Tab(label="📄 Report", tab_id="static-report",
                active_tab_style={"font-weight": "bold"}),
        dbc.Tab(label="📊 Interactive Maps", tab_id="interactive-plots",
                disabled=True, active_tab_style={"font-weight": "bold"}),
    ]


_TAB_PANEL_STYLE = {
    "height": "calc(100vh - 200px)",
    "overflow-y": "auto",
    "padding": "20px",
    "background-color": "white",
    "border": "1px solid #dee2e6",
    "border-radius": "8px",
    "box-shadow": "0 2px 4px rgba(0,0,0,0.1)",
}


@app.callback(
    [
        Output("report-content", "style"),
        Output("interactive-plots-content", "style"),
    ],
    [Input("report-tabs", "active_tab")],
)
def switch_report_tabs(active_tab):
    if active_tab == "interactive-plots":
        return {"display": "none"}, {"display": "block", **_TAB_PANEL_STYLE}
    return _TAB_PANEL_STYLE, {"display": "none"}


@app.callback(
    Output("interactive-choropleth-map", "figure"),
    [Input("interactive-variable-dropdown", "value")],
    prevent_initial_call=True,
)
def update_interactive_choropleth(selected_variable):
    if not selected_variable:
        return {}
    try:
        return plotter.create_choropleth_map(selected_variable) or {}
    except Exception as e:
        print(f"❌ Error updating choropleth map: {str(e)}")
        return {}


@app.callback(
    Output("interactive-scatter-map", "figure"),
    [Input("interactive-variable-dropdown", "value")],
    prevent_initial_call=True,
)
def update_interactive_scatter(_selected_variable):
    try:
        return plotter.create_supermarket_scatter_map() or {}
    except Exception as e:
        print(f"❌ Error updating scatter map: {str(e)}")
        return {}