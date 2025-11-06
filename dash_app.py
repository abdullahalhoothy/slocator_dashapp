import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from pathlib import Path

# Import configuration for FastAPI endpoints
import sys
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import our custom modules
from ReportDataManager import report_data_manager
from ReportAppUIBuilder import report_app_ui_builder
from interactive_plots import plotter, load_and_create_plots
from mcp_client import get_or_create_client, ensure_client_connected, get_thread_id
from config import Config

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

# Configure Flask server with secret key for session management
app.server.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production-' + os.urandom(24).hex())

# Configure static file serving for plots using Config
Config.validate_paths()  # Ensure all directories exist
app.server.static_folder = Config.get_static_dir()
app.server.static_url_path = Config.STATIC_URL_PATH
print(f"📁 Static files configured: {Config.STATIC_DIR}")
print(f"📁 Static plots directory: {Config.STATIC_PLOTS_DIR}")
print(f"📁 Static data directory: {Config.STATIC_DATA_DIR}")

# Store for conversation history
conversation_history = []

# Authentication functions
async def authenticate_user_direct(email: str, password: str) -> dict:
    """
    Call FastAPI /login endpoint directly and return authentication result
    
    Returns:
        dict with 'success', 'data' (if success), 'error' (if failure)
    """
    try:
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        endpoint_url = backend_url + "/fastapi/login"
        payload = {
            "message": "login request from dash app",
            "request_info": {},
            "request_body": {"email": email, "password": password},
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint_url, json=payload) as response:
                if response.status == 200:
                    response_json = await response.json()
                    login_data = response_json.get("data")
                    if login_data:
                        return {
                            "success": True,
                            "data": login_data,
                            "user_email": email
                        }
                    else:
                        return {
                            "success": False,
                            "error": "Invalid response format from server"
                        }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"Login failed: {error_text}"
                    }
                    
    except Exception as e:
        return {
            "success": False,
            "error": f"Network error: {str(e)}"
        }

async def update_mcp_session_auth(user_id: str, id_token: str, refresh_token: str, expires_in: int) -> bool:
    """
    Update MCP session manager with authentication tokens via SSE server

    For SSE mode, authentication is typically handled by the server.
    This function stores credentials locally for the Dash app to pass to MCP tools.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get MCP client to ensure it's connected
        client = get_or_create_client()
        await ensure_client_connected()

        session_id = "dash_session"  # Matches the client session ID

        print(f"[AUTH] Storing auth tokens for MCP SSE session (user: {user_id})", flush=True)

        # Store tokens locally for the Dash app
        # The SSE server will handle its own session management
        metadata_path = Path(Config.get_session_file_path(f"{session_id}_auth.json"))

        # Store auth data locally
        auth_data = {
            "session_id": session_id,
            "user_id": user_id,
            "id_token": id_token,
            "refresh_token": refresh_token,
            "token_expires_at": (datetime.now() + timedelta(seconds=expires_in - 60)).isoformat(),
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=8)).isoformat()
        }

        # Save auth metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(auth_data, f, indent=2)

        print(f"[OK] Successfully stored auth tokens for SSE session (user: {user_id})", flush=True)
        print(f"[INFO] Auth tokens are managed by the MCP SSE server at http://localhost:8001", flush=True)
        return True

    except Exception as e:
        print(f"[ERROR] Failed to store session auth: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return False

async def logout_user() -> bool:
    """
    Clear authentication tokens from local session storage

    For SSE mode, we clear local auth tokens.
    The SSE server manages its own session state.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        session_id = "dash_session"

        # Clear local session storage
        metadata_path = Path(Config.get_session_file_path(f"{session_id}_auth.json"))

        if metadata_path.exists():
            metadata_path.unlink()
            print("[LOGOUT] Successfully removed local auth tokens", flush=True)
        else:
            print("[LOGOUT] No local auth tokens found", flush=True)

        print("[INFO] SSE server manages its own session state", flush=True)
        return True

    except Exception as e:
        print(f"[ERROR] Failed to logout user: {str(e)}", flush=True)
        return False

def get_current_auth_status() -> dict:
    """
    Get current authentication status from local session storage

    For SSE mode, we check local auth tokens.
    The SSE server manages actual authentication state.

    Returns:
        dict with 'authenticated', 'user_id', 'expires_at'
    """
    try:
        session_id = "dash_session"

        # Check local session storage
        metadata_path = Path(Config.get_session_file_path(f"{session_id}_auth.json"))

        if not metadata_path.exists():
            return {"authenticated": False}

        with open(metadata_path, 'r', encoding='utf-8') as f:
            auth_data = json.load(f)

        # Check if user is authenticated and token is not expired
        user_id = auth_data.get("user_id")
        token_expires_at = auth_data.get("token_expires_at")

        if user_id and token_expires_at:
            expiry_time = datetime.fromisoformat(token_expires_at)
            if datetime.now() < expiry_time:
                return {
                    "authenticated": True,
                    "user_id": user_id,
                    "expires_at": token_expires_at
                }

        return {"authenticated": False}

    except Exception as e:
        print(f"[ERROR] Failed to get auth status: {str(e)}", flush=True)
        return {"authenticated": False}

# Define the layout (following original pattern exactly)
app.layout = html.Div([
    # Store components for authentication state
    dcc.Store(id="auth-state-store", data={"authenticated": False}),
    dcc.Store(id="user-data-store", data={}),
    
    # Login Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Login to Access AI Assistant")),
        dbc.ModalBody([
            dbc.Form([
                dbc.Row([
                    dbc.Label("Email", html_for="login-email", width=2),
                    dbc.Col([
                        dbc.Input(
                            type="email", 
                            id="login-email", 
                            placeholder="Enter your email",
                            required=True
                        ),
                    ], width=10),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Label("Password", html_for="login-password", width=2),
                    dbc.Col([
                        dbc.Input(
                            type="password", 
                            id="login-password", 
                            placeholder="Enter your password",
                            required=True
                        ),
                    ], width=10),
                ], className="mb-3"),
            ]),
            html.Div(id="login-error-message", style={'color': 'red', 'margin-top': '10px'}),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="login-cancel-btn", className="me-2", color="secondary"),
            dbc.Button("Login", id="login-submit-btn", color="primary", type="submit"),
        ]),
    ],
    id="login-modal",
    is_open=False,
    backdrop="static",  # Prevent closing by clicking outside
    keyboard=False,  # Prevent closing with escape key
    ),
    
    dbc.Row([
        # Left column (70% width) - Report display area
        dbc.Col([
            html.Div(
                id="left-column-content",
                children=[
                    report_app_ui_builder.create_report_layout()
                ],
                style={
                    'height': '100vh',
                    'overflow-y': 'auto',
                    'padding': '20px',
                    'background-color': '#f8f9fa'
                }
            )
        ], id="left-column", width=8),
        
        # Right column (30% width) - Chat interface
        dbc.Col([
            html.Div([
                # Header with authentication status
                html.Div([
                    html.H4("AI Assistant", style={'margin': '0', 'text-align': 'center'}),
                    # Authentication status indicator
                    html.Div(
                        id="auth-status-indicator",
                        children=[
                            html.Small("🔒 Not logged in", 
                                      style={'color': '#dc3545', 'font-weight': 'bold'}),
                            html.Br(),
                            dbc.ButtonGroup([
                                dbc.Button("Login", id="login-btn", size="sm", color="primary"),
                                dbc.Button("Logout", id="logout-btn", size="sm", color="secondary", 
                                         style={'display': 'none'})
                            ], size="sm", style={'margin-top': '5px'})
                        ], 
                        style={'margin-top': '10px', 'padding': '8px', 
                               'background-color': '#f8f9fa', 'border-radius': '5px',
                               'text-align': 'center'}
                    )
                ], style={'margin-bottom': '20px'}),
                
                # Results area (scrollable)
                html.Div(
                    id="conversation-div",
                    children=[],
                    style={
                        'height': 'calc(100vh - 250px)',  # Adjusted for memory indicator
                        'overflow-y': 'auto',
                        'padding': '15px',
                        'border': '1px solid #dee2e6',
                        'border-radius': '5px',
                        'background-color': 'white',
                        'margin-bottom': '15px',
                        'display': 'flex',
                        'flex-direction': 'column-reverse'  # Show latest messages at bottom
                    }
                ),
                
                # Input area (fixed at bottom)
                html.Div([
                    dbc.InputGroup([
                        dbc.Input(
                            id="query-input",
                            placeholder="Enter your query here...",
                            type="text",
                            style={'border-radius': '20px 0 0 20px'}
                        ),
                        dbc.Button(
                            "Send",
                            id="send-button",
                            color="primary",
                            n_clicks=0,
                            style={'border-radius': '0 20px 20px 0'}
                        )
                    ])
                ], style={'position': 'sticky', 'bottom': '0'})
            ], style={
                'height': '100vh',
                'padding': '20px',
                'display': 'flex',
                'flex-direction': 'column'
            })
        ], id="right-column", width=4)
    ], style={'margin': '0', 'height': '100vh'}),
    
    # Floating toggle button
    dbc.Button(
        "−",
        id="minimize-button",
        style={
            'position': 'fixed',
            'top': '20px',
            'right': '20px',
            'width': '50px',
            'height': '50px',
            'border-radius': '50%',
            'background-color': '#28a745',
            'border': 'none',
            'color': 'white',
            'font-size': '24px',
            'font-weight': 'bold',
            'box-shadow': '0 4px 8px rgba(0,0,0,0.3)',
            'z-index': '1000',
            'display': 'flex',
            'align-items': 'center',
            'justify-content': 'center'
        },
        n_clicks=0
    )
], style={'height': '100vh', 'overflow': 'hidden'})

# Authentication callbacks
@app.callback(
    Output('login-modal', 'is_open'),
    [Input('login-btn', 'n_clicks'), Input('login-cancel-btn', 'n_clicks')],
    [State('login-modal', 'is_open')]
)
def toggle_login_modal(login_clicks, cancel_clicks, is_open):
    """Toggle login modal visibility"""
    if login_clicks or cancel_clicks:
        return not is_open
    return is_open

@app.callback(
    [Output('auth-state-store', 'data'),
     Output('user-data-store', 'data'),
     Output('login-error-message', 'children'),
     Output('login-email', 'value'),
     Output('login-password', 'value')],
    [Input('login-submit-btn', 'n_clicks')],
    [State('login-email', 'value'),
     State('login-password', 'value')]
)
def handle_login(n_clicks, email, password):
    """Handle login form submission"""
    if n_clicks and email and password:
        try:
            # Create event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Authenticate user
            auth_result = loop.run_until_complete(authenticate_user_direct(email, password))
            
            if auth_result["success"]:
                # Update MCP session with auth tokens
                login_data = auth_result["data"]
                success = loop.run_until_complete(update_mcp_session_auth(
                    login_data["localId"],
                    login_data["idToken"],
                    login_data["refreshToken"],
                    int(login_data["expiresIn"])
                ))
                
                if success:
                    # Return success state
                    return (
                        {"authenticated": True}, 
                        {"user_id": login_data["localId"], "email": email},
                        "",  # Clear error message
                        "",  # Clear email field
                        ""   # Clear password field
                    )
                else:
                    return (
                        {"authenticated": False}, 
                        {},
                        "Failed to update session. Please try again.",
                        email, 
                        ""
                    )
            else:
                return (
                    {"authenticated": False}, 
                    {},
                    f"Login failed: {auth_result['error']}",
                    email, 
                    ""
                )
                
        except Exception as e:
            return (
                {"authenticated": False}, 
                {},
                f"An error occurred: {str(e)}",
                email, 
                ""
            )
    
    # No change if button not clicked or missing credentials
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    [Output('auth-state-store', 'data', allow_duplicate=True),
     Output('user-data-store', 'data', allow_duplicate=True)],
    [Input('logout-btn', 'n_clicks')],
    prevent_initial_call=True
)
def handle_logout(n_clicks):
    """Handle logout button click"""
    if n_clicks:
        try:
            # Create event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Clear authentication
            success = loop.run_until_complete(logout_user())
            
            if success:
                return {"authenticated": False}, {}
            else:
                # Even if logout failed, clear the UI state
                return {"authenticated": False}, {}
                
        except Exception as e:
            print(f"Logout error: {str(e)}")
            return {"authenticated": False}, {}
    
    return dash.no_update, dash.no_update

@app.callback(
    [Output('auth-status-indicator', 'children'),
     Output('login-modal', 'is_open', allow_duplicate=True)],
    [Input('auth-state-store', 'data'),
     Input('user-data-store', 'data')],
    prevent_initial_call=True
)
def update_auth_status_display(auth_state, user_data):
    """Update the authentication status display"""
    if auth_state.get("authenticated", False):
        user_email = user_data.get("email", "Unknown User")
        return [
            html.Small(f"✅ Logged in as {user_email}", 
                      style={'color': '#28a745', 'font-weight': 'bold'}),
            html.Br(),
            dbc.ButtonGroup([
                dbc.Button("Login", id="login-btn", size="sm", color="primary", 
                         style={'display': 'none'}),
                dbc.Button("Logout", id="logout-btn", size="sm", color="secondary")
            ], size="sm", style={'margin-top': '5px'})
        ], False  # Close login modal
    else:
        return [
            html.Small("🔒 Not logged in", 
                      style={'color': '#dc3545', 'font-weight': 'bold'}),
            html.Br(),
            dbc.ButtonGroup([
                dbc.Button("Login", id="login-btn", size="sm", color="primary"),
                dbc.Button("Logout", id="logout-btn", size="sm", color="secondary", 
                         style={'display': 'none'})
            ], size="sm", style={'margin-top': '5px'})
        ], dash.no_update  # Don't change modal state

# Callback to check authentication status on page load
@app.callback(
    [Output('auth-state-store', 'data', allow_duplicate=True),
     Output('user-data-store', 'data', allow_duplicate=True)],
    [Input('auth-state-store', 'id')],  # Trigger on page load
    prevent_initial_call='initial_duplicate'
)
def check_auth_on_load(_):
    """Check authentication status when page loads"""
    try:
        auth_status = get_current_auth_status()
        if auth_status.get("authenticated", False):
            return (
                {"authenticated": True}, 
                {"user_id": auth_status.get("user_id", ""), "email": ""}  # We don't store email in session metadata
            )
        else:
            return {"authenticated": False}, {}
    except Exception as e:
        print(f"Auth check error: {str(e)}")
        return {"authenticated": False}, {}


# Callback for minimize/expand functionality (unchanged)
@app.callback(
    [Output('left-column', 'width'),
     Output('right-column', 'width'),
     Output('minimize-button', 'children')],
    [Input('minimize-button', 'n_clicks')]
)
def toggle_right_panel(n_clicks):
    if n_clicks % 2 == 1:  # Odd clicks = minimized
        return 12, 0, "+"  # Left column full width, right hidden, show expand button
    else:  # Even clicks = expanded
        return 8, 4, "−"   # Normal layout, show minimize button

# Main callback function with memory support and report display
@app.callback(
    [Output('conversation-div', 'children'),
     Output('query-input', 'value'),
     Output('report-content', 'children'),
     Output('report-status', 'children'),
     Output('interactive-plots-content', 'children'),
     Output('interactive-data-available', 'data')],
    [Input('send-button', 'n_clicks'),
     Input('query-input', 'n_submit')],
    [State('query-input', 'value'),
     State('conversation-div', 'children'),
     State('report-content', 'children'),
     State('report-status', 'children'),
     State('interactive-plots-content', 'children'),
     State('interactive-data-available', 'data')]
)
def process_query(n_clicks, n_submit, query, current_conversation, current_report_content, current_report_status, current_interactive_plots, current_data_available):
    if (n_clicks and n_clicks > 0) or n_submit:
        if query and query.strip():
            # Check authentication status first
            auth_status = get_current_auth_status()
            if not auth_status.get("authenticated", False):
                # Add authentication error message to conversation
                auth_error_message = html.Div([
                    html.Div("System:", style={
                        'font-weight': 'bold', 
                        'color': '#dc3545',
                        'margin-bottom': '5px'
                    }),
                    html.Div("🔒 Please log in first to use the AI assistant.", style={
                        'background-color': '#f8d7da',
                        'padding': '10px',
                        'border-radius': '10px',
                        'color': '#721c24'
                    })
                ], style={'margin-bottom': '15px'})
                
                user_message = html.Div([
                    html.Div("Me:", style={
                        'font-weight': 'bold', 
                        'color': '#007bff',
                        'margin-bottom': '5px'
                    }),
                    html.Div(query, style={
                        'background-color': '#e3f2fd',
                        'padding': '10px',
                        'border-radius': '10px',
                        'margin-bottom': '10px'
                    })
                ], style={'margin-bottom': '15px'})
                
                if current_conversation is None:
                    current_conversation = []
                
                updated_conversation = [auth_error_message, user_message] + current_conversation
                
                # Return current state with auth error
                preserved_report = current_report_content if current_report_content is not None else report_app_ui_builder._create_empty_state()
                preserved_status = current_report_status if current_report_status is not None else report_app_ui_builder.create_report_status_indicator('empty')
                preserved_interactive_plots = current_interactive_plots if current_interactive_plots is not None else report_app_ui_builder._create_interactive_plots_placeholder()
                preserved_data_available = current_data_available if current_data_available is not None else False
                
                return updated_conversation, "", preserved_report, preserved_status, preserved_interactive_plots, preserved_data_available
                
            try:
                # Add user message to conversation
                user_message = html.Div([
                    html.Div("Me:", style={
                        'font-weight': 'bold', 
                        'color': '#007bff',
                        'margin-bottom': '5px'
                    }),
                    html.Div(query, style={
                        'background-color': '#e3f2fd',
                        'padding': '10px',
                        'border-radius': '10px',
                        'margin-bottom': '10px'
                    })
                ], style={'margin-bottom': '15px'})
                
                # Process MCP client query with memory and file handle support
                async def run_query_with_memory():
                    client = await ensure_client_connected()
                    if not client:
                        return {"response": "Error: Could not connect to MCP client", "raw_content": ""}
                    
                    try:
                        # Use persistent thread ID for conversation continuity
                        # Use the new method that returns both response and raw content
                        result = await client.analyze_territories_with_file_handle(query, thread_id=get_thread_id())
                        return result
                    except Exception as e:
                        return {"response": f"Error processing query: {str(e)}", "raw_content": ""}
                
                # Create new event loop if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(run_query_with_memory())
                
                # Extract response and raw content
                if isinstance(result, dict):
                    agent_response = str(result.get('response', ''))
                    raw_content = result.get('raw_content', '')
                else:
                    agent_response = str(result)
                    raw_content = str(result)
                
                # Add agent message to conversation
                agent_message = html.Div([
                    html.Div("Agent:", style={
                        'font-weight': 'bold', 
                        'color': '#28a745',
                        'margin-bottom': '5px'
                    }),
                    html.Div(agent_response, style={
                        'background-color': '#f8f9fa',
                        'padding': '10px',
                        'border-radius': '10px',
                        'white-space': 'pre-wrap'
                    })
                ], style={'margin-bottom': '15px'})
                
                # Update conversation history
                if current_conversation is None:
                    current_conversation = []
                
                updated_conversation = [agent_message, user_message] + current_conversation
                
                # Handle report display - Start with current report state (preserve existing reports)
                report_content = current_report_content if current_report_content is not None else report_app_ui_builder._create_empty_state()
                report_status = current_report_status if current_report_status is not None else report_app_ui_builder.create_report_status_indicator('empty')
                interactive_plots_content = current_interactive_plots if current_interactive_plots is not None else report_app_ui_builder._create_interactive_plots_placeholder()
                interactive_data_available = current_data_available if current_data_available is not None else False
                
                # Try to extract file handle from structured output and display report
                structured_output = result.get('structured_output')
                if structured_output:
                    print(f"[DEBUG] Processing structured output for file handle extraction...")
                    file_handle = report_data_manager.parse_file_handle_from_response(structured_output)
                    if file_handle:
                        print(f"📄 Found file handle: {file_handle}")
                        # Try to read the report
                        md_content = report_data_manager.read_md_report(file_handle)
                        if md_content:
                            report_content = report_app_ui_builder.format_markdown_for_dash(md_content)
                            # Get report metadata
                            metadata = report_data_manager.extract_report_metadata(file_handle)
                            report_status = report_app_ui_builder.create_report_status_indicator('loaded', metadata)
                            print(f"✅ Report loaded and displayed")
                            
                            # Check for interactive data files
                            data_files = report_data_manager.get_data_files(structured_output)
                            print(f"[DEBUG] Checking for data files from report handler: {len(data_files)} files found")
                            
                            if data_files:
                                print(f"[DEBUG] Found data files for interactive plotting: {list(data_files.keys())}")
                                success, plot_info = load_and_create_plots(data_files)
                                if success:
                                    interactive_plots_content = report_app_ui_builder.create_interactive_plots_layout(
                                        plot_info['variables'], 
                                        plot_info['default_variable']
                                    )
                                    interactive_data_available = True
                                    print(f"✅ Interactive plots ready with variables: {list(plot_info['variables'].keys())}")
                                else:
                                    print(f"❌ Failed to load interactive plots: {plot_info.get('error', 'Unknown error')}")
                            else:
                                # FALLBACK: Try to find data files in the MCP client session
                                print("[DEBUG] No data files from report handler, trying fallback method...")
                                try:
                                    # Get the MCP client to check for recent territory data
                                    client = ensure_client_connected()
                                    if client and hasattr(client, 'session_manager'):
                                        # This is a more advanced fallback - we could implement this later
                                        print("[DEBUG] Could implement session-based data file retrieval here")
                                    
                                    # For now, let's try to construct expected file paths based on recent session data
                                    # Check if there are recent GeoJSON files in static/data
                                    import glob

                                    static_data_dir = Config.STATIC_DATA_DIR
                                    if static_data_dir.exists():
                                        # Get the most recent set of territory data files
                                        geojson_files = list(static_data_dir.glob("*_*.geojson"))
                                        if len(geojson_files) >= 3:
                                            # Group by request ID (assuming format: requestid_type.geojson)
                                            file_groups = {}
                                            for file_path in geojson_files:
                                                parts = file_path.stem.split('_', 1)
                                                if len(parts) == 2:
                                                    request_id, file_type = parts
                                                    if request_id not in file_groups:
                                                        file_groups[request_id] = {}
                                                    file_groups[request_id][file_type] = str(file_path)
                                            
                                            # Get the most recent complete set
                                            for request_id, files in sorted(file_groups.items(), reverse=True):
                                                if all(key in files for key in ['grid_data', 'places_data', 'boundaries']):
                                                    fallback_data_files = {
                                                        'grid_data': files['grid_data'],
                                                        'places_data': files['places_data'],
                                                        'boundaries': files['boundaries']
                                                    }
                                                    print(f"[DEBUG] Found fallback data files: {list(fallback_data_files.keys())}")
                                                    success, plot_info = load_and_create_plots(fallback_data_files)
                                                    if success:
                                                        interactive_plots_content = report_app_ui_builder.create_interactive_plots_layout(
                                                            plot_info['variables'], 
                                                            plot_info['default_variable']
                                                        )
                                                        interactive_data_available = True
                                                        print(f"✅ Interactive plots ready using fallback data with variables: {list(plot_info['variables'].keys())}")
                                                    break
                                    
                                except Exception as e:
                                    print(f"❌ Error in fallback data file detection: {str(e)}")
                                    import traceback
                                    traceback.print_exc()
                        else:
                            print(f"❌ Could not read report from handle: {file_handle}")
                            report_status = report_app_ui_builder.create_report_status_indicator('error')
                    else:
                        print("ℹ️ No file handle found in response")
                
                return updated_conversation, "", report_content, report_status, interactive_plots_content, interactive_data_available
                
            except Exception as e:
                # Add error message to conversation
                error_message = html.Div([
                    html.Div("Agent:", style={
                        'font-weight': 'bold', 
                        'color': '#dc3545',
                        'margin-bottom': '5px'
                    }),
                    html.Div(f"Error: {str(e)}", style={
                        'background-color': '#f8d7da',
                        'padding': '10px',
                        'border-radius': '10px',
                        'color': '#721c24'
                    })
                ], style={'margin-bottom': '15px'})
                
                user_message = html.Div([
                    html.Div("Me:", style={
                        'font-weight': 'bold', 
                        'color': '#007bff',
                        'margin-bottom': '5px'
                    }),
                    html.Div(query, style={
                        'background-color': '#e3f2fd',
                        'padding': '10px',
                        'border-radius': '10px',
                        'margin-bottom': '10px'
                    })
                ], style={'margin-bottom': '15px'})
                
                if current_conversation is None:
                    current_conversation = []
                
                updated_conversation = [error_message, user_message] + current_conversation
                
                # Return error state for report display
                error_report_content = report_app_ui_builder.create_error_display(str(e))
                error_report_status = report_app_ui_builder.create_report_status_indicator('error')
                error_interactive_plots = report_app_ui_builder._create_interactive_plots_placeholder()
                
                return updated_conversation, "", error_report_content, error_report_status, error_interactive_plots, False
    
    # Return current state if no valid input - preserve existing reports
    preserved_report = current_report_content if current_report_content is not None else report_app_ui_builder._create_empty_state()
    preserved_status = current_report_status if current_report_status is not None else report_app_ui_builder.create_report_status_indicator('empty')
    preserved_interactive_plots = current_interactive_plots if current_interactive_plots is not None else report_app_ui_builder._create_interactive_plots_placeholder()
    preserved_data_available = current_data_available if current_data_available is not None else False
    return current_conversation or [], query or "", preserved_report, preserved_status, preserved_interactive_plots, preserved_data_available

# Callback to enable/disable interactive plots tab based on data availability
@app.callback(
    Output('report-tabs', 'children'),
    [Input('interactive-data-available', 'data')]
)
def update_tab_state(data_available):
    print(f"🔄 Updating tab state - Interactive data available: {data_available}")
    if data_available:
        print("✅ Enabling interactive plots tab")
        return [
            dbc.Tab(label="📄 Report", tab_id="static-report", active_tab_style={"font-weight": "bold"}),
            dbc.Tab(label="📊 Interactive Maps", tab_id="interactive-plots", active_tab_style={"font-weight": "bold"})
        ]
    else:
        print("⏸️ Disabling interactive plots tab")
        return [
            dbc.Tab(label="📄 Report", tab_id="static-report", active_tab_style={"font-weight": "bold"}),
            dbc.Tab(label="📊 Interactive Maps", tab_id="interactive-plots", disabled=True, active_tab_style={"font-weight": "bold"})
        ]

# Callback for tab switching between static report and interactive plots
@app.callback(
    [Output('report-content', 'style'),
     Output('interactive-plots-content', 'style')],
    [Input('report-tabs', 'active_tab')]
)
def switch_report_tabs(active_tab):
    print(f"🔄 Switching tabs - Active tab: {active_tab}")
    if active_tab == 'interactive-plots':
        print("📊 Showing interactive plots content")
        return {'display': 'none'}, {'display': 'block', 'height': 'calc(100vh - 200px)', 'overflow-y': 'auto', 'padding': '20px', 'background-color': 'white', 'border': '1px solid #dee2e6', 'border-radius': '8px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'}
    else:
        print("📄 Showing static report content")
        return {'height': 'calc(100vh - 200px)', 'overflow-y': 'auto', 'padding': '20px', 'background-color': 'white', 'border': '1px solid #dee2e6', 'border-radius': '8px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'}, {'display': 'none'}

# Callback for updating the choropleth map based on selected variable
@app.callback(
    Output('interactive-choropleth-map', 'figure'),
    [Input('interactive-variable-dropdown', 'value')],
    prevent_initial_call=True
)
def update_interactive_choropleth(selected_variable):
    print(f"🔄 Updating choropleth map with variable: {selected_variable}")
    if not selected_variable:
        print("⚠️ No variable selected for choropleth")
        return {}
    
    try:
        print(f"🎯 Creating choropleth map for: {selected_variable}")
        fig = plotter.create_choropleth_map(selected_variable)
        if fig:
            print("✅ Choropleth map created successfully")
            return fig
        else:
            print("❌ Choropleth map creation returned None")
            return {}
    except Exception as e:
        print(f"❌ Error updating choropleth map: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

# Callback for updating the scatter map (supermarket locations)
@app.callback(
    Output('interactive-scatter-map', 'figure'),
    [Input('interactive-variable-dropdown', 'value')],  # Trigger on dropdown change
    prevent_initial_call=True
)
def update_interactive_scatter(selected_variable):
    print(f"🔄 Updating scatter map (triggered by variable: {selected_variable})")
    try:
        print("🎯 Creating scatter map for supermarkets")
        fig = plotter.create_supermarket_scatter_map()
        if fig:
            print("✅ Scatter map created successfully")
            return fig
        else:
            print("❌ Scatter map creation returned None")
            return {}
    except Exception as e:
        print(f"❌ Error updating scatter map: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=False)