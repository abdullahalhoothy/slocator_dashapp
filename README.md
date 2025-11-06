# DashApp - Geospatial Intelligence Dashboard

A Dash-based web application for visualizing and interacting with territory optimization and geospatial analysis reports powered by AI.

## Features

- **Interactive Reports**: View AI-generated territory analysis reports with rich markdown formatting
- **Interactive Maps**: Explore territory data with dynamic Plotly visualizations
- **AI Assistant**: Chat with an AI agent that can generate new analyses and answer questions about existing reports
- **Authentication**: Secure login system integrated with Firebase
- **MCP Integration**: Connects to backend services via Model Context Protocol (MCP)

## Architecture

DashApp is a standalone application that:
- Runs independently as a Dash/Flask web server (port 8050)
- Communicates with backend services via HTTP API
- Includes its own MCP server for tool orchestration
- Manages its own reports, sessions, and static files

## Directory Structure

```
DashApp/
├── dash_app.py                  # Main Dash application
├── report_agent.py              # MCP client for AI interactions
├── config.py                    # Configuration settings
├── ReportAppUIBuilder.py        # UI components
├── ReportDataManager.py         # Report file management
├── interactive_plots.py         # Plotly visualization logic
├── system_prompts.py            # AI system prompts
├── tool_bridge_mcp_server/      # MCP server and tools
│   ├── mcp_server.py           # MCP server implementation
│   ├── tools/                  # Analysis tools
│   └── sessions/               # Session storage
├── Backend/                     # Business type configurations
├── all_types/                   # Data type definitions
├── backend_common/              # Common utilities
├── secrets/                     # API keys (not in git)
├── static/                      # Static files
│   ├── plots/                  # Generated visualizations
│   └── data/                   # GeoJSON data files
└── reports/                     # Generated reports

```

## Prerequisites

- Python 3.13+
- UV package manager (or pip)
- Firebase account (for authentication)
- Gemini API key (for AI analysis)
- Backend API service (optional, for full functionality)

## Installation

### Using UV (Recommended)

```bash
# Install dependencies
uv sync

# Run the application
uv run python dash_app.py
```

### Using Docker

```bash
# Build and run with docker-compose
docker-compose up --build
```

The application will be available at `http://localhost:8050`

## Configuration

### Environment Variables

Create a `.env` file or set these environment variables:

```bash
# Backend API URL (optional)
BACKEND_URL=http://localhost:8000

# Flask secret key for sessions
FLASK_SECRET_KEY=your-secret-key-here
```

### API Keys

Place your API keys in `secrets/secrets_llm.json`:

```json
{
  "gemini_api_key": "your-gemini-api-key-here"
}
```

### Config Settings

Edit `config.py` to customize:
- Model settings (default: gemini-2.5-flash)
- Report directories
- MCP server configuration

## Usage

1. **Start the Application**:
   ```bash
   uv run python dash_app.py
   ```

2. **Login**: Click the "Login" button and enter your Firebase credentials

3. **Generate Analysis**:
   - Type a request in the chat box, e.g.:
     - "Create 6 sales territories for supermarkets in Riyadh"
     - "Analyze optimal locations for 5 new pharmacies in Jeddah"

4. **View Reports**: Reports appear automatically in the left panel

5. **Explore Interactive Maps**: Switch to the "Interactive Maps" tab to visualize territory data

## Development

### Project Structure

- **dash_app.py**: Main application with Dash layout and callbacks
- **report_agent.py**: SimpleMCPClient for AI agent interactions
- **tool_bridge_mcp_server/**: MCP server with analysis tools:
  - `optimize_sales_territories`: Create sales territories
  - `hub_expansion_analyzer`: Find optimal facility locations
  - `generate_territory_report`: Generate formatted reports
  - `report_analysis`: Answer questions about reports
  - `user_login`: Authentication tool

### Adding New Tools

1. Create a new tool file in `tool_bridge_mcp_server/tools/`
2. Register it in `mcp_server.py`
3. Update system prompts in `system_prompts.py` if needed

## Docker Deployment

### Standalone Deployment

```bash
# Build the image
docker build -t dashapp:latest .

# Run with environment variables
docker run -p 8050:8050 \
  -e BACKEND_URL=http://backend:8000 \
  -e FLASK_SECRET_KEY=your-secret-key \
  dashapp:latest
```

### With Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f dashapp

# Stop services
docker-compose down
```

## Connecting to Backend

DashApp can work standalone or connect to the backend2-1 service:

- **Standalone**: All analysis runs locally via MCP tools
- **With Backend**: Can access additional backend endpoints for data fetching

Set `BACKEND_URL` environment variable to connect to backend:
```bash
BACKEND_URL=http://localhost:8000  # Local backend
BACKEND_URL=http://backend:8000    # Docker network
```

## Troubleshooting

### Reports Not Loading
- Check that `reports/` directory exists and is writable
- Verify MCP server is running (check console logs)
- Ensure API keys are configured correctly

### Interactive Maps Not Showing
- Verify GeoJSON data files exist in `static/data/`
- Check that analysis completed successfully
- Look for errors in browser console

### Authentication Errors
- Verify Firebase credentials
- Check BACKEND_URL is set correctly
- Ensure `tool_bridge_mcp_server/sessions/` is writable

### MCP Connection Issues
- Check Python executable path in `config.py`
- Verify `tool_bridge_mcp_server/mcp_server.py` exists
- Review MCP server logs in `tool_bridge_mcp_server/logs/`

## License

This project is part of the backend2-1 system for geospatial intelligence analysis.

## Support

For issues or questions, please check the logs:
- Application logs: Console output
- MCP server logs: `tool_bridge_mcp_server/logs/`
- Session data: `tool_bridge_mcp_server/sessions/`