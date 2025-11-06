# Use slim image for smaller size and better security
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies required for psycopg2-binary and other packages
RUN apt-get update && \
    apt-get install -y \
    git \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    postgresql-client \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy uv configuration files
COPY pyproject.toml ./

# Install dependencies with uv
RUN uv sync

# Copy all DashApp files (self-contained now)
COPY . /app/

# Create required directories for runtime
# These will be mounted as volumes in docker-compose, but create them for local development
RUN mkdir -p /app/reports && \
    mkdir -p /app/static && \
    mkdir -p /app/.sessions && \
    chmod -R 777 /app/.sessions && \
    chmod -R 777 /app/reports && \
    chmod -R 777 /app/static

EXPOSE 8050

# Run DashApp
CMD ["uv", "run", "python", "dash_app.py"]