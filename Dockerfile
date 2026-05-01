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

# Copy the rest of the source tree
COPY . /app/

# Create required runtime directories (also volume-mounted in docker-compose)
RUN mkdir -p /app/reports /app/static /app/.sessions && \
    chmod -R 777 /app/.sessions /app/reports /app/static

EXPOSE 8050

# Run DashApp via the package entrypoint
CMD ["uv", "run", "python", "-m", "slocator"]