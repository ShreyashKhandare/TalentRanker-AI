# Use lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- CACHE BREAKER ----
ARG CACHE_BUST=1

# Install Python dependencies (force fresh + correct versions)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Copy FULL project
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port (using environment variable with fallback)
EXPOSE ${PORT:-10000}

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "${PORT:-10000}"]

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/health || exit 1
