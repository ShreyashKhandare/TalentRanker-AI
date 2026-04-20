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

# Expose Render port
EXPOSE 10000

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
