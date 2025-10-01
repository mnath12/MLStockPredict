FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages that might be needed
RUN pip install --no-cache-dir \
    jupyter \
    ipykernel \
    pytest

# Create a non-root user for development
RUN useradd -m -u 1000 developer && \
    chown -R developer:developer /app
USER developer

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["bash"] 