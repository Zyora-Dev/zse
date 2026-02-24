# ZSE - Z Server Engine
# Ultra memory-efficient LLM inference engine
#
# Build: docker build -t zse .
# Run:   docker run -p 8000:8000 zse
# GPU:   docker run --gpus all -p 8000:8000 zse

# =============================================================================
# Base Stage - Common dependencies
# =============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 zse
WORKDIR /app

# =============================================================================
# Builder Stage - Install Python packages
# =============================================================================
FROM base as builder

# Install Python dependencies
COPY pyproject.toml setup.py ./
COPY zse/ ./zse/

RUN pip install --user -e .

# =============================================================================
# Production Stage - CPU Only
# =============================================================================
FROM base as cpu

# Copy installed packages from builder
COPY --from=builder /root/.local /home/zse/.local

# Copy application code
COPY --chown=zse:zse . .

# Set PATH for user-installed packages
ENV PATH="/home/zse/.local/bin:$PATH"

# Switch to non-root user
USER zse

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - start server
CMD ["zse", "serve", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# GPU Stage - CUDA Support
# =============================================================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Create app user
RUN useradd -m -u 1000 zse
WORKDIR /app

# Copy application code
COPY --chown=zse:zse . .

# Install Python dependencies
RUN pip install -e . && \
    pip install torch --index-url https://download.pytorch.org/whl/cu121

# Switch to non-root user
USER zse

# Set PATH
ENV PATH="/home/zse/.local/bin:$PATH"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["zse", "serve", "--host", "0.0.0.0", "--port", "8000"]
