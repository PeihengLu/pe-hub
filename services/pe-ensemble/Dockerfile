FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install pe-common package as a Python library
COPY packages/pe-common /tmp/pe-common
RUN pip install --no-cache-dir /tmp/pe-common && \
    rm -rf /tmp/pe-common

# Copy and install service dependencies
COPY services/pe-ensemble/pyproject.toml /app/
COPY services/pe-ensemble/app /app/app

# Install dependencies from pyproject.toml
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    pandas>=1.3.0 \
    numpy>=1.21.0 \
    scikit-learn>=1.0.0 \
    scipy>=1.7.0 \
    requests>=2.28.0 \
    pydantic>=2.0.0 \
    python-multipart>=0.0.6 \
    tqdm>=4.65.0 \
    matplotlib>=3.5.0

# Copy vendor models (these will be mounted as volumes in docker-compose)
COPY vendor/models /app/vendor/models

# Copy datasets (for training access)
COPY datasets /app/datasets

# Set Python path to include vendor models
# Note: pe-common is installed as a package and available via standard imports
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/vendor/models"

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
