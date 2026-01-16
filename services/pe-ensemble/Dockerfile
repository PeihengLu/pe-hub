FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy pe-common package
COPY packages/pe-common /app/packages/pe-common
RUN pip install --no-cache-dir /app/packages/pe-common

# Copy service files
COPY services/pe-ensemble/pyproject.toml /app/
COPY services/pe-ensemble/pe-ensemble /app/pe-ensemble

# Install service dependencies
RUN pip install --no-cache-dir -e .

# Copy vendor models
COPY vendor/models /app/models

# Copy source utilities
COPY src /app/src

# Expose port
EXPOSE 8001

# Run the application
CMD ["uvicorn", "pe_ensemble.main:app", "--host", "0.0.0.0", "--port", "8001"]
