FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install pe-common package as a Python library
COPY packages/pe-common /tmp/pe-common
RUN pip install --no-cache-dir /tmp/pe-common && \
    rm -rf /tmp/pe-common

# Copy and install service requirements
COPY services/pe-db/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy service code
COPY services/pe-db/app /app/app

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
