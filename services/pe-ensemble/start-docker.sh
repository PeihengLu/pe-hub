#!/bin/bash

# Quick start script for PE-Ensemble Docker deployment

set -e

echo "=================================="
echo "PE-Ensemble Docker Deployment"
echo "=================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Navigate to project root
cd "$(dirname "$0")/../.."

echo "Building and starting PE-Ensemble service..."
echo ""

# Build and start
docker-compose -f services/pe-ensemble/docker-compose.yml up --build -d

echo ""
echo "Waiting for service to be healthy..."
sleep 10

# Check health
if curl -f http://localhost:8001/health > /dev/null 2>&1; then
    echo ""
    echo "✓ PE-Ensemble is running!"
    echo ""
    echo "Access points:"
    echo "  - API:          http://localhost:8001"
    echo "  - Health:       http://localhost:8001/health"
    echo "  - API Docs:     http://localhost:8001/docs"
    echo "  - ReDoc:        http://localhost:8001/redoc"
    echo ""
    echo "Available models:"
    curl -s http://localhost:8001/models | python3 -m json.tool
    echo ""
    echo "View logs:"
    echo "  docker-compose -f services/pe-ensemble/docker-compose.yml logs -f"
    echo ""
    echo "Stop service:"
    echo "  docker-compose -f services/pe-ensemble/docker-compose.yml down"
    echo ""
else
    echo ""
    echo "⚠ Service started but health check failed. Checking logs..."
    echo ""
    docker-compose -f services/pe-ensemble/docker-compose.yml logs --tail=50 pe-ensemble
    echo ""
    echo "Service may still be starting. Check status with:"
    echo "  docker-compose -f services/pe-ensemble/docker-compose.yml ps"
    echo "  docker-compose -f services/pe-ensemble/docker-compose.yml logs -f"
fi
