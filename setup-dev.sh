#!/bin/bash
# Setup script for PE-DB project

set -e

echo "=== PE-DB Project Setup ==="

# Check Python version
echo "Checking Python version..."
python --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install pe-common package in editable mode
echo "Installing pe-common package..."
pip install -e packages/pe-common

# Install PE-DB service dependencies
echo "Installing PE-DB service dependencies..."
pip install -r services/pe-db/requirements.txt

# Check if datasets directory exists
if [ ! -d "datasets" ]; then
    echo "Warning: datasets directory not found"
    echo "Please ensure your datasets are placed in the 'datasets' directory"
else
    echo "Datasets directory found"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run PE Database service:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run: cd services/pe-db && uvicorn app.main:app --reload"
echo ""
echo "Or use Docker:"
echo "  docker-compose up pe-db"
echo ""
echo "API will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
