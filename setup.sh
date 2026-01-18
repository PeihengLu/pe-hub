#!/bin/bash

# Setup script for PE-DB project
# This script prepares the datasets and environment

set -e  # Exit on error

echo "======================================"
echo "PE-DB Project Setup"
echo "======================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo ""
echo "Step 1: Installing project packages..."
pip install -e .
pip install -e packages/pe-common
echo "✓ Project packages installed"

echo ""
echo "Step 2: Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "⚠ Warning: requirements.txt not found"
fi

echo ""
echo "Step 3: Restoring PRIDICT datasets..."
if [ -f "datasets/dataprep/restore_pridict.py" ]; then
    python3 datasets/dataprep/restore_pridict.py -a restore
    echo "✓ PRIDICT datasets restored"
else
    echo "⚠ Warning: PRIDICT restoration script not found, skipping..."
fi

echo ""
echo "Step 4: Converting DeepPrime datasets..."
if [ -f "datasets/dataprep/standarzied_data.py" ]; then
    python3 datasets/dataprep/standarzied_data.py
    echo "✓ DeepPrime datasets converted"
else
    echo "⚠ Warning: DeepPrime conversion script not found, skipping..."
fi


echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Run 'docker-compose -f docker-compose.full.yml up' to start all services"
echo "2. Or run 'make docker-up' if you have make installed"
echo "3. Access services at:"
echo "   - PE Database API: http://localhost:8000"
echo "   - PE Ensemble API: http://localhost:8001"
echo ""
