#!/bin/bash
# Quick setup script for AI models synchronization

set -e  # Exit on error

echo "===================================================================="
echo "  Petrolimex AI Backend - Model Synchronization"
echo "===================================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "scripts/sync_models_to_db.py" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "Step 1: Synchronizing AI models to database..."
echo "--------------------------------------------------------------------"
python3 scripts/sync_models_to_db.py --sync

echo ""
echo "Step 2: Listing all models in database..."
echo "--------------------------------------------------------------------"
python3 scripts/sync_models_to_db.py --list

echo ""
echo "===================================================================="
echo "  ✓ Setup Complete!"
echo "===================================================================="
echo ""
echo "Next steps:"
echo "  1. Verify model weights exist in src/ai_models/model_weights/"
echo "  2. Assign models to cameras via API or database"
echo "  3. Configure additional parameters if needed"
echo ""
echo "To update parameters:"
echo "  python3 scripts/update_model_parameters.py --help"
echo ""
