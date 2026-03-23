#!/bin/bash
# Run the complete churn prediction pipeline

echo "================================"
echo "Churn Prediction Pipeline"
echo "================================"

cd "$(dirname "$0")/.."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if requirements are installed
echo "Checking dependencies..."
python3 -c "import pandas, sklearn, shap" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements_prod.txt
fi

# Run pipeline
echo ""
echo "Running pipeline..."
python3 main.py
