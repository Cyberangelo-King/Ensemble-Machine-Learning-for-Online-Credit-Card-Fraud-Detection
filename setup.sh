#!/bin/bash
# Quick setup script to generate demo model and launch Streamlit

echo "🚀 Fraud Detection Dashboard - Quick Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3.10 -m venv .venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Create demo model and data
echo "🤖 Creating demo stacking model..."
python scripts/create_demo_model.py

# Check if model was created successfully
if [ -f "artifacts/stacking_model.joblib" ] && [ -f "data/creditcard.csv" ]; then
    echo ""
    echo "✅ Setup complete!"
    echo ""
    echo "📊 Starting Streamlit dashboard..."
    streamlit run app.py
else
    echo "❌ Error: Model or data files not created"
    exit 1
fi
