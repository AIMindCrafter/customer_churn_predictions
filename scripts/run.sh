#!/bin/bash
# ═══════════════════════════════════════════════════════
# ChurnGuard — Quick Start Script
# ═══════════════════════════════════════════════════════

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "⚡ ChurnGuard — Production Churn Prediction"
echo "════════════════════════════════════════════"

# ── Virtual environment setup ─────────────────────────
VENV_DIR="$PROJECT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✅ Virtual environment created at $VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo "✅ Virtual environment activated"

# Install dependencies if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
fi

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✅ Environment loaded from .env"
fi

# Check if model artifacts exist
if [ ! -f "./artifacts/best_model_bayesian_rf.pkl" ]; then
    echo ""
    echo "⚠️  Model artifacts not found. Training models..."
    echo "   This will take 5-10 minutes."
    echo ""
    python3 scripts/train_pipeline.py --skip-shap
    echo ""
    echo "✅ Training complete!"
fi

echo ""
echo "🚀 Starting API server on http://localhost:${APP_PORT:-8000}"
echo "   Press Ctrl+C to stop"
echo ""

python3 -m uvicorn src.api:app \
    --host "${APP_HOST:-0.0.0.0}" \
    --port "${APP_PORT:-8000}" \
    --reload
