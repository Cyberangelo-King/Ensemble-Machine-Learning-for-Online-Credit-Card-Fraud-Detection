#!/usr/bin/env bash
# =============================================================================
# setup.sh -- One-command setup for the Fraud Detection Streamlit demo
# =============================================================================
set -euo pipefail

# ── Colour helpers (ASCII-safe) ───────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'   # No Colour

info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  Ensemble ML - Credit Card Fraud Detection"
echo "  Setup Script"
echo "======================================================"
echo ""

# ── 1. Python version check ───────────────────────────────────────────────────
info "Checking Python version..."
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 9 ]; then
            PYTHON_CMD="$cmd"
            info "Found $cmd $VER -- OK"
            break
        else
            warn "Found $cmd $VER but Python >= 3.9 is required."
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    error "Python 3.9 or higher not found. Please install Python and re-run."
    exit 1
fi

# ── 2. Create virtual environment ─────────────────────────────────────────────
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment at $VENV_DIR ..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
else
    info "Virtual environment already exists at $VENV_DIR -- skipping creation."
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Virtual environment activated."

# ── 3. Upgrade pip ────────────────────────────────────────────────────────────
info "Upgrading pip..."
pip install --quiet --upgrade pip

# ── 4. Install dependencies ───────────────────────────────────────────────────
info "Installing Python dependencies from requirements.txt..."
if [ ! -f "requirements.txt" ]; then
    error "requirements.txt not found. Are you running this from the repo root?"
    exit 1
fi
pip install --quiet -r requirements.txt
info "Dependencies installed."

# ── 5. Generate synthetic data (if not already present) ───────────────────────
DATA_FILE="data/creditcard.csv"
if [ ! -f "$DATA_FILE" ]; then
    info "Generating synthetic dataset..."
    "$PYTHON_CMD" scripts/generate_synthetic_data.py
    info "Synthetic dataset written to $DATA_FILE."
else
    info "Dataset already exists at $DATA_FILE -- skipping generation."
fi

# ── 6. Create demo model (if not already present) ─────────────────────────────
MODEL_FILE="artifacts/fraud_model.pkl"
if [ ! -f "$MODEL_FILE" ]; then
    info "Training demo model..."
    "$PYTHON_CMD" scripts/create_demo_model.py
    info "Demo model written to $MODEL_FILE."
else
    info "Demo model already exists at $MODEL_FILE -- skipping training."
fi

# ── 7. Launch Streamlit app ───────────────────────────────────────────────────
info "Launching Streamlit dashboard..."
echo ""
echo "------------------------------------------------------"
echo "  Dashboard URL: http://localhost:8501"
echo "  Press Ctrl+C to stop."
echo "------------------------------------------------------"
echo ""
streamlit run app.py
