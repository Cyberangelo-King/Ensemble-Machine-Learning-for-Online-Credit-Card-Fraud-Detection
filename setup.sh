#!/usr/bin/env bash
# One-command setup for the Streamlit fraud detection dashboard demo.
set -euo pipefail

APP_PORT="${APP_PORT:-8501}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"

printf '\n🚀 Fraud Detection Dashboard - One-command setup\n'
printf '================================================\n\n'

find_python() {
    if [[ -n "$PYTHON_BIN" ]]; then
        command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
            echo "❌ PYTHON_BIN is set to '$PYTHON_BIN' but it was not found."
            exit 1
        }
        echo "$PYTHON_BIN"
        return
    fi

    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(not ((3, 10) <= sys.version_info[:2] <= (3, 12)))
PY
            then
                echo "$candidate"
                return
            fi
        fi
    done

    echo "❌ Could not find Python 3.10, 3.11, or 3.12. Install one of those versions and rerun this script."
    exit 1
}

PYTHON_CMD="$(find_python)"
echo "✓ Using Python: $($PYTHON_CMD --version)"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "📦 Creating virtual environment in $VENV_DIR..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
else
    echo "✓ Reusing existing virtual environment: $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ ! -f artifacts/stacking_model.joblib || ! -f data/creditcard.csv ]]; then
    echo "🤖 Creating demo model and sample data..."
    python scripts/create_demo_model.py
else
    echo "✓ Demo/model artifacts already exist."
fi

cat <<EOF

✅ Setup complete.

Starting Streamlit at: http://localhost:${APP_PORT}
Press Ctrl+C to stop the dashboard.
EOF

streamlit run app.py --server.port "$APP_PORT"
