#!/usr/bin/env bash
set -e

echo "========================================"
echo " Starting RAGex Intelligence Assistant   "
echo "========================================"

# 1. Virtual environment check
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# 2. Dependencies
echo "[2/4] Verifying dependencies..."
pip install -q -r requirements.txt

# 3. NLTK corpora
echo "[3/4] Checking NLTK lexical databases..."
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# 4. Launch
echo "[4/4] Launching application on http://localhost:5000..."
python app.py
