# RAGex One-Command Startup Script (Windows PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Starting RAGex Intelligence Assistant   " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Check or create virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "[1/4] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
}

# 2. Activate virtual environment
$pythonPath = ".\.venv\Scripts\python.exe"
$pipPath = ".\.venv\Scripts\pip.exe"

# 3. Check dependencies
Write-Host "[2/4] Verifying dependencies..." -ForegroundColor Yellow
& $pipPath install -q -r requirements.txt

# 4. Verify NLTK corpora
Write-Host "[3/4] Checking NLTK lexical databases..." -ForegroundColor Yellow
& $pythonPath -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# 5. Launch application
Write-Host "[4/4] Launching Flask server on http://localhost:5000..." -ForegroundColor Green
& $pythonPath app.py
