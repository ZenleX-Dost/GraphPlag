#!/usr/bin/env pwsh
# GraphPlag Enhanced Web Interface Launcher

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " GraphPlag Enhanced Interface" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Verify activation
$pythonPath = python -c "import sys; print(sys.executable)"
Write-Host "Python: $pythonPath" -ForegroundColor Gray

# Launch the enhanced app
Write-Host ""
Write-Host "Starting enhanced web interface..." -ForegroundColor Green
Write-Host "Open your browser to: http://localhost:7860" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

python app_enhanced.py
