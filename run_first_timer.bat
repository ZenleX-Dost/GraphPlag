@echo off
REM ============================================================
REM GraphPlag - First Time Setup and Run Script
REM ============================================================
REM This script sets up the environment and runs GraphPlag
REM for users who just cloned/downloaded the project.
REM ============================================================

echo ============================================================
echo           GraphPlag - First Time Setup
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/5] Checking Python version...
python --version
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [2/5] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo       Virtual environment created successfully.
) else (
    echo [2/5] Virtual environment already exists.
)
echo.

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo       Virtual environment activated.
echo.

REM Install dependencies
echo [4/5] Installing dependencies (this may take a few minutes)...
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    echo Try running: pip install -r requirements.txt
    pause
    exit /b 1
)
echo       Dependencies installed successfully.
echo.

REM Download spaCy model if not present
echo [5/5] Downloading language model...
python -m spacy download en_core_web_sm >nul 2>&1
echo       Language model ready.
echo.

echo ============================================================
echo           Setup Complete - Starting GraphPlag
echo ============================================================
echo.
echo Starting API server (backend) on http://localhost:8000
echo Starting Web UI (frontend) on http://localhost:7860
echo.
echo Press Ctrl+C to stop the servers.
echo ============================================================
echo.

REM Start the API server in the background
start "GraphPlag API" cmd /c "venv\Scripts\activate.bat && python -m uvicorn api:app --host 0.0.0.0 --port 8000"

REM Wait a moment for API to start
timeout /t 3 /nobreak >nul

REM Start the Gradio web interface (foreground)
python app.py

REM If we reach here, the user closed the app
echo.
echo GraphPlag has been stopped.
pause
