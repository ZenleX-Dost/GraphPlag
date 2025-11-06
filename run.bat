@echo off
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║              GraphPlag - Complete Launcher                 ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Check if first run
if not exist "venv\Scripts\activate.bat" (
    echo ⚠ Virtual environment not found!
    echo.
    echo This appears to be your first time running GraphPlag.
    echo Would you like to set up the environment now?
    echo.
    choice /c YN /n /m "Setup now? (Y/N): "
    if errorlevel 2 goto :no_setup
    if errorlevel 1 goto :setup
)

REM Activate virtual environment
call venv\Scripts\activate.bat 2>nul
if errorlevel 1 (
    echo ⚠ Failed to activate virtual environment
    echo Please run setup first
    pause
    exit /b 1
)

:menu
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║              GraphPlag - Main Menu                         ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo   SETUP & MAINTENANCE
echo  [1] Setup GraphPlag (First time installation)
echo  [2] Fix NumPy Compatibility Issues
echo.
echo   RUN APPLICATION
echo  [3] Start Web Interface (Gradio)  Recommended
echo  [4] Start Enhanced Web Interface
echo  [5] Start REST API Server
echo.
echo   ADVANCED OPTIONS
echo  [6] Run CLI (Command line comparison)
echo  [7] Run Tests
echo  [8] Open API Documentation
echo.
echo   EXIT
echo  [9] Exit
echo.
choice /c 123456789 /n /m "Enter your choice (1-9): "

if errorlevel 9 goto :end
if errorlevel 8 goto :api_docs
if errorlevel 7 goto :tests
if errorlevel 6 goto :cli
if errorlevel 5 goto :api
if errorlevel 4 goto :web_enhanced
if errorlevel 3 goto :web
if errorlevel 2 goto :fix_numpy
if errorlevel 1 goto :setup

:setup
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║          GraphPlag - Setup & Installation                  ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo This will install ALL features including:
echo  ✓ Core plagiarism detection
echo  ✓ Graph kernels (GraKeL)
echo  ✓ NLP tools (spaCy, Stanza, Transformers)
echo  ✓ Web interface (Gradio)
echo  ✓ REST API (FastAPI)
echo  ✓ Visualization tools
echo  ✓ PDF/Excel report generation
echo.
echo This may take 5-10 minutes...
echo.
pause

echo.
echo [1/5] Checking Python version...
python --version
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.10+
    pause
    goto :menu
)

echo.
echo [2/5] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        goto :menu
    )
)

echo.
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [4/5] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [5/5] Installing NumPy 1.x (GraKeL compatibility)...
pip install "numpy<2.0.0"

echo.
echo [6/6] Installing dependencies...
echo This will take several minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo ⚠ Some packages failed to install
    echo.
)

echo.
echo Installing API dependencies...
pip install -r requirements-api.txt
if errorlevel 1 (
    echo ⚠ Some API packages failed to install
    echo.
)

echo.
echo ═══════════════════════════════════════════════════════════
echo Verifying Installation...
echo ═══════════════════════════════════════════════════════════
python -c "import numpy; print(f'✓ NumPy: {numpy.__version__}')" 2>nul || echo  NumPy failed
python -c "import scipy; print('✓ SciPy: OK')" 2>nul || echo  SciPy failed
python -c "import torch; print('✓ PyTorch: OK')" 2>nul || echo  PyTorch failed
python -c "import spacy; print('✓ spaCy: OK')" 2>nul || echo  spaCy failed
python -c "import grakel; print('✓ GraKeL: OK')" 2>nul || echo  GraKeL failed
python -c "import gradio; print('✓ Gradio: OK')" 2>nul || echo  Gradio failed
python -c "import fastapi; print('✓ FastAPI: OK')" 2>nul || echo  FastAPI failed
python -c "from graphplag.detection.detector import PlagiarismDetector; print('✓ GraphPlag Core: OK')" 2>nul || echo ❌ GraphPlag Core failed

echo.
echo ═══════════════════════════════════════════════════════════
echo   Setup Complete!
echo ═══════════════════════════════════════════════════════════
echo.
pause
goto :menu

:fix_numpy
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║          Fix NumPy Compatibility                           ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo Current NumPy version:
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>nul || echo NumPy not installed
echo.
echo This will fix NumPy 2.x compatibility issues with GraKeL
echo by downgrading to NumPy 1.x
echo.
pause

echo.
echo [1/3] Uninstalling NumPy 2.x...
pip uninstall -y numpy 2>nul

echo.
echo [2/3] Installing NumPy 1.x...
pip install "numpy<2.0.0"

echo.
echo [3/3] Reinstalling dependencies...
pip install -r requirements.txt --force-reinstall --no-deps

echo.
echo Verifying fix...
python -c "import numpy; print(f'✓ NumPy version: {numpy.__version__}')"
python -c "import grakel; print('✓ GraKeL: OK')" 2>nul || echo ❌ GraKeL failed

echo.
echo ═══════════════════════════════════════════════════════════
echo   NumPy Fix Complete!
echo ═══════════════════════════════════════════════════════════
echo.
pause
goto :menu

:web
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║          Starting Web Interface (Gradio)                   ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo The web interface will open automatically in your browser
echo.
echo  URL: http://localhost:7860
echo.
echo Features:
echo  ✓ Upload documents (PDF, DOCX, TXT)
echo  ✓ Compare text directly
echo  ✓ Visual similarity graphs
echo  ✓ Download PDF/Excel reports
echo.
echo Press Ctrl+C to stop the server
echo.
python app.py
pause
goto :menu

:web_enhanced
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║       Starting Enhanced Web Interface (Gradio)             ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo The enhanced web interface will open automatically
echo.
echo  URL: http://localhost:7860
echo.
echo Enhanced Features:
echo  ✓ Beautiful custom design
echo  ✓ Advanced visualizations
echo  ✓ Detailed match analysis
echo  ✓ Professional reports
echo.
echo Press Ctrl+C to stop the server
echo.
python app_enhanced.py
pause
goto :menu

:api
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║          Starting REST API Server (FastAPI)                ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo The API server will start at:
echo  • Swagger UI:  http://localhost:8000/docs
echo  • ReDoc:       http://localhost:8000/redoc
echo  • API:         http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
pause
goto :menu

:cli
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║          CLI - Command Line Comparison                     ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
set /p file1="Enter path to first document: "
set /p file2="Enter path to second document: "
echo.
echo Output format:
echo  [1] Console output only
echo  [2] Generate PDF report
echo  [3] Generate Excel report
echo.
choice /c 123 /n /m "Choose format (1-3): "

if errorlevel 3 goto :cli_excel
if errorlevel 2 goto :cli_pdf
if errorlevel 1 goto :cli_console

:cli_console
echo.
echo Running comparison...
python cli.py compare --file1 "%file1%" --file2 "%file2%"
echo.
pause
goto :menu

:cli_pdf
set /p output="Enter PDF filename (e.g., report.pdf): "
echo.
echo Running comparison and generating PDF...
python cli.py compare --file1 "%file1%" --file2 "%file2%" --output "%output%"
echo.
pause
goto :menu

:cli_excel
set /p output="Enter Excel filename (e.g., report.xlsx): "
echo.
echo Running comparison and generating Excel...
python cli.py compare --file1 "%file1%" --file2 "%file2%" --output "%output%"
echo.
pause
goto :menu

:tests
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║          Running Comprehensive Tests                       ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
python test_comprehensive.py
echo.
pause
goto :menu

:api_docs
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║          Opening API Documentation                         ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo Starting API server in background...
start /B python -m uvicorn api:app --port 8000 >nul 2>&1
echo Waiting for server to start...
timeout /t 5 /nobreak >nul
echo Opening browser...
start http://localhost:8000/docs
echo.
echo ✓ API Documentation opened in browser
echo.
echo The API server is running in the background.
echo You can close this window when done.
echo.
pause
goto :menu

:no_setup
echo.
echo Please run setup before using GraphPlag
echo.
pause
exit /b 1

:end
cls
echo ╔════════════════════════════════════════════════════════════╗
echo ║          Thank you for using GraphPlag!                  ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
exit /b 0
