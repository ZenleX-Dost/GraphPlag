@echo off
REM GraphPlag Enhanced Web Interface Launcher
REM Windows Batch Script

echo.
echo ========================================
echo  GraphPlag Enhanced Interface
echo ========================================
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Verify activation
echo.
python -c "import sys; print('Python:', sys.executable)"

REM Launch the enhanced app
echo.
echo Starting enhanced web interface...
echo Open your browser to: http://localhost:7860
echo Press Ctrl+C to stop the server
echo.

python app_enhanced.py

pause
