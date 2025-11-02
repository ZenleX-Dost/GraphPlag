@echo off
REM GraphPlag Web Interface Launcher
echo.
echo ========================================
echo  Starting GraphPlag Web Interface
echo ========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if activation worked
python -c "import sys; print('Python:', sys.executable)"

REM Launch the app
echo.
echo Starting Gradio server...
echo Open your browser to: http://localhost:7860
echo.
python app.py

pause
