@echo off
REM GraphPlag Gradio Frontend
cd /d "%~dp0"

echo ================================================
echo   GraphPlag - Gradio Interactive Interface
echo ================================================
echo.

REM Activate virtual environment
echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if gradio is installed
echo [2/3] Checking dependencies...
python -c "import gradio" 2>nul
if errorlevel 1 (
    echo Installing Gradio and dependencies...
    pip install -q gradio plotly pandas numpy
)

REM Launch Gradio app
echo [3/3] Starting Gradio interface...
echo.
echo ================================================
echo   Access the interface at:
echo   http://localhost:7860
echo ================================================
echo.

python app.py

pause
