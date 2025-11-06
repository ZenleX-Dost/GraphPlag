@echo off
echo ========================================
echo GraphPlag - Full Installation (Local)
echo ========================================
echo.

echo Current NumPy version:
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>nul || echo NumPy not installed
echo.

echo This will install ALL dependencies with NumPy 1.x for GraKeL compatibility.
echo This includes: Core deps, NLP tools, API deps, visualization, dev tools
echo.
pause

echo.
echo [1/4] Uninstalling NumPy 2.x (if present)...
pip uninstall -y numpy 2>nul

echo.
echo [2/4] Installing NumPy 1.x...
pip install "numpy<2.0.0"

echo.
echo [3/4] Installing all core dependencies...
pip install -r requirements.txt

echo.
echo [4/4] Installing API dependencies...
pip install -r requirements-api.txt

echo.
echo ========================================
echo Verifying installation...
echo ========================================
python -c "import numpy; print(f'✓ NumPy version: {numpy.__version__}')"
python -c "import grakel; print('✓ GraKeL: OK')"
python -c "import spacy; print('✓ spaCy: OK')"
python -c "import fastapi; print('✓ FastAPI: OK')"
python -c "from graphplag.detection.detector import PlagiarismDetector; print('✓ GraphPlag: OK')"

echo.
echo ========================================
echo Fix complete!
echo ========================================
pause
