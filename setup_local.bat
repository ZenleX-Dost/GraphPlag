@echo off
echo ========================================
echo GraphPlag - Complete Setup
echo ========================================
echo.
echo This will set up GraphPlag with ALL features:
echo - Core plagiarism detection
echo - Graph kernels (GraKeL)
echo - NLP tools (spaCy, Stanza, Transformers)
echo - REST API (FastAPI)
echo - Visualization (Matplotlib, Plotly)
echo - Report generation (PDF, Excel)
echo - Development tools
echo.
pause

echo.
echo [Step 1/5] Checking Python version...
python --version
echo.

echo [Step 2/5] Upgrading pip...
python -m pip install --upgrade pip
echo.

echo [Step 3/5] Installing NumPy 1.x (GraKeL compatibility)...
pip install "numpy<2.0.0"
echo.

echo [Step 4/5] Installing all core dependencies...
pip install -r requirements.txt
echo.

echo [Step 5/5] Installing API dependencies...
pip install -r requirements-api.txt
echo.

echo ========================================
echo Verifying Installation
echo ========================================
echo.
python -c "import numpy; print(f'✓ NumPy: {numpy.__version__}')"
python -c "import scipy; print('✓ SciPy: OK')"
python -c "import torch; print('✓ PyTorch: OK')"
python -c "import spacy; print('✓ spaCy: OK')"
python -c "import grakel; print('✓ GraKeL: OK')"
python -c "import fastapi; print('✓ FastAPI: OK')"
python -c "import networkx; print('✓ NetworkX: OK')"
python -c "from sentence_transformers import SentenceTransformer; print('✓ Sentence Transformers: OK')"
python -c "from graphplag.detection.detector import PlagiarismDetector; print('✓ GraphPlag Core: OK')"
echo.

echo ========================================
echo Installation Complete! ✓
echo ========================================
echo.
echo You can now use GraphPlag:
echo.
echo   CLI:  python cli.py compare --file1 doc1.txt --file2 doc2.txt
echo   API:  python -m uvicorn api:app --reload
echo   Test: python test_comprehensive.py
echo.
pause
