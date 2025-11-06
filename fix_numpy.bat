@echo off
echo ========================================
echo GraphPlag - NumPy Compatibility Fix
echo ========================================
echo.

echo Current NumPy version:
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
echo.

echo This will reinstall dependencies with NumPy 1.x for GraKeL compatibility.
echo.
pause

echo.
echo [1/3] Uninstalling NumPy 2.x...
pip uninstall -y numpy

echo.
echo [2/3] Installing NumPy 1.x...
pip install "numpy<2.0.0"

echo.
echo [3/3] Reinstalling dependencies...
pip install -r requirements.txt --force-reinstall --no-deps

echo.
echo Verifying installation...
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import grakel; print('GraKeL: OK')" 2>nul || echo GraKeL: Failed to import

echo.
echo ========================================
echo Fix complete!
echo ========================================
pause
