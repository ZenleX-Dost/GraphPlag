# NumPy Compatibility Fix for GraKeL

## Problem

GraKeL (Graph Kernel Library) was compiled against NumPy 1.x and is not compatible with NumPy 2.x. This causes the following error during deployment:

```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash.
```

## Solution

Pin NumPy to version 1.x across all dependency files and build configurations.

## Files Modified

### 1. requirements.txt
```diff
- numpy>=1.21.0
+ numpy>=1.21.0,<2.0.0  # Pin to 1.x for GraKeL compatibility
```

### 2. Dockerfile
```diff
# Copy requirements
COPY requirements.txt .

+ # Install NumPy 1.x first (GraKeL compatibility)
+ RUN pip install --no-cache-dir "numpy<2.0.0"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
```

### 3. railway.json
```diff
"buildCommand": "pip install --upgrade pip && 
+                pip install 'numpy<2.0.0' && 
                 pip install -r requirements.txt && 
                 pip install -r requirements-api.txt"
```

### 4. railway.toml
```diff
buildCommand = "pip install --upgrade pip && 
+               pip install 'numpy<2.0.0' && 
                pip install -r requirements.txt && 
                pip install -r requirements-api.txt"
```

### 5. nixpacks.toml
```diff
[phases.install]
cmds = [
  "pip install --upgrade pip",
+ "pip install 'numpy<2.0.0'",
  "pip install -r requirements.txt",
  "pip install -r requirements-api.txt"
]
```

## Testing Locally

To ensure the fix works locally:

```bash
# Create fresh virtual environment
python -m venv venv_test
.\venv_test\Scripts\activate  # Windows
# source venv_test/bin/activate  # Linux/Mac

# Install with NumPy constraint
pip install "numpy<2.0.0"
pip install -r requirements.txt
pip install -r requirements-api.txt

# Verify NumPy version
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
# Should show 1.x.x (e.g., 1.26.4)

# Verify GraKeL works
python -c "import grakel; print('GraKeL loaded successfully')"

# Run tests
python test_comprehensive.py
```

## Docker Build Test

```bash
# Build Docker image
docker build -t graphplag-test .

# Run container
docker run -p 8000:8000 graphplag-test

# Test endpoint
curl http://localhost:8000/health
```

## Railway Deployment

With these changes, Railway will:
1. Install NumPy 1.x first
2. Install other dependencies (which will respect NumPy 1.x)
3. Deploy successfully without NumPy compatibility errors

## Why This Works

Installing NumPy 1.x **before** other packages ensures:
- Other packages don't pull NumPy 2.x as a dependency
- pip's dependency resolver respects the already-installed NumPy 1.x
- GraKeL can import without compatibility issues

## Alternative Solutions (Not Recommended)

1. **Wait for GraKeL update**: GraKeL needs to be recompiled for NumPy 2.x
2. **Use different graph kernel library**: Would require major code changes
3. **Downgrade all packages**: Would lose features and security updates

## Future Updates

Monitor GraKeL repository for NumPy 2.x compatibility:
- GitHub: https://github.com/ysig/GraKeL
- Once GraKeL supports NumPy 2.x, remove the `<2.0.0` constraint

## Related Issues

- GraKeL Issue: https://github.com/ysig/GraKeL/issues (check for NumPy 2.x support)
- NumPy 2.0 Migration Guide: https://numpy.org/devdocs/numpy_2_0_migration_guide.html
