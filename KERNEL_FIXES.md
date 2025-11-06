# Kernel Fixes Summary

## Overview
Successfully fixed the RandomWalk and ShortestPath graph kernels that were previously disabled due to compatibility issues with scipy 1.15+ and NaN value problems.

## Issues Resolved

### 1. RandomWalk Kernel - scipy 1.15+ Compatibility ✓
**Problem:**
- RandomWalk kernel was failing with `TypeError: cg() got an unexpected keyword argument 'tol'`
- scipy 1.15+ changed the API for `scipy.sparse.linalg.cg()` function
- Old API: `cg(A, b, tol=1.0e-6, maxiter=20, atol='legacy')`
- New API: `cg(A, b, rtol=1.0e-6, maxiter=20, atol=1.0e-8)`

**Solution:**
- Patched `venv\lib\site-packages\grakel\kernels\random_walk.py`
- Replaced `tol=` parameter with `rtol=` (relative tolerance)
- Changed `atol='legacy'` to `atol=1.0e-8` (numeric value)
- Applied patch to 2 locations in the file (lines 271 and 470)

### 2. ShortestPath Kernel - NaN Values ✓
**Problem:**
- ShortestPath kernel was producing NaN values during normalization
- Division by zero when normalizing kernel matrices with zero diagonal values

**Solution:**
- Patched `venv\lib\site-packages\grakel\kernels\kernel.py`
- Added zero-check before normalization in 2 locations:
  - `transform()` method (~line 169)
  - `fit_transform()` method (~line 202)
- Replace zero values with 1 before division to prevent NaN

## Files Modified

### 1. Created: `patch_grakel.py`
Automated patching tool that:
- Locates installed GraKeL library
- Applies RandomWalk scipy compatibility fix
- Applies ShortestPath NaN prevention fix
- Creates backups before modifying files
- Provides detailed status reporting

### 2. Updated: `graphplag\similarity\graph_kernels.py`
- Changed default `kernel_types` from `['wl', 'sp']` to `['wl', 'rw', 'sp']`
- Updated documentation to reflect that all kernels are now working
- Added comments explaining the fixes

### 3. Updated: `graphplag\detection\detector.py`
- Changed default kernel configuration from `['wl']` to `['wl', 'rw', 'sp']`
- Enables all three kernels in ensemble detection by default

### 4. Created: `test_kernel_fixes.py`
Comprehensive test suite that verifies:
- RandomWalk kernel compatibility with scipy 1.15+
- ShortestPath kernel NaN-free operation
- GraphKernelSimilarity ensemble functionality
- PlagiarismDetector with all three kernels

## Test Results

All tests passed successfully:

```
======================================================================
TEST SUMMARY
======================================================================
RandomWalk Kernel              ✓ PASSED
ShortestPath Kernel            ✓ PASSED
GraphKernelSimilarity          ✓ PASSED
PlagiarismDetector             ✓ PASSED
======================================================================

✓ All tests passed! RandomWalk and ShortestPath kernels are fixed.
```

### PlagiarismDetector Output:
```
Similarity: 1.0000
Is plagiarism: True
Method: kernel
Kernel scores: {'wl': 1.0, 'rw': 1.0, 'sp': 1.0}
```

All three kernels are now working correctly!

## How to Apply Fixes

### Option 1: Automated (Recommended)
```bash
cd "c:\Users\Amine EL-Hend\Documents\GitHub\GraphPlag"
.\venv\Scripts\python.exe patch_grakel.py
```

### Option 2: Manual
If you install GraphPlag on another system:
1. Install dependencies: `pip install -r requirements.txt`
2. Run patch script: `python patch_grakel.py`
3. Verify with tests: `python test_kernel_fixes.py`

## Technical Details

### RandomWalk Kernel Patch
**File:** `grakel/kernels/random_walk.py`

**Before:**
```python
x_sol, _ = cg(A, b, tol=1.0e-6, maxiter=20, atol='legacy')
```

**After:**
```python
x_sol, _ = cg(A, b, rtol=1.0e-6, maxiter=20, atol=1.0e-8)
```

### Normalization Patch
**File:** `grakel/kernels/kernel.py`

**Before:**
```python
km /= np.sqrt(np.outer(Y_diag, X_diag))
```

**After:**
```python
# PATCHED: Added zero-check for normalization to prevent NaN
normalizer = np.sqrt(np.outer(Y_diag, X_diag))
normalizer = np.where(normalizer == 0, 1, normalizer)
km /= normalizer
```

## Benefits

### Improved Detection Accuracy
- **Weisfeiler-Lehman (WL)**: Fast, effective for general graphs
- **Random Walk (RW)**: Captures graph structure through random walks
- **Shortest Path (SP)**: Considers path-based similarities

### Ensemble Method
All three kernels are now combined in the ensemble method:
```python
ensemble_score = (wl_score + rw_score + sp_score) / 3
```

This provides more robust plagiarism detection by leveraging multiple perspectives.

## Backward Compatibility

The changes are backward compatible:
- Can still use individual kernels: `kernel_types=['wl']`
- Default behavior now uses all three kernels
- Existing code continues to work without modifications

## Dependencies

- Python 3.10.11
- scipy 1.15.3 (new API)
- numpy 1.26.4 (compatible with GraKeL)
- grakel 0.1.10 (patched)
- networkx 3.4.2

## Verification

Run the test suite to verify fixes:
```bash
python test_kernel_fixes.py
```

Expected output:
- All 4 tests should pass
- No NaN values detected
- All kernel scores computed successfully

## Notes

1. **Backup Files**: The patch script creates `.backup` files before modifying GraKeL sources
2. **Re-applying**: Patch script is idempotent - safe to run multiple times
3. **Virtual Environment**: Patches are applied to the venv-specific GraKeL installation
4. **Future Updates**: If GraKeL is updated/reinstalled, re-run the patch script

## Status

✅ **FULLY RESOLVED**

Both RandomWalk and ShortestPath kernels are now working correctly with scipy 1.15+ and produce valid (non-NaN) similarity scores.
