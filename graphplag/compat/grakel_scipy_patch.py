"""
GraKeL/SciPy Compatibility Patch

This module patches GraKeL's random walk kernel to work with SciPy >= 1.10.0
where the cg() function signature changed from 'tol' to 'rtol'/'atol' parameters.

The patch is automatically applied on import.
"""

import sys
import warnings


def apply_grakel_scipy_patch():
    """
    Apply compatibility patch for GraKeL with modern SciPy versions.
    
    In SciPy 1.10.0+, the scipy.sparse.linalg.cg() function signature changed:
    - Old: cg(A, b, tol=1.0e-6, maxiter=20, atol='legacy')
    - New: cg(A, b, rtol=1e-05, atol=0.0, maxiter=None)
    
    This patch wraps the cg() function to convert old-style calls to new-style.
    """
    
    try:
        from scipy.sparse.linalg import cg as scipy_cg
        from grakel.kernels import random_walk
        
        # Store the original cg function
        _original_cg = scipy_cg
        
        def cg_wrapper(A, b, x0=None, *, rtol=None, atol=None, maxiter=None, M=None, callback=None, tol=None):
            """
            Wrapper for scipy.sparse.linalg.cg that handles both old and new signatures.
            
            Parameters:
            -----------
            tol : float, optional (deprecated)
                Tolerance for convergence (old SciPy parameter)
                Maps to rtol in new SciPy versions
            
            rtol : float, optional
                Relative tolerance for convergence (new SciPy parameter)
                
            atol : float or 'legacy', optional
                Absolute tolerance for convergence
                New SciPy parameter (>= 1.10.0)
                'legacy' is converted to 0.0 for compatibility
            
            maxiter : int, optional
                Maximum number of iterations
            
            Other parameters match scipy.sparse.linalg.cg
            """
            
            # Handle old 'tol' parameter
            if tol is not None and rtol is None:
                rtol = tol
            
            # Set defaults if not specified
            if rtol is None:
                rtol = 1e-6  # Default from old GraKeL call
            
            # Handle 'legacy' atol value from old GraKeL code
            if atol == 'legacy':
                atol = 0.0  # Use default atol value
            elif atol is None:
                atol = 0.0  # SciPy default
            
            if maxiter is None:
                maxiter = 20  # Default from old GraKeL call
            
            # Call the new scipy.sparse.linalg.cg with proper parameters
            return _original_cg(
                A, b,
                x0=x0,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
                M=M,
                callback=callback
            )
        
        # Monkey-patch the grakel random_walk module
        import grakel.kernels.random_walk as rw_module
        rw_module.cg = cg_wrapper
        
        # Also patch it in scipy if it's directly imported there
        import scipy.sparse.linalg
        
        return True
        
    except Exception as e:
        warnings.warn(f"Failed to apply GraKeL/SciPy compatibility patch: {e}")
        return False


# Apply patch on module import
if 'grakel' in sys.modules or 'scipy' in sys.modules:
    apply_grakel_scipy_patch()


__all__ = ['apply_grakel_scipy_patch']
