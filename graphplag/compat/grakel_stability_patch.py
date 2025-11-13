"""
GraKeL Kernel Numerical Stability Patch

This module patches GraKeL's kernel computation to handle numerical instability
issues that result in NaN values when computing kernels for certain graph types.

The patch is automatically applied on import.
"""

import warnings
import numpy as np


def apply_grakel_kernel_stability_patch():
    """
    Apply numerical stability patch for GraKeL kernel computation.
    
    GraKeL's shortest_path kernel can produce NaN values due to division by zero
    when normalizing the kernel matrix. This patch wraps fit_transform to
    replace NaN values with sensible defaults.
    """
    
    try:
        from grakel.kernels.kernel import Kernel
        
        # Store the original fit_transform method
        _original_fit_transform = Kernel.fit_transform
        
        def fit_transform_wrapper(self, X, y=None):
            """
            Wrapper for fit_transform that handles NaN values in kernel matrix.
            """
            result = _original_fit_transform(self, X, y)
            
            # If result is array, replace any NaN values with 0
            if isinstance(result, np.ndarray):
                # Replace NaN with 0 (no similarity), Inf with 1 (perfect similarity)
                result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
            
            return result
        
        # Monkey-patch the method
        Kernel.fit_transform = fit_transform_wrapper
        
        return True
        
    except Exception as e:
        warnings.warn(f"Failed to apply GraKeL kernel stability patch: {e}")
        return False


# Apply patch on module import
apply_grakel_kernel_stability_patch()

__all__ = ['apply_grakel_kernel_stability_patch']
