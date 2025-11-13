"""
Compatibility modules for GraphPlag.

This package contains patches and compatibility layers for third-party dependencies
that have changed their APIs or have known issues.
"""

# Apply GraKeL/SciPy compatibility patch on import
try:
    from .grakel_scipy_patch import apply_grakel_scipy_patch
    apply_grakel_scipy_patch()
except ImportError:
    pass

# Apply GraKeL kernel numerical stability patch
try:
    from .grakel_stability_patch import apply_grakel_kernel_stability_patch
    apply_grakel_kernel_stability_patch()
except ImportError:
    pass

__all__ = ['grakel_scipy_patch', 'grakel_stability_patch']
