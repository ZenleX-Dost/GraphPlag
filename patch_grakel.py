"""
Patch GraKeL library to fix compatibility issues with scipy 1.15+
and resolve NaN issues in kernels.

This script modifies the installed GraKeL library files to:
1. Fix RandomWalk kernel compatibility with scipy 1.15+ (change tol to rtol)
2. Add safety checks for division by zero in normalization

Run this script after installing GraKeL.
"""
import os
import sys
import re


def patch_random_walk_kernel(grakel_path):
    """
    Patch the RandomWalk kernel to be compatible with scipy 1.15+.
    Changes 'tol=' to 'rtol=' in cg() function calls.
    """
    random_walk_file = os.path.join(grakel_path, 'kernels', 'random_walk.py')
    
    if not os.path.exists(random_walk_file):
        print(f"ERROR: Could not find {random_walk_file}")
        return False
    
    print(f"Patching RandomWalk kernel at: {random_walk_file}")
    
    # Read the file
    with open(random_walk_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count occurrences before patching
    tol_count = content.count('tol=1.0e-6')
    print(f"  Found {tol_count} occurrences of 'tol=1.0e-6'")
    
    # Replace tol= with rtol= in cg() calls
    # Pattern: cg(A, b, tol=1.0e-6, maxiter=20, atol='legacy')
    # Replace with: cg(A, b, rtol=1.0e-6, maxiter=20, atol=1.0e-8)
    original_pattern = r"cg\(A,\s*b,\s*tol=1\.0e-6,\s*maxiter=20,\s*atol='legacy'\)"
    replacement = "cg(A, b, rtol=1.0e-6, maxiter=20, atol=1.0e-8)"
    
    new_content = re.sub(original_pattern, replacement, content)
    
    if new_content != content:
        # Create backup
        backup_file = random_walk_file + '.backup'
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Created backup at: {backup_file}")
        
        # Write patched content
        with open(random_walk_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"  ✓ Successfully patched RandomWalk kernel (replaced {tol_count} occurrences)")
        return True
    else:
        print("  No changes needed - file may already be patched")
        return True


def patch_shortest_path_kernel(grakel_path):
    """
    Patch ShortestPath kernel to handle division by zero in its own normalization.
    """
    sp_file = os.path.join(grakel_path, 'kernels', 'shortest_path.py')
    
    if not os.path.exists(sp_file):
        print(f"ERROR: Could not find {sp_file}")
        return False
    
    print(f"Patching ShortestPath kernel at: {sp_file}")
    
    # Read the file
    with open(sp_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if 'PATCHED: Added zero-check' in content:
        print("  File already patched - skipping")
        return True
    
    # Pattern: return np.divide(km, np.sqrt(np.outer(self._X_diag, self._X_diag)))
    pattern = r'(\s+)(return np\.divide\(km, np\.sqrt\(np\.outer\(self\._X_diag, self\._X_diag\)\)\))'
    replacement = r'''\1# PATCHED: Added zero-check for normalization to prevent NaN
\1normalizer = np.sqrt(np.outer(self._X_diag, self._X_diag))
\1normalizer = np.where(normalizer == 0, 1, normalizer)
\1return np.divide(km, normalizer)'''
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:
        # Create backup
        backup_file = sp_file + '.backup'
        if not os.path.exists(backup_file):
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Created backup at: {backup_file}")
        
        # Write patched content
        with open(sp_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"  ✓ Successfully patched ShortestPath kernel normalization")
        return True
    else:
        print("  No normalization pattern found")
        return True


def patch_kernel_base(grakel_path):
    """
    Patch the base Kernel class to handle division by zero in normalization.
    This prevents NaN values when normalizing kernel matrices.
    """
    kernel_file = os.path.join(grakel_path, 'kernels', 'kernel.py')
    
    if not os.path.exists(kernel_file):
        print(f"ERROR: Could not find {kernel_file}")
        return False
    
    print(f"Patching base Kernel class at: {kernel_file}")
    
    # Read the file
    with open(kernel_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if 'PATCHED: Added zero-check for normalization' in content:
        print("  File already patched - skipping")
        return True
    
    patches_applied = 0
    
    # Patch 1: transform() method - line ~169
    # Pattern: km /= np.sqrt(np.outer(Y_diag, X_diag))
    pattern1 = r'(\s+)(km /= np\.sqrt\(np\.outer\(Y_diag, X_diag\)\))'
    replacement1 = r'''\1# PATCHED: Added zero-check for normalization to prevent NaN
\1normalizer = np.sqrt(np.outer(Y_diag, X_diag))
\1normalizer = np.where(normalizer == 0, 1, normalizer)
\1km /= normalizer'''
    
    new_content = re.sub(pattern1, replacement1, content)
    if new_content != content:
        patches_applied += 1
        content = new_content
    
    # Patch 2: fit_transform() method - line ~202
    # Pattern: return km / np.sqrt(np.outer(self._X_diag, self._X_diag))
    pattern2 = r'(\s+)(return km / np\.sqrt\(np\.outer\(self\._X_diag, self\._X_diag\)\))'
    replacement2 = r'''\1# PATCHED: Added zero-check for normalization to prevent NaN
\1normalizer = np.sqrt(np.outer(self._X_diag, self._X_diag))
\1normalizer = np.where(normalizer == 0, 1, normalizer)
\1return km / normalizer'''
    
    new_content = re.sub(pattern2, replacement2, content)
    if new_content != content:
        patches_applied += 1
        content = new_content
    
    if patches_applied > 0:
        # Create backup
        backup_file = kernel_file + '.backup'
        if not os.path.exists(backup_file):  # Don't overwrite existing backup
            with open(kernel_file, 'r', encoding='utf-8') as f:
                original = f.read()
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(original)
            print(f"  Created backup at: {backup_file}")
        
        # Write patched content
        with open(kernel_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✓ Successfully patched base Kernel class ({patches_applied} locations)")
        return True
    else:
        print("  No normalization patterns found - file may already be patched")
        return True


def main():
    """Main patching function."""
    print("=" * 70)
    print("GraKeL Patching Tool")
    print("=" * 70)
    print()
    
    # Try to find GraKeL installation
    try:
        import grakel
        grakel_path = os.path.dirname(grakel.__file__)
        print(f"Found GraKeL at: {grakel_path}")
        print()
    except ImportError:
        print("ERROR: GraKeL is not installed")
        print("Please install GraKeL first: pip install grakel")
        return 1
    
    # Apply patches
    success = True
    
    # Patch 1: RandomWalk kernel scipy compatibility
    print("Patch 1: RandomWalk Kernel (scipy 1.15+ compatibility)")
    print("-" * 70)
    if not patch_random_walk_kernel(grakel_path):
        success = False
    print()
    
    # Patch 2: ShortestPath kernel normalization
    print("Patch 2: ShortestPath Kernel Normalization (NaN prevention)")
    print("-" * 70)
    if not patch_shortest_path_kernel(grakel_path):
        success = False
    print()
    
    # Patch 3: Base kernel normalization
    print("Patch 3: Base Kernel Normalization (NaN prevention)")
    print("-" * 70)
    if not patch_kernel_base(grakel_path):
        success = False
    print()
    
    # Summary
    print("=" * 70)
    if success:
        print("✓ All patches applied successfully!")
        print()
        print("Next steps:")
        print("1. Test the patched kernels: python test_kernel_fixes.py")
        print("2. All kernels should now work correctly in the plagiarism detector")
        print("3. Run CLI tests to verify end-to-end functionality")
    else:
        print("⚠ Some patches failed - please check the errors above")
        print("You may need to manually edit the GraKeL files")
    print("=" * 70)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
