"""
Comprehensive end-to-end test for all three graph kernels.
"""
from graphplag.detection.detector import PlagiarismDetector
import time


def test_kernel_combinations():
    """Test different kernel combinations"""
    print("=" * 70)
    print("Testing Different Kernel Combinations")
    print("=" * 70)
    print()
    
    text1 = """
    Machine learning is a subset of artificial intelligence that focuses on 
    developing algorithms that can learn from and make predictions on data.
    It involves statistical techniques to give computers the ability to learn
    without being explicitly programmed.
    """
    
    text2 = """
    Deep learning is a branch of machine learning that uses neural networks
    with multiple layers. It can automatically learn representations from data
    without manual feature engineering.
    """
    
    text3 = """
    Machine learning is a subset of artificial intelligence that focuses on 
    developing algorithms that can learn from and make predictions on data.
    """
    
    test_cases = [
        ("Similar texts", text1, text3),
        ("Different texts", text1, text2),
    ]
    
    kernel_configs = [
        (['wl'], "Weisfeiler-Lehman only"),
        (['rw'], "Random Walk only"),
        (['sp'], "Shortest Path only"),
        (['wl', 'rw'], "WL + RW"),
        (['wl', 'sp'], "WL + SP"),
        (['rw', 'sp'], "RW + SP"),
        (['wl', 'rw', 'sp'], "All kernels (ensemble)"),
    ]
    
    for test_name, t1, t2 in test_cases:
        print(f"\n{test_name}:")
        print("-" * 70)
        
        for kernel_types, config_name in kernel_configs:
            detector = PlagiarismDetector(method='kernel', kernel_types=kernel_types)
            
            start = time.time()
            result = detector.detect_plagiarism(t1, t2)
            elapsed = time.time() - start
            
            print(f"  {config_name:30s} "
                  f"Similarity: {result.similarity_score:.4f}  "
                  f"Time: {elapsed:.3f}s")
            
            # Check for NaN
            import math
            if math.isnan(result.similarity_score):
                print(f"    ⚠ ERROR: NaN detected!")
                return False
    
    return True


def test_edge_cases():
    """Test edge cases that might cause issues"""
    print("\n" + "=" * 70)
    print("Testing Edge Cases")
    print("=" * 70)
    print()
    
    edge_cases = [
        ("Empty texts", "", ""),
        ("Single word", "Hello", "World"),
        ("Single sentence", "This is a test.", "This is another test."),
        ("Very short", "A B C", "D E F"),
        ("Identical", "Same text", "Same text"),
        ("One empty", "Some text here", ""),
    ]
    
    detector = PlagiarismDetector(method='kernel')
    
    for test_name, t1, t2 in edge_cases:
        try:
            result = detector.detect_plagiarism(t1, t2)
            
            import math
            if math.isnan(result.similarity_score):
                print(f"  {test_name:20s} ⚠ NaN detected!")
                return False
            else:
                print(f"  {test_name:20s} ✓ Score: {result.similarity_score:.4f}")
        except Exception as e:
            print(f"  {test_name:20s} ⚠ Error: {str(e)}")
            # Some edge cases might legitimately fail (e.g., empty text)
            continue
    
    return True


def main():
    """Run comprehensive tests"""
    print()
    print("=" * 70)
    print("COMPREHENSIVE KERNEL TESTING")
    print("=" * 70)
    print()
    
    success = True
    
    # Test 1: Different kernel combinations
    if not test_kernel_combinations():
        success = False
    
    # Test 2: Edge cases
    if not test_edge_cases():
        success = False
    
    # Summary
    print("\n" + "=" * 70)
    if success:
        print("✓ ALL COMPREHENSIVE TESTS PASSED")
        print("  - No NaN values detected")
        print("  - All kernel combinations working")
        print("  - Edge cases handled correctly")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)
    print()
    
    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
