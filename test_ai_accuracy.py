#!/usr/bin/env python3
"""Test AI detection with known AI-generated content"""

import sys
sys.path.insert(0, 'c:\\Users\\Amine EL-Hend\\Documents\\GitHub\\GraphPlag')

from graphplag.detection.ai_detector import AIDetector

# Real AI-generated text examples (ChatGPT, Claude, etc.)
test_cases = {
    "ChatGPT Example": """The intersection of artificial intelligence and human creativity represents one of the most compelling frontiers in contemporary technology. As machine learning algorithms continue to evolve in sophistication and capability, the question of whether AI systems can genuinely create original content has transitioned from theoretical speculation to practical reality. This paradigm shift necessitates a careful examination of the criteria by which we evaluate authenticity and originality in an increasingly digital world.""",
    
    "Formal AI Style": """Artificial intelligence systems have demonstrated remarkable proficiency in natural language generation tasks. The deployment of transformer-based architectures has fundamentally altered our understanding of computational linguistic capability. Contemporary models exhibit behavior patterns that closely approximate human-level performance across numerous benchmarks, thereby challenging conventional assumptions about the nature of human cognitive uniqueness.""",
    
    "ChatGPT Intro Pattern": """I appreciate your interest in this topic. As an AI language model, I can provide insights into the mechanisms underlying modern machine learning. The exponential growth in computational resources, coupled with innovations in neural network design, has enabled unprecedented advances in artificial intelligence applications. These developments carry profound implications for society at large.""",
}

detector = AIDetector()

print("=" * 80)
print("AI DETECTION TEST - Known AI-Generated Content")
print("=" * 80)

for name, text in test_cases.items():
    print(f"\n{'='*80}")
    print(f"Test Case: {name}")
    print(f"{'='*80}")
    print(f"Text Preview: {text[:100]}...")
    print()
    
    # Test ensemble (recommended)
    result = detector.detect_ai_content(text, method='ensemble')
    print(f"Ensemble Detection:")
    print(f"  Is AI: {result['is_ai']}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Individual Scores: {result['scores']}")
    print()
    
    # Test each method individually
    for method in ['statistical', 'linguistic', 'neural']:
        result = detector.detect_ai_content(text, method=method)
        print(f"{method.capitalize()} Detection:")
        print(f"  Is AI: {result['is_ai']}")
        print(f"  Confidence: {result['confidence']:.1%}")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print("If all tests show 'Is AI: False', the detector needs tuning.")
print("If most show confidence < 60%, the threshold is too high.")
