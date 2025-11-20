#!/usr/bin/env python3
"""Debug AI detection accuracy"""

from src.ai_detection.ai_detector import AIDetector

# Test cases
test_cases = {
    "Obviously AI (formal)": """Artificial intelligence represents a transformative technology that continues to reshape industries and societies worldwide. The exponential growth in computational power, coupled with advances in machine learning algorithms, has enabled unprecedented capabilities in data processing and pattern recognition. Contemporary applications span diverse domains including natural language processing, computer vision, and autonomous systems. As organizations increasingly integrate AI solutions into their operations, the importance of robust governance frameworks and ethical considerations cannot be overstated.""",
    
    "Human (casual)": """I think AI is pretty cool. Like, it does a lot of stuff that used to take people forever. You know? I was reading about how it can write essays now. That's wild. But also kind of scary? Anyway, I think it's gonna change a lot of things.""",
    
    "AI ChatGPT style": """The rapid advancement of artificial intelligence technology has fundamentally altered our understanding of computational possibilities. Through sophisticated algorithms and extensive training datasets, modern AI systems demonstrate remarkable proficiency in tasks previously thought to require human cognition. This paradigm shift necessitates careful consideration of both opportunities and potential societal implications.""",
}

detector = AIDetector()

print("=" * 70)
print("AI DETECTION ACCURACY TEST")
print("=" * 70)

for name, text in test_cases.items():
    print(f"\n{name}:")
    print(f"Text: {text[:80]}...")
    
    result = detector.detect_ai_content(text, method='ensemble')
    print(f"Is AI: {result['is_ai']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Scores: {result['scores']}")
    print("-" * 70)
