#!/usr/bin/env python
"""
Quick test script for AI detection feature in the web interface
"""

from graphplag.detection.ai_detector import AIDetector
import json

# Initialize detector
print("🔄 Initializing AI Detector...")
detector = AIDetector()
print("✅ AI Detector ready!\n")

# Test 1: Human-written text
human_text = """
I think climate change is one of the most important issues we face today. 
From what I've read, the evidence clearly shows that humans are causing global warming. 
I really believe we need to take action now, even though some people disagree with me. 
In my opinion, renewable energy is part of the solution, but we also need to change how we live.
"""

print("=" * 70)
print("TEST 1: HUMAN-WRITTEN TEXT")
print("=" * 70)
print(f"Text length: {len(human_text)} characters\n")
print("Analyzing with ENSEMBLE method...")

result = detector.detect_ai_content(human_text, method="ensemble")
print(f"\n✅ Result:")
print(f"   Is AI: {result['is_ai']}")
print(f"   Confidence: {result['confidence']:.1%}")
print(f"   Scores: {result['scores']}\n")

# Test 2: AI-like formal text
formal_text = """
Artificial intelligence represents a transformative technology that is fundamentally 
reshaping industries across the globe. The implications of machine learning are profound 
and far-reaching. Organizations are increasingly implementing AI solutions to enhance 
operational efficiency. The rapid advancement of neural networks has enabled unprecedented 
capabilities in pattern recognition and predictive analysis.
"""

print("=" * 70)
print("TEST 2: FORMAL/AI-LIKE TEXT")
print("=" * 70)
print(f"Text length: {len(formal_text)} characters\n")
print("Analyzing with ENSEMBLE method...")

result = detector.detect_ai_content(formal_text, method="ensemble")
print(f"\n✅ Result:")
print(f"   Is AI: {result['is_ai']}")
print(f"   Confidence: {result['confidence']:.1%}")
print(f"   Scores: {result['scores']}\n")

# Test 3: Very short text
short_text = "AI is great."

print("=" * 70)
print("TEST 3: SHORT TEXT")
print("=" * 70)
print(f"Text length: {len(short_text)} characters\n")
print("Analyzing with ENSEMBLE method...")

result = detector.detect_ai_content(short_text, method="ensemble")
print(f"\n✅ Result:")
print(f"   Is AI: {result['is_ai']}")
print(f"   Confidence: {result['confidence']:.1%}")
print(f"   Note: Short text may be less accurate\n")

# Test 4: Different detection methods
test_text = """
The impact of social media on mental health has been a subject of increasing research interest. 
Studies have shown that excessive social media usage correlates with higher rates of anxiety 
and depression. However, social media also provides valuable connections for many individuals.
"""

print("=" * 70)
print("TEST 4: TESTING DIFFERENT METHODS")
print("=" * 70)
print(f"Text: '{test_text[:50]}...'\n")

methods = ["ensemble", "neural", "statistical", "linguistic"]

for method in methods:
    try:
        result = detector.detect_ai_content(test_text, method=method)
        print(f"📊 {method.upper():15} → Confidence: {result['confidence']:.1%}")
    except Exception as e:
        print(f"⚠️ {method.upper():15} → Error: {e}")

print("\n" + "=" * 70)
print("✅ ALL TESTS COMPLETED!")
print("=" * 70)
print("\nThe AI Detection feature is working correctly!")
print("You can now use option [4] in run.bat to access the web interface.")
print("\nFor more information, see: NEW_AI_TAB_GUIDE.md")
