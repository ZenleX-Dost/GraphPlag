"""
Quick test script for GraphPlag interfaces
"""

print("="*60)
print("🧪 Testing GraphPlag Interfaces")
print("="*60)

# Test 1: Import check
print("\n1️⃣ Testing imports...")
try:
    from graphplag import PlagiarismDetector
    print("   ✅ PlagiarismDetector imported successfully")
except Exception as e:
    print(f"   ❌ Import error: {e}")
    exit(1)

# Test 2: Basic functionality
print("\n2️⃣ Testing basic detection...")
try:
    detector = PlagiarismDetector(method='kernel', threshold=0.7)
    
    doc1 = "Machine learning is a subset of artificial intelligence that enables systems to learn."
    doc2 = "Machine learning, part of AI, allows systems to learn automatically."
    
    report = detector.detect_plagiarism(doc1, doc2)
    
    print(f"   ✅ Detection completed")
    print(f"   📊 Similarity: {report.similarity_score:.2%}")
    print(f"   ⚠️  Plagiarism: {'YES' if report.is_plagiarism else 'NO'}")
    
except Exception as e:
    print(f"   ❌ Detection error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check interface files
print("\n3️⃣ Checking interface files...")
import os

files_to_check = [
    ('app.py', 'Web Interface'),
    ('cli.py', 'CLI Interface'),
]

for filename, description in files_to_check:
    if os.path.exists(filename):
        print(f"   ✅ {description} ({filename}) exists")
    else:
        print(f"   ❌ {description} ({filename}) not found")

# Test 4: Check Gradio
print("\n4️⃣ Testing Gradio installation...")
try:
    import gradio as gr
    print(f"   ✅ Gradio {gr.__version__} installed")
except Exception as e:
    print(f"   ❌ Gradio import error: {e}")

print("\n" + "="*60)
print("✅ All tests completed!")
print("\n📝 Next steps:")
print("   1. Launch web interface: python app.py")
print("   2. Use CLI: python cli.py compare --help")
print("   3. Check QUICKSTART.md for detailed usage")
print("="*60)
