"""
Test File Parser with Sample Documents
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from graphplag.utils.file_parser import FileParser

print("="*60)
print("🧪 Testing File Parser")
print("="*60)

parser = FileParser()

# Test files
test_files = [
    "test_data/sample1.md",
    "test_data/sample2.txt",
]

print("\n📁 Parsing test files...\n")

texts = []
for file_path in test_files:
    if Path(file_path).exists():
        try:
            print(f"📄 Parsing: {file_path}")
            text = parser.parse_file(file_path)
            texts.append(text)
            print(f"   ✅ Extracted {len(text)} characters")
            print(f"   📝 Preview: {text[:100]}...")
            print()
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
    else:
        print(f"❌ File not found: {file_path}\n")

# Test plagiarism detection if we have texts
if len(texts) >= 2:
    print("="*60)
    print("🔍 Testing Plagiarism Detection")
    print("="*60)
    
    from graphplag import PlagiarismDetector
    
    detector = PlagiarismDetector(method='kernel', threshold=0.7)
    
    print(f"\n📊 Comparing documents...")
    report = detector.detect_plagiarism(texts[0], texts[1])
    
    print(f"\n✅ Analysis Complete!")
    print(f"   Similarity: {report.similarity_score:.2%}")
    print(f"   Plagiarism: {'YES' if report.is_plagiarism else 'NO'}")
    print()

print("="*60)
print("✅ Test Complete!")
print("="*60)
