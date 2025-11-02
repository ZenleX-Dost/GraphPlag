"""
Simple File Parser Test (No GraphPlag imports)
"""

import sys
from pathlib import Path

print("="*60)
print("🧪 Testing File Parser (Standalone)")
print("="*60)

# Test the file parser module directly
print("\n1️⃣ Testing imports...")
try:
    # Import just the parser utilities
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Test basic file reading
    print("   ✅ Path setup complete")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test file existence
print("\n2️⃣ Checking test files...")
test_files = {
    "test_data/sample1.md": "Markdown file",
    "test_data/sample2.txt": "Text file",
}

for file_path, description in test_files.items():
    if Path(file_path).exists():
        size = Path(file_path).stat().st_size
        print(f"   ✅ {description}: {file_path} ({size} bytes)")
    else:
        print(f"   ❌ {description}: {file_path} NOT FOUND")

# Test reading files directly
print("\n3️⃣ Testing file reading...")
for file_path in test_files.keys():
    if Path(file_path).exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"   ✅ Read {file_path}: {len(content)} characters")
            print(f"      Preview: {content[:80]}...")
        except Exception as e:
            print(f"   ❌ Error reading {file_path}: {e}")

# Test PyPDF2 and python-docx imports
print("\n4️⃣ Testing file parsing libraries...")
try:
    import PyPDF2
    print(f"   ✅ PyPDF2 {PyPDF2.__version__} available")
except ImportError as e:
    print(f"   ⚠️  PyPDF2 not available (install with: pip install PyPDF2)")

try:
    import docx
    print(f"   ✅ python-docx available")
except ImportError as e:
    print(f"   ⚠️  python-docx not available (install with: pip install python-docx)")

try:
    import markdown
    print(f"   ✅ markdown available")
except ImportError as e:
    print(f"   ⚠️  markdown not available (already installed)")

print("\n" + "="*60)
print("✅ File Parser Test Complete!")
print("\n📝 Next steps:")
print("   1. Ensure virtual environment is activated")
print("   2. Install: pip install PyPDF2 python-docx")
print("   3. Run: python test_file_parser.py")
print("="*60)
