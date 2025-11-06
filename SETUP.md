# GraphPlag - Local Development Setup

## 🚀 Quick Start (One-Click Installation)

### Windows:
```batch
setup_local.bat
```

### Linux/Mac:
```bash
chmod +x setup_local.sh
./setup_local.sh
```

This will install **ALL** features including:
- ✅ Core plagiarism detection
- ✅ Graph kernels (GraKeL)
- ✅ NLP tools (spaCy, Stanza, Transformers)
- ✅ REST API (FastAPI)
- ✅ Visualization (Matplotlib, Plotly, Seaborn)
- ✅ Report generation (PDF, Excel)
- ✅ Development tools (pytest, black, flake8)
- ✅ Experiment tracking (wandb, tensorboard)
- ✅ Interactive UI (Gradio)

---

## 📋 Manual Installation

If you prefer manual setup:

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Run setup script
.\setup_local.bat  # Windows
# ./setup_local.sh  # Linux/Mac
```

---

## 🔧 Fix NumPy Issues

If you already have GraphPlag installed but getting NumPy errors:

```batch
# Windows
.\fix_numpy.bat

# Linux/Mac
./fix_numpy.sh
```

This will:
- Remove NumPy 2.x
- Install NumPy 1.x (GraKeL compatible)
- Reinstall all dependencies

---

## 📦 Dependencies

### Core (requirements.txt):
- NumPy 1.x (pinned for GraKeL)
- SciPy, NetworkX
- PyTorch, Transformers
- spaCy, Stanza, Sentence-Transformers
- GraKeL, PyTorch Geometric
- Scikit-learn
- Visualization: Matplotlib, Plotly, Seaborn, PyVis
- Gradio (interactive UI)
- File parsing: PyPDF2, python-docx
- Development: pytest, black, flake8

### API (requirements-api.txt):
- FastAPI, Uvicorn
- ReportLab (PDF generation)
- OpenPyXL (Excel generation)
- python-multipart (file uploads)

---

## 🎯 Usage

### Command Line:
```bash
# Compare two documents
python cli.py compare --file1 doc1.txt --file2 doc2.txt

# Generate PDF report
python cli.py compare --file1 doc1.txt --file2 doc2.txt --output report.pdf

# Generate Excel report
python cli.py compare --file1 doc1.txt --file2 doc2.txt --output report.xlsx
```

### Python API:
```python
from graphplag.detection.detector import PlagiarismDetector

# Initialize with caching
detector = PlagiarismDetector(method='kernel', use_cache=True)

# Detect plagiarism
result = detector.detect_plagiarism(text1, text2)
print(f"Similarity: {result.overall_similarity:.2%}")

# Generate reports
from graphplag.utils.export import PDFReportGenerator, ExcelReportGenerator

pdf_gen = PDFReportGenerator()
pdf_gen.generate_report(result, "report.pdf")

excel_gen = ExcelReportGenerator()
excel_gen.generate_report(result, "report.xlsx")
```

### REST API:
```bash
# Start API server
python -m uvicorn api:app --reload

# Access interactive docs
open http://localhost:8000/docs

# Test endpoint
curl -X POST "http://localhost:8000/compare/text" \
  -H "Authorization: Bearer demo_key_123" \
  -H "Content-Type: application/json" \
  -d '{"text1":"test","text2":"test"}'
```

---

## 🧪 Run Tests:
```bash
python test_comprehensive.py
```

---

## 📚 Documentation

- **FEATURES_SUMMARY.md** - All features overview
- **NUMPY_FIX.md** - NumPy compatibility details
- **SIZE_OPTIMIZATION.md** - Deployment optimization (if needed later)

---

## 💡 Tips

1. **First time setup**: Run `setup_local.bat` - it handles everything
2. **NumPy issues**: Run `fix_numpy.bat` to fix compatibility
3. **Cache**: Enable caching for 8x performance boost
4. **Large files**: Automatic chunking for 50MB+ documents
5. **API**: Use `/docs` for interactive API testing

---

## 🆘 Troubleshooting

### NumPy 2.x Error:
```
Error: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```
**Solution**: Run `.\fix_numpy.bat`

### GraKeL Import Error:
```
Error: cannot import grakel
```
**Solution**: Ensure NumPy 1.x is installed first: `pip install "numpy<2.0.0"`

### Out of Memory:
```
Error: MemoryError
```
**Solution**: Enable chunking for large files:
```python
detector = PlagiarismDetector(enable_chunking=True, max_chunk_size=1000)
```

---

## ✨ Features

- **8x faster** with caching
- Handles **50MB+** documents
- **Multilingual** support
- **PDF & Excel** reports with highlighting
- **REST API** ready
- **Professional output** with color-coding

---

**Ready to detect plagiarism! 🚀**
