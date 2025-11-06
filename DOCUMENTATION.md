# GraphPlag - Complete Documentation

> **Last Updated:** November 6, 2025  
> **Version:** 1.0  
> **Python:** 3.10+

---

## 📑 Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Running the Application](#running-the-application)
4. [Features](#features)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting](#troubleshooting)
7. [Technical Details](#technical-details)
8. [API Reference](#api-reference)

---

## 🚀 Quick Start

### One Command for Everything

```powershell
.\run.bat
```

This single launcher provides:
1. **Setup** - First-time installation
2. **Fix NumPy** - Compatibility fixes
3. **Web Interface** - User-friendly UI (recommended)
4. **Enhanced Web** - Advanced features
5. **REST API** - Developer endpoints
6. **CLI** - Command-line tools
7. **Tests** - Run test suite
8. **API Docs** - Open documentation

**First time?** Just run `.\run.bat` and choose option [1] to setup everything!

---

## 📦 Installation

### Single Command Installation

```powershell
.\run.bat
```

Choose **[1] Setup GraphPlag** from the menu.

This automatically installs ALL features:
- ✅ Core plagiarism detection with graph kernels
- ✅ NLP tools (spaCy, Stanza, Transformers)
- ✅ Web interface (Gradio)
- ✅ REST API (FastAPI)
- ✅ Visualization tools
- ✅ PDF/Excel report generation
- ✅ Development tools

**First time setup takes 5-10 minutes.**

### Fix NumPy Issues

If you encounter NumPy 2.x compatibility errors:

```powershell
.\run.bat
```

Choose **[2] Fix NumPy Compatibility Issues** from the menu.

---

## 🎯 Running the Application

### Option 1: Interactive Menu (Recommended)

```powershell
.\run.bat
```

**Menu Options:**
1. **Web Interface (Gradio)** - User-friendly web UI
2. **Enhanced Web Interface** - Advanced features
3. **REST API Server** - For developers
4. **CLI Mode** - Command-line comparison
5. **Run Tests** - Comprehensive testing

### Option 2: Direct Launchers

```powershell
.\start_web.bat     # Web interface
.\start_api.bat     # REST API only
```

### Option 3: Manual Commands

```bash
# Web Interface
python app.py

# Enhanced Web Interface  
python app_enhanced.py

# REST API
python -m uvicorn api:app --reload

# CLI
python cli.py compare --file1 doc1.txt --file2 doc2.txt
```

---

## ✨ Features

### Core Plagiarism Detection
- **Graph-based Analysis**: Uses syntactic dependency graphs
- **Multiple Kernels**: Weisfeiler-Lehman, Shortest Path, Random Walk
- **Sentence Embeddings**: Multilingual support via Sentence-Transformers
- **Paraphrase Detection**: Detects heavily modified text
- **Cross-lingual**: Compare documents in different languages

### Performance
- ⚡ **8x faster** with disk-based caching
- 📊 **50MB+ files** supported with intelligent chunking
- 🔄 **Async processing** for batch operations
- 💾 **Smart caching** with LRU eviction

### File Support
- **Text Files**: .txt, .md
- **Documents**: .pdf, .docx
- **Direct Input**: Paste text directly

### Output Formats
- 📄 **PDF Reports**: Professional with highlighting
- 📊 **Excel Reports**: Multi-sheet with color-coding
  - Red (90-100%): High similarity
  - Orange (70-89%): Medium similarity
  - Yellow (<70%): Low similarity
- 🖥️ **Console Output**: Detailed text results
- 📈 **Visual Graphs**: Similarity matrices, match highlights

### Interfaces
1. **Web UI** (Gradio): User-friendly, no coding required
2. **Enhanced Web UI**: Advanced visualizations
3. **REST API**: FastAPI with Swagger docs
4. **CLI**: Command-line for automation
5. **Python API**: Direct import in scripts

---

## 📖 Usage Examples

### 1. Web Interface (Easiest)

```powershell
.\start_web.bat
```

1. Upload two documents or paste text
2. Click "Compare"
3. View results with visual graphs
4. Download PDF/Excel report

### 2. Command Line Interface

```bash
# Basic comparison
python cli.py compare --file1 doc1.txt --file2 doc2.txt

# Generate PDF report
python cli.py compare --file1 doc1.txt --file2 doc2.txt --output report.pdf

# Generate Excel report
python cli.py compare --file1 doc1.txt --file2 doc2.txt --output report.xlsx

# Specify method and threshold
python cli.py compare --file1 doc1.txt --file2 doc2.txt --method kernel --threshold 0.7
```

### 3. Python API

```python
from graphplag.detection.detector import PlagiarismDetector

# Initialize with caching
detector = PlagiarismDetector(
    method='kernel',
    use_cache=True,
    enable_chunking=True
)

# Compare documents
text1 = "Your first document..."
text2 = "Your second document..."

result = detector.detect_plagiarism(text1, text2)

# Access results
print(f"Overall Similarity: {result.overall_similarity:.2%}")
print(f"Is Plagiarism: {result.is_plagiarism}")
print(f"Kernel Scores: {result.kernel_scores}")

# Get matched sentences
for match in result.sentence_matches:
    print(f"Similarity: {match.similarity:.2%}")
    print(f"Doc1: {match.sentence1}")
    print(f"Doc2: {match.sentence2}\n")

# Generate reports
from graphplag.utils.export import PDFReportGenerator, ExcelReportGenerator

pdf_gen = PDFReportGenerator()
pdf_gen.generate_report(result, "report.pdf", include_full_text=True)

excel_gen = ExcelReportGenerator()
excel_gen.generate_report(result, "report.xlsx")
```

### 4. REST API

**Start API:**
```bash
python -m uvicorn api:app --reload
```

**Access Docs:** http://localhost:8000/docs

**Example Request:**
```bash
curl -X POST "http://localhost:8000/compare/text" \
  -H "Authorization: Bearer demo_key_123" \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "This is the first document.",
    "text2": "This is the second document.",
    "method": "kernel",
    "threshold": 0.7
  }'
```

**Upload Files:**
```bash
curl -X POST "http://localhost:8000/compare/files" \
  -H "Authorization: Bearer demo_key_123" \
  -F "file1=@document1.pdf" \
  -F "file2=@document2.docx"
```

---

## 🔧 Troubleshooting

### NumPy Compatibility Error

**Error:**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

**Solution:**
```bash
.\fix_numpy.bat
```

This uninstalls NumPy 2.x and installs NumPy 1.x (GraKeL compatible).

### GraKeL Import Error

**Error:**
```
ModuleNotFoundError: No module named 'grakel'
```

**Solution:**
```bash
pip install "numpy<2.0.0"
pip install grakel
```

### Out of Memory

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```python
# Enable chunking for large files
detector = PlagiarismDetector(
    enable_chunking=True,
    max_chunk_size=1000  # Adjust based on RAM
)
```

### Gradio Not Found

**Error:**
```
ModuleNotFoundError: No module named 'gradio'
```

**Solution:**
```bash
pip install gradio>=5.0.0
```

### Cache Issues

**Clear cache:**
```python
from graphplag.core.graph_builder import GraphBuilder

builder = GraphBuilder(use_cache=True)
builder.clear_cache()
```

Or delete manually:
```bash
rm -r .cache/  # Linux/Mac
rmdir /s .cache\  # Windows
```

---

## 🏗️ Technical Details

### Architecture

```
GraphPlag/
├── graphplag/
│   ├── core/               # Graph construction
│   │   ├── graph_builder.py    # Graph building with caching
│   │   └── parsers.py          # Document parsers
│   ├── detection/          # Detection logic
│   │   └── detector.py         # Main detector class
│   ├── preprocessing/      # Text preprocessing
│   │   └── text_processor.py   # NLP processing
│   └── utils/             # Utilities
│       ├── cache.py            # Embedding cache (8x speedup)
│       ├── large_file_utils.py # Large file handling
│       └── export.py           # PDF/Excel generation
├── app.py                  # Gradio web interface
├── app_enhanced.py         # Enhanced web interface
├── api.py                  # FastAPI REST API
├── cli.py                  # Command-line interface
├── run.bat                 # Interactive launcher
└── setup_local.bat         # One-click setup
```

### Technology Stack

- **NLP**: spaCy, Stanza, Transformers, Sentence-Transformers
- **Graph Processing**: NetworkX, GraKeL
- **Deep Learning**: PyTorch
- **API**: FastAPI, Uvicorn
- **UI**: Gradio
- **Reports**: ReportLab (PDF), OpenPyXL (Excel)
- **Visualization**: Matplotlib, Plotly, Seaborn

### How It Works

1. **Preprocessing**
   - Tokenization and sentence segmentation
   - Language detection
   - Dependency parsing with spaCy/Stanza

2. **Graph Construction**
   - Each sentence becomes a node
   - Node features from sentence embeddings
   - Edges from document structure

3. **Similarity Computation**
   - **Kernel Methods**: Weisfeiler-Lehman, Shortest Path, Random Walk
   - **Ensemble**: Combines multiple kernel scores
   - **Threshold**: Configurable similarity threshold (default: 0.7)

4. **Result Generation**
   - Overall similarity score
   - Sentence-level matches
   - Kernel-specific scores
   - Visual reports

### Performance Optimization

**Caching System:**
- SHA-256 content hashing
- Disk-based storage (pickle format)
- LRU eviction (500MB default limit)
- 30-day expiration
- ~8x speedup on repeated comparisons

**Large File Handling:**
- Automatic chunking for 50MB+ files
- Overlap between chunks (configurable)
- Streaming file parser
- Memory monitoring
- Progress tracking

**Batch Processing:**
- Async API endpoints
- Background job queue
- Status tracking via job ID
- Parallel processing

---

## 📡 API Reference

### REST API Endpoints

Base URL: `http://localhost:8000`

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "cache_enabled": true,
  "cache_stats": {...}
}
```

#### Compare Text
```http
POST /compare/text
Authorization: Bearer demo_key_123
Content-Type: application/json

{
  "text1": "First document text",
  "text2": "Second document text",
  "method": "kernel",
  "threshold": 0.7
}
```

**Response:**
```json
{
  "overall_similarity": 0.85,
  "is_plagiarism": true,
  "method": "kernel",
  "kernel_scores": {
    "weisfeiler_lehman": 0.83,
    "shortest_path": 0.87,
    "random_walk": 0.85
  },
  "processing_time": 1.23,
  "matched_sentences": [...]
}
```

#### Compare Files
```http
POST /compare/files
Authorization: Bearer demo_key_123
Content-Type: multipart/form-data

file1: <binary>
file2: <binary>
method: kernel (optional)
threshold: 0.7 (optional)
```

#### Batch Comparison
```http
POST /batch/compare
Authorization: Bearer demo_key_123
Content-Type: application/json

{
  "comparisons": [
    {"text1": "...", "text2": "..."},
    {"text1": "...", "text2": "..."}
  ],
  "method": "kernel",
  "threshold": 0.7
}
```

**Response:**
```json
{
  "job_id": "abc123",
  "status": "processing",
  "total_comparisons": 2
}
```

#### Check Job Status
```http
GET /job/{job_id}
Authorization: Bearer demo_key_123
```

#### Cache Management
```http
GET /cache/stats      # Get cache statistics
DELETE /cache         # Clear cache
```

### Python API

```python
from graphplag.detection.detector import PlagiarismDetector

# Initialize
detector = PlagiarismDetector(
    method='kernel',           # 'kernel', 'gnn', or 'ensemble'
    threshold=0.7,             # Similarity threshold
    use_cache=True,            # Enable caching (8x faster)
    enable_chunking=True,      # Handle large files
    max_chunk_size=1000        # Sentences per chunk
)

# Detect plagiarism
result = detector.detect_plagiarism(text1, text2)

# Access results
result.overall_similarity    # float: 0.0-1.0
result.is_plagiarism        # bool
result.kernel_scores        # dict
result.sentence_matches     # list of matches
result.processing_time      # float (seconds)
```

### Cache API

```python
from graphplag.core.graph_builder import GraphBuilder

builder = GraphBuilder(use_cache=True, cache_dir=".cache")

# Get statistics
stats = builder.get_cache_stats()
print(f"Cache hits: {stats['hits']}")
print(f"Cache size: {stats['size_mb']} MB")

# Clear cache
builder.clear_cache()
```

---

## 📊 Configuration

### Environment Variables

Create `.env` file:

```ini
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_KEY_1=your-secure-key-here

# Cache Settings
ENABLE_CACHE=true
CACHE_MAX_SIZE_MB=500
CACHE_MAX_AGE_DAYS=30
CACHE_DIR=.cache

# Processing
ENABLE_CHUNKING=true
MAX_CHUNK_SIZE=1000
MAX_FILE_SIZE_MB=50

# Model
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2
DEFAULT_METHOD=kernel
DEFAULT_THRESHOLD=0.7

# Logging
LOG_LEVEL=INFO
LOG_FILE=graphplag.log
```

### Python Configuration

```python
from graphplag.detection.detector import PlagiarismDetector

detector = PlagiarismDetector(
    # Detection method
    method='kernel',  # 'kernel', 'gnn', 'ensemble'
    
    # Similarity threshold
    threshold=0.7,  # 0.0-1.0
    
    # Performance
    use_cache=True,
    cache_dir='.cache',
    
    # Large files
    enable_chunking=True,
    max_chunk_size=1000,
    chunk_overlap=50,
    
    # Models
    embedding_model='paraphrase-multilingual-mpnet-base-v2',
    
    # NLP backend
    nlp_backend='spacy',  # 'spacy' or 'stanza'
    language='en'
)
```

---

## 🎓 Citation

If you use GraphPlag in your research, please cite:

```bibtex
@software{graphplag2025,
  title={GraphPlag: Semantic Graph-Based Plagiarism Detection System},
  author={GraphPlag Team},
  year={2025},
  url={https://github.com/ZenleX-Dost/GraphPlag}
}
```

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🆘 Support

- **Issues**: https://github.com/ZenleX-Dost/GraphPlag/issues
- **Documentation**: This file
- **API Docs**: http://localhost:8000/docs (when server running)

---

## ✅ Quick Reference

### Installation
```bash
.\setup_local.bat
```

### Run Web Interface
```bash
.\run.bat
# Choose [1]
```

### Run API
```bash
.\start_api.bat
```

### CLI Usage
```bash
python cli.py compare --file1 doc1.txt --file2 doc2.txt --output report.pdf
```

### Fix Issues
```bash
.\fix_numpy.bat
```

---

**GraphPlag - Detect plagiarism with graph kernels and machine learning** 🚀
