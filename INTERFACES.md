# GraphPlag Interfaces Guide

GraphPlag provides three powerful interfaces for plagiarism detection:

## 🌐 Web Interface (Gradio)

### Quick Start

```bash
python app.py
```

Then open: **http://localhost:7860**

### Features

#### Tab 1: Compare Two Documents
- **Side-by-side editor** for easy document comparison
- **Real-time analysis** with similarity scoring
- **Interactive gauge visualization** showing similarity percentage
- **Detailed reports** with interpretation

#### Tab 2: Batch Compare
- **Multiple document analysis** - separate documents with `---`
- **Similarity matrix heatmap** showing all comparisons
- **Suspicious pairs detection** highlighting potential plagiarism
- **Exportable results** in multiple formats

#### Tab 3: Examples & Help
- **Complete documentation** within the app
- **Example documents** for testing
- **Tips and best practices**
- **System information**

### Configuration Options

- **Method**: `kernel`, `gnn`, or `ensemble`
- **Threshold**: 0.0 to 1.0 (default: 0.7)
- **Language**: `en`, `es`, `fr`, `de`

### Advanced Options

```python
# In app.py, modify the launch parameters:
app.launch(
    server_name="0.0.0.0",  # Allow external access
    server_port=7860,        # Port number
    share=True,              # Create public URL (optional)
    auth=("user", "pass")    # Add authentication (optional)
)
```

---

## 💻 Command Line Interface (CLI)

### Installation

The CLI is ready to use after installing GraphPlag:

```bash
pip install -e .
```

### Commands

#### Compare Two Documents

```bash
# Compare files
python cli.py compare --file1 doc1.txt --file2 doc2.txt

# Compare text directly
python cli.py compare \
  --text1 "First document text here" \
  --text2 "Second document text here"

# With custom settings
python cli.py compare \
  --file1 doc1.txt \
  --file2 doc2.txt \
  --method ensemble \
  --threshold 0.8 \
  --language en

# Save report
python cli.py compare \
  --file1 doc1.txt \
  --file2 doc2.txt \
  --output report.html  # or .json, .txt
```

#### Batch Compare

```bash
# Compare all .txt files in directory
python cli.py batch --directory ./documents

# Compare specific files
python cli.py batch \
  --files doc1.txt doc2.txt doc3.txt doc4.txt

# With settings and output
python cli.py batch \
  --directory ./essays \
  --threshold 0.75 \
  --output results.json
```

### CLI Options

| Option | Description | Values | Default |
|--------|-------------|--------|---------|
| `--method` | Detection algorithm | `kernel`, `gnn`, `ensemble` | `kernel` |
| `--threshold` | Plagiarism threshold | 0.0 - 1.0 | 0.7 |
| `--language` | Document language | `en`, `es`, `fr`, `de` | `en` |
| `--output` | Save report to file | `.txt`, `.json`, `.html` | None |

### Output Formats

#### Terminal Output
```
============================================================
📊 PLAGIARISM DETECTION RESULTS
============================================================

📈 Similarity Score: 87.50%
🎯 Threshold: 70.00%
⚠️  Plagiarism Detected: YES ✓
🔬 Method Used: KERNEL

📝 Interpretation:
   🟠 High Similarity - Likely plagiarism detected

============================================================
```

#### JSON Output (`--output report.json`)
```json
{
  "similarity_score": 0.875,
  "is_plagiarism": true,
  "method": "kernel",
  "threshold": 0.7,
  "doc1_length": 1234,
  "doc2_length": 1189
}
```

#### HTML Output (`--output report.html`)
- Full formatted report
- Visualizations included
- Printable format

---

## 🐍 Python API

### Basic Usage

```python
from graphplag import PlagiarismDetector

# Initialize detector
detector = PlagiarismDetector(
    method='kernel',
    threshold=0.7,
    language='en'
)

# Compare documents
doc1 = "Your first document..."
doc2 = "Your second document..."

report = detector.detect_plagiarism(doc1, doc2)

print(f"Similarity: {report.similarity_score:.2%}")
print(f"Plagiarism: {report.is_plagiarism}")
```

### Advanced Usage

```python
from graphplag import PlagiarismDetector
from graphplag.detection.report_generator import ReportGenerator

# Initialize
detector = PlagiarismDetector(method='ensemble', threshold=0.75)
report_gen = ReportGenerator()

# Batch processing
documents = [
    "Document 1 content...",
    "Document 2 content...",
    "Document 3 content..."
]

# Get similarity matrix
results = []
for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        report = detector.detect_plagiarism(documents[i], documents[j])
        if report.is_plagiarism:
            results.append({
                'pair': (i+1, j+1),
                'similarity': report.similarity_score
            })

# Generate reports
for result in results:
    print(f"Documents {result['pair']}: {result['similarity']:.2%}")
```

---

## 🎯 Use Cases

### Academic Institutions

**Web Interface**: Best for instructors checking individual assignments
```bash
python app.py
# Upload student submissions through browser
```

**CLI**: Best for batch processing all submissions
```bash
python cli.py batch --directory ./submissions --threshold 0.8
```

### Content Creators

**API**: Integrate into CMS or publishing workflow
```python
detector = PlagiarismDetector(method='kernel', threshold=0.6)
result = detector.detect_plagiarism(user_content, existing_content)
if result.is_plagiarism:
    reject_submission()
```

### Research & Development

**CLI**: Automate testing and experiments
```bash
for file1 in corpus/*.txt; do
  for file2 in corpus/*.txt; do
    python cli.py compare --file1 "$file1" --file2 "$file2" >> results.txt
  done
done
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Set default method
export GRAPHPLAG_METHOD=ensemble

# Set default threshold
export GRAPHPLAG_THRESHOLD=0.75

# Set cache directory
export GRAPHPLAG_CACHE=~/.graphplag/cache
```

### Config File

Create `config.yaml`:

```yaml
detection:
  method: ensemble
  threshold: 0.75
  
similarity:
  kernels:
    types: [wl]
    
embedding:
  model: paraphrase-multilingual-mpnet-base-v2
  batch_size: 32
```

---

## 🐛 Troubleshooting

### Web Interface Issues

**Port already in use:**
```python
# In app.py, change:
app.launch(server_port=7861)  # Use different port
```

**Can't access from other devices:**
```python
# In app.py:
app.launch(server_name="0.0.0.0")  # Allow external access
```

### CLI Issues

**Module not found:**
```bash
# Make sure you're in the virtual environment
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac

# Reinstall in development mode
pip install -e .
```

**Encoding errors:**
```bash
# Specify encoding for files
python cli.py compare --file1 doc1.txt --file2 doc2.txt
# Files should be UTF-8 encoded
```

---

## 📊 Performance Tips

### Web Interface
- Keep documents under 10,000 characters for best performance
- Use `kernel` method for real-time analysis
- Use `ensemble` method for final verification

### CLI
- Use `batch` command for multiple comparisons (more efficient)
- Redirect output to file for large batches: `python cli.py batch ... > results.txt`
- Use JSON output for programmatic processing

### API
- Reuse detector instance for multiple comparisons
- Cache embeddings when possible
- Use batch processing methods

---

## 🚀 Next Steps

1. **Try the Web Interface**: `python app.py`
2. **Test the CLI**: `python cli.py compare --help`
3. **Read Full Documentation**: See `docs/` folder
4. **Explore Examples**: Check `examples/` folder

---

## 📝 Support

- **Documentation**: [docs/](../docs/)
- **Examples**: [examples/](../examples/)
- **Issues**: [GitHub Issues](https://github.com/ZenleX-Dost/GraphPlag/issues)
- **Quickstart**: [QUICKSTART.md](QUICKSTART.md)
