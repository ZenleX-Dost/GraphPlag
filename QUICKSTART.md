# Quick Start Guide

This guide will help you get GraphPlag up and running quickly.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ZenleX-Dost/GraphPlag.git
cd GraphPlag
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy language models

```bash
# English (required)
python -m spacy download en_core_web_sm

# Optional: Better accuracy with transformer model
python -m spacy download en_core_web_trf

# Optional: Other languages
python -m spacy download es_core_news_sm  # Spanish
python -m spacy download fr_core_news_sm  # French
python -m spacy download de_core_news_sm  # German
```

### 5. Install in development mode

```bash
pip install -e .
```

## Usage Options

GraphPlag offers three ways to use the system:

1. **Web Interface** (Recommended for beginners) - Interactive browser-based UI
2. **Command Line Interface** - Terminal-based for automation and scripts
3. **Python API** - For integration into your own projects

---

## 🌐 Web Interface (Interactive)

The easiest way to use GraphPlag is through the web interface.

### Launch the Web App

```bash
python app.py
```

Then open your browser to: **http://localhost:7860**

### Features

- 📝 **Compare Two Documents** - Side-by-side comparison with real-time results
- 📚 **Batch Compare** - Analyze multiple documents at once
- 📊 **Interactive Visualizations** - Similarity gauges and heatmaps
- 💾 **Export Reports** - Save results as HTML, JSON, or text

### Web Interface Demo

![Web Interface](docs/images/web-interface.png)

---

## 💻 Command Line Interface

For automation and batch processing, use the CLI.

### Compare Two Documents

```bash
# Compare two files
python cli.py compare --file1 doc1.txt --file2 doc2.txt

# Compare text directly
python cli.py compare --text1 "First document" --text2 "Second document"

# With custom settings
python cli.py compare --file1 doc1.txt --file2 doc2.txt --method ensemble --threshold 0.8

# Save report
python cli.py compare --file1 doc1.txt --file2 doc2.txt --output report.html
```

### Batch Compare Multiple Documents

```bash
# Compare all files in a directory
python cli.py batch --directory ./documents

# Compare specific files
python cli.py batch --files doc1.txt doc2.txt doc3.txt --output results.json
```

### CLI Options

- `--method`: Detection method (`kernel`, `gnn`, `ensemble`)
- `--threshold`: Plagiarism threshold (0.0 to 1.0, default: 0.7)
- `--language`: Document language (`en`, `es`, `fr`, `de`)
- `--output`: Save report to file

---

## 🐍 Python API

For programmatic use, import GraphPlag in your Python code.

## Basic Usage

### Example 1: Simple Plagiarism Detection

```python
from graphplag import PlagiarismDetector

# Create detector
detector = PlagiarismDetector(method='kernel', threshold=0.7)

# Define documents
doc1 = "Machine learning is a field of artificial intelligence."
doc2 = "Machine learning is an AI field."

# Detect plagiarism
report = detector.detect_plagiarism(doc1, doc2)

# Print results
print(report.summary())
```

### Example 2: Compare Multiple Documents

```python
from graphplag import PlagiarismDetector

detector = PlagiarismDetector()

documents = [
    "Python is a programming language.",
    "Python is used for programming.",
    "Java is an object-oriented language.",
]

# Get similarity matrix
similarity_matrix = detector.batch_compare(documents)
print(similarity_matrix)

# Find suspicious pairs
suspicious = detector.identify_suspicious_pairs(documents, threshold=0.6)
for idx1, idx2, score in suspicious:
    print(f"Doc {idx1} <-> Doc {idx2}: {score:.3f}")
```

### Example 3: Visualize Results

```python
from graphplag import PlagiarismDetector
from graphplag.detection.report_generator import ReportGenerator

detector = PlagiarismDetector()
report_gen = ReportGenerator(output_dir="./reports")

# Detect plagiarism
doc1 = "Your first document..."
doc2 = "Your second document..."
report = detector.detect_plagiarism(doc1, doc2)

# Generate HTML report
report_gen.save_report(report, filename="my_report.html")
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_parser.py -v

# Run with coverage
pytest tests/ --cov=graphplag --cov-report=html
```

## Running Examples

```bash
python examples/basic_usage.py
```

## Configuration

Edit `graphplag/configs/config.yaml` to customize settings:

```yaml
detection:
  method: "ensemble"  # kernel, gnn, or ensemble
  threshold: 0.70

graph_builder:
  embedding_model: "paraphrase-multilingual-mpnet-base-v2"
  edge_strategy: "sequential"

similarity:
  kernels:
    types: ["wl", "rw", "sp"]
```

## Troubleshooting

### Issue: spaCy model not found

**Solution:**
```bash
python -m spacy download en_core_web_sm
```

### Issue: Out of memory errors

**Solutions:**
- Reduce batch size
- Use smaller documents
- Limit max_edge_distance in GraphBuilder
- Use CPU instead of GPU for small tasks

### Issue: Slow performance

**Solutions:**
- Use GPU if available
- Cache embeddings
- Reduce number of kernel iterations
- Use kernel-only method (faster than GNN)

## Next Steps

1. Read the [API Documentation](docs/API.md)
2. Explore the [Architecture](docs/ARCHITECTURE.md)
3. Check out [example scripts](examples/)
4. Run the test suite
5. Customize for your use case

## Common Use Cases

### Academic Plagiarism Detection

```python
detector = PlagiarismDetector(
    method='ensemble',
    threshold=0.75,  # Higher threshold for academic rigor
    language='en'
)
```

### Code Comment Similarity

```python
# Extract comments from code files first
detector = PlagiarismDetector(
    method='kernel',
    threshold=0.65
)
```

### Multilingual Detection

```python
detector = PlagiarismDetector(
    method='ensemble',
    embedding_model='paraphrase-multilingual-mpnet-base-v2'
)

# Will work across languages
english_doc = "This is English text."
spanish_doc = "Este es texto en español."
report = detector.detect_plagiarism(english_doc, spanish_doc)
```

## Performance Tips

1. **Batch Processing:** Process multiple documents together
2. **Cache Results:** Store computed embeddings and graphs
3. **Parallel Processing:** Use multiprocessing for large datasets
4. **GPU Acceleration:** Enable CUDA for GNN models
5. **Approximate Methods:** Trade accuracy for speed when needed

## Support

- Issues: [GitHub Issues](https://github.com/ZenleX-Dost/GraphPlag/issues)
- Documentation: [Full Documentation](docs/)
- Examples: [Example Scripts](examples/)

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.
