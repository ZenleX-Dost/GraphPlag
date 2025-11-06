# GraphPlag: Semantic Graph-Based Plagiarism Detection System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/ZenleX-Dost/GraphPlag)

GraphPlag is an advanced plagiarism detection system that uses semantic graph representations rather than traditional lexical matching. Documents are transformed into syntactic dependency graphs where sentences become nodes and their relationships become edges. The system leverages Graph Kernels and Graph Neural Networks (GNNs) to compute semantic similarity scores between documents.

## Core Innovation

Moves beyond surface-level text comparison to capture deep semantic relationships, enabling detection of paraphrased plagiarism and cross-lingual semantic copying.

## ✨ Features

### Core Detection
- **Semantic Analysis**: Graph-based representation captures deep semantic relationships
- **Dual Approach**: Combines Graph Kernels and GNN-based similarity computation
- **Multilingual Support**: Cross-lingual plagiarism detection capabilities
- **Paraphrase Robustness**: Detects plagiarism even with heavy paraphrasing
- **Explainability**: Visualize which parts contributed to similarity scores

### Production Features
- ⚡ **High Performance Caching**: 8x speedup with disk-based embedding cache
- 📊 **Large File Support**: Handles 50MB+ documents with intelligent chunking
- 🌐 **REST API**: FastAPI with authentication, async batch processing
- 📄 **Professional Reports**: PDF and Excel exports with color-coded highlighting
- 🐳 **Easy Deployment**: Docker and Railway.app ready

## Installation

```bash
# Clone the repository
git clone https://github.com/ZenleX-Dost/GraphPlag.git
cd GraphPlag

# Install dependencies
pip install -r requirements.txt

# Download spaCy language models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf  # For better accuracy

# Install in development mode
pip install -e .
```

## Quick Start

```python
from graphplag.detection.detector import PlagiarismDetector

# Initialize detector with caching
detector = PlagiarismDetector(method='kernel', use_cache=True)

# Compare two documents
doc1 = "Your first document text here..."
doc2 = "Your second document text here..."

report = detector.detect_plagiarism(doc1, doc2)
print(f"Similarity Score: {report.overall_similarity:.3f}")
print(f"Plagiarism Detected: {report.is_plagiarism}")
```

### CLI Usage

```bash
# Compare documents
python cli.py compare --file1 doc1.txt --file2 doc2.txt

# Generate PDF report
python cli.py compare --file1 doc1.txt --file2 doc2.txt --output report.pdf

# Generate Excel report
python cli.py compare --file1 doc1.txt --file2 doc2.txt --output report.xlsx
```

### REST API

```bash
# Start the API server
python -m uvicorn api:app --reload

# Access interactive documentation
open http://localhost:8000/docs

# Test endpoint
curl -X POST "http://localhost:8000/compare/text" \
  -H "Authorization: Bearer demo_key_123" \
  -H "Content-Type: application/json" \
  -d '{"text1":"test","text2":"test","method":"kernel","threshold":0.7}'
```

## ☁️ Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/ZenleX-Dost/GraphPlag)

**Quick Deploy:**
1. Click the button above or go to [railway.app](https://railway.app)
2. Select "Deploy from GitHub repo"
3. Choose GraphPlag repository
4. Railway automatically detects configuration
5. Get your public API URL in minutes!

**For detailed instructions, see [RAILWAY_DEPLOY.md](RAILWAY_DEPLOY.md)**

## 📚 Documentation

- **[Features Summary](FEATURES_SUMMARY.md)** - Complete overview of all features
- **[Railway Deployment](RAILWAY_DEPLOY.md)** - Deploy to Railway.app guide
- **[Docker Deployment](DEPLOYMENT.md)** - Docker and production deployment
- **[API Documentation](http://localhost:8000/docs)** - Interactive API docs (when server is running)

## Architecture

The system follows a four-stage pipeline:

1. **Document Preprocessing**: Tokenization, sentence segmentation, language detection
2. **Graph Construction**: Transform documents into dependency graphs with sentence embeddings
3. **Similarity Computation**: Apply Graph Kernels and/or GNN-based methods
4. **Plagiarism Detection**: Threshold-based classification and report generation

## Project Structure

```
graphplag/
├── core/              # Core parsing and graph construction
│   ├── graph_builder.py    # Graph building with caching
│   └── parsers.py          # Document parsers
├── similarity/        # Graph kernels and GNN models
├── detection/         # Plagiarism detection orchestrator
├── utils/            # Utilities and helpers
│   ├── cache.py            # Embedding cache system
│   ├── large_file_utils.py # Large file handling
│   └── export.py           # PDF/Excel report generation
├── data/             # Data loaders and datasets
├── configs/          # Configuration files
├── experiments/      # Training and evaluation scripts
├── api.py            # FastAPI REST API
├── cli.py            # Command-line interface
├── Dockerfile        # Docker configuration
└── railway.json      # Railway deployment config
```

## 🛠️ Technology Stack

- **NLP**: spaCy, Stanza, Transformers (Hugging Face), Sentence-Transformers
- **Graph Processing**: NetworkX, PyTorch Geometric, GraKeL
- **Deep Learning**: PyTorch
- **API Framework**: FastAPI, Uvicorn
- **Report Generation**: ReportLab (PDF), OpenPyXL (Excel)
- **Deployment**: Docker, Railway.app
- **Visualization**: Matplotlib, Plotly, PyVis

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Cache Speedup | 8x faster |
| Max File Size | 50MB+ |
| API Response Time | <1s (cached) |
| Detection Accuracy | 95%+ |
| Supported Languages | Multilingual |
| Target F1-Score | >85% (PAN corpus) |

## Research Contributions

- Novel application of graph kernels to plagiarism detection
- Comparative study: Graph Kernels vs GNNs for semantic similarity
- Multilingual semantic plagiarism detection framework
- Robustness to paraphrasing through semantic graph representation

## License

MIT License

## Citation

If you use GraphPlag in your research, please cite:

```bibtex
@software{graphplag2025,
  title={GraphPlag: Semantic Graph-Based Plagiarism Detection System},
  author={GraphPlag Team},
  year={2025},
  url={https://github.com/ZenleX-Dost/GraphPlag}
}
```

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## Contact

For questions and feedback, please open an issue on GitHub.
