# GraphPlag: Semantic Graph-Based Plagiarism Detection System

GraphPlag is an advanced plagiarism detection system that uses semantic graph representations rather than traditional lexical matching. Documents are transformed into syntactic dependency graphs where sentences become nodes and their relationships become edges. The system leverages Graph Kernels and Graph Neural Networks (GNNs) to compute semantic similarity scores between documents.

## Core Innovation

Moves beyond surface-level text comparison to capture deep semantic relationships, enabling detection of paraphrased plagiarism and cross-lingual semantic copying.

## Features

- **Semantic Analysis**: Graph-based representation captures deep semantic relationships
- **Dual Approach**: Combines Graph Kernels and GNN-based similarity computation
- **Multilingual Support**: Cross-lingual plagiarism detection capabilities
- **Paraphrase Robustness**: Detects plagiarism even with heavy paraphrasing
- **Explainability**: Visualize which parts contributed to similarity scores

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

# Initialize detector
detector = PlagiarismDetector(method='ensemble')

# Compare two documents
doc1 = "Your first document text here..."
doc2 = "Your second document text here..."

report = detector.detect_plagiarism(doc1, doc2)
print(f"Similarity Score: {report.similarity_score:.3f}")
print(f"Plagiarism Detected: {report.is_plagiarism}")
```

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
├── similarity/        # Graph kernels and GNN models
├── detection/         # Plagiarism detection orchestrator
├── utils/            # Utilities and visualization
├── data/             # Data loaders and datasets
├── configs/          # Configuration files
└── experiments/      # Training and evaluation scripts
```

## Technology Stack

- **NLP**: spaCy, Stanza, Transformers (Hugging Face)
- **Graph Processing**: NetworkX, PyTorch Geometric, GraKeL
- **Deep Learning**: PyTorch
- **Visualization**: Matplotlib, Plotly, PyVis

## Evaluation

Target performance metrics:
- F1-Score: >85% on PAN plagiarism corpus
- Support for 5+ languages with cross-lingual detection
- Processing time: <5 seconds per document pair

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
