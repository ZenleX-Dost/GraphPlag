# GraphPlag API Documentation

## Core Modules

### DocumentParser

The `DocumentParser` class handles document preprocessing and linguistic analysis.

```python
from graphplag.core.document_parser import DocumentParser

# Initialize parser
parser = DocumentParser(language='en')

# Parse a document
document = parser.parse_document(
    "Your text here...",
    doc_id="doc1",
    auto_detect_language=True
)

# Batch parsing
documents = parser.parse_batch(
    texts=["text1", "text2", "text3"],
    doc_ids=["doc1", "doc2", "doc3"]
)
```

**Methods:**
- `parse_document(text, doc_id=None, auto_detect_language=False)`: Parse a single document
- `extract_sentences(doc)`: Extract sentences from spaCy Doc
- `detect_language(text)`: Detect document language
- `parse_batch(texts, doc_ids=None, batch_size=32)`: Parse multiple documents

### GraphBuilder

The `GraphBuilder` class transforms documents into graph representations.

```python
from graphplag.core.graph_builder import GraphBuilder

# Initialize builder
builder = GraphBuilder(
    embedding_model="paraphrase-multilingual-mpnet-base-v2",
    edge_strategy="sequential",
    max_edge_distance=3
)

# Build graph
graph = builder.build_graph(document, graph_type="networkx")

# Batch building
graphs = builder.build_batch(documents)
```

**Methods:**
- `build_graph(document, graph_type='networkx')`: Build graph from document
- `create_node_features(sentence)`: Generate node features
- `extract_dependencies(sentence)`: Extract dependency relations
- `build_batch(documents, graph_type='networkx')`: Build multiple graphs

### GraphKernelSimilarity

Compute similarity using graph kernel methods.

```python
from graphplag.similarity.graph_kernels import GraphKernelSimilarity

# Initialize
kernel_sim = GraphKernelSimilarity(
    kernel_types=['wl', 'rw', 'sp'],
    normalize=True,
    wl_iterations=5
)

# Compute similarity
result = kernel_sim.compute_similarity(graph1, graph2, method='ensemble')
print(f"Similarity: {result.score:.3f}")

# Individual kernels
wl_score = kernel_sim.compute_wl_kernel(graph1.graph_data, graph2.graph_data)
rw_score = kernel_sim.compute_random_walk_kernel(graph1.graph_data, graph2.graph_data)
sp_score = kernel_sim.compute_shortest_path_kernel(graph1.graph_data, graph2.graph_data)
```

**Kernel Types:**
- `wl`: Weisfeiler-Lehman kernel
- `rw`: Random Walk kernel
- `sp`: Shortest Path kernel

### GNNSimilarity

Compute similarity using Graph Neural Networks.

```python
from graphplag.similarity.gnn_models import GNNSimilarity, SiameseGNN

# Initialize with pre-trained model
gnn_sim = GNNSimilarity(model_path="path/to/model.pth")

# Or with default model
gnn_sim = GNNSimilarity()

# Compute similarity
result = gnn_sim.compute_similarity(graph1, graph2)

# Encode graphs to embeddings
embedding1 = gnn_sim.encode_graph(graph1)
embedding2 = gnn_sim.encode_graph(graph2)
```

### PlagiarismDetector

Main orchestrator for plagiarism detection.

```python
from graphplag.detection.detector import PlagiarismDetector

# Initialize detector
detector = PlagiarismDetector(
    method='ensemble',  # Options: 'kernel', 'gnn', 'ensemble'
    threshold=0.7,
    language='en'
)

# Detect plagiarism between two documents
report = detector.detect_plagiarism(doc1, doc2)

print(report.summary())
print(f"Is Plagiarism: {report.is_plagiarism}")
print(f"Similarity: {report.similarity_score:.2%}")

# Batch comparison
similarity_matrix = detector.batch_compare(documents, doc_ids)

# Find suspicious pairs
suspicious = detector.identify_suspicious_pairs(documents, threshold=0.7)
```

### ReportGenerator

Generate comprehensive reports.

```python
from graphplag.detection.report_generator import ReportGenerator

# Initialize
report_gen = ReportGenerator(output_dir="./reports")

# Generate HTML report
report_gen.save_report(report, filename="plagiarism_report.html")

# Plot similarity heatmap
report_gen.plot_similarity_heatmap(
    similarity_matrix,
    labels=doc_ids,
    output_file="heatmap.png"
)
```

## Data Models

### Document

Represents a parsed document with sentences and metadata.

**Attributes:**
- `text`: Raw document text
- `sentences`: List of Sentence objects
- `language`: LanguageCode enum
- `doc_id`: Document identifier
- `metadata`: Additional metadata

### Sentence

Represents a single sentence with linguistic features.

**Attributes:**
- `text`: Sentence text
- `index`: Sentence position in document
- `tokens`: List of tokens
- `lemmas`: List of lemmas
- `pos_tags`: Part-of-speech tags
- `dependencies`: List of dependency relations
- `embedding`: Sentence embedding vector

### DocumentGraph

Represents a document as a graph.

**Attributes:**
- `document`: Source Document object
- `nodes`: List of GraphNode objects
- `edges`: List of GraphEdge objects
- `graph_data`: NetworkX or PyG graph object
- `metadata`: Graph metadata

### PlagiarismReport

Comprehensive detection results.

**Attributes:**
- `document1`, `document2`: Compared documents
- `similarity_score`: Overall similarity [0, 1]
- `is_plagiarism`: Boolean detection result
- `threshold`: Detection threshold used
- `method`: Detection method used
- `matches`: List of specific matches
- `kernel_scores`: Individual kernel scores
- `gnn_score`: GNN similarity score
- `processing_time`: Time taken for detection

**Methods:**
- `summary()`: Generate human-readable summary

## Utilities

### GraphVisualizer

Visualize graphs and results.

```python
from graphplag.utils.visualization import GraphVisualizer

visualizer = GraphVisualizer()

# Visualize document graph
visualizer.visualize_graph(
    graph,
    highlight_nodes=[0, 1, 2],
    output_file="graph.png",
    interactive=True
)

# Visualize plagiarism alignment
visualizer.visualize_plagiarism_alignment(
    matches=report.matches,
    doc1_sentences=[s.text for s in doc1.sentences],
    doc2_sentences=[s.text for s in doc2.sentences],
    output_file="alignment.png"
)
```

### Metrics

Evaluation metrics for detection performance.

```python
from graphplag.utils.metrics import (
    evaluate_detection,
    find_optimal_threshold,
    print_evaluation_report
)

# Evaluate detection
metrics = evaluate_detection(y_true, y_pred, y_scores)
print_evaluation_report(metrics)

# Find optimal threshold
threshold, score = find_optimal_threshold(y_true, y_scores, metric='f1')
print(f"Optimal threshold: {threshold:.3f} (F1: {score:.3f})")
```

## Configuration

Use YAML configuration file:

```yaml
detection:
  method: "ensemble"
  threshold: 0.70
  
graph_builder:
  embedding_model: "paraphrase-multilingual-mpnet-base-v2"
  edge_strategy: "sequential"
  
similarity:
  kernels:
    types: ["wl", "rw", "sp"]
    normalize: true
```

Load configuration:

```python
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

detector = PlagiarismDetector(
    method=config['detection']['method'],
    threshold=config['detection']['threshold']
)
```

## Examples

See `examples/basic_usage.py` for complete examples.
