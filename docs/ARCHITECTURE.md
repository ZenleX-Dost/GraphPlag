# GraphPlag Architecture

## System Overview

GraphPlag is a semantic plagiarism detection system that uses graph representations and advanced machine learning techniques to identify content similarity beyond surface-level text matching.

```
┌─────────────┐
│   Raw Text  │
└──────┬──────┘
       │
       v
┌─────────────────────┐
│  Document Parser    │  ← spaCy, Stanza
│  - Tokenization     │
│  - POS Tagging      │
│  - Dependencies     │
└──────┬──────────────┘
       │
       v
┌─────────────────────┐
│   Graph Builder     │  ← SentenceTransformers
│  - Node Creation    │
│  - Edge Creation    │
│  - Feature Gen      │
└──────┬──────────────┘
       │
       v
┌─────────────────────────────────┐
│    Similarity Computation       │
├─────────────────┬───────────────┤
│  Graph Kernels  │  GNN Models   │
│  - WL Kernel    │  - GCN/GAT    │
│  - Random Walk  │  - Siamese    │
│  - Short. Path  │  - Pooling    │
└────────┬────────┴───────┬───────┘
         │                │
         v                v
    ┌────────────────────────┐
    │   Ensemble Scoring     │
    └────────┬───────────────┘
             │
             v
    ┌────────────────────────┐
    │  Plagiarism Detector   │
    │  - Threshold Check     │
    │  - Match Extraction    │
    │  - Report Generation   │
    └────────────────────────┘
```

## Component Architecture

### 1. Document Preprocessing Layer

**Purpose:** Transform raw text into structured linguistic representations.

**Components:**
- `DocumentParser`: Main parsing interface
- Language detection
- Sentence segmentation
- Tokenization and lemmatization
- POS tagging
- Dependency parsing

**Technologies:**
- spaCy: Primary NLP engine
- Stanza: Multilingual support
- langdetect: Language identification

### 2. Graph Construction Layer

**Purpose:** Convert documents into graph representations.

**Components:**
- `GraphBuilder`: Graph construction engine
- Node feature generation (sentence embeddings)
- Edge creation strategies
- Format conversion (NetworkX, PyTorch Geometric)

**Node Representation:**
- Each sentence becomes a node
- Features: Sentence embeddings (768-dim from transformer models)
- Metadata: Text, position, linguistic features

**Edge Strategies:**
1. **Sequential**: Connect nearby sentences (discourse flow)
2. **Dependency**: Connect sentences with shared entities/terms
3. **Hybrid**: Combine both approaches

**Technologies:**
- SentenceTransformers: Multilingual embeddings
- NetworkX: Graph manipulation
- PyTorch Geometric: GNN-ready format

### 3. Similarity Computation Layer

#### 3.1 Graph Kernel Methods

**Purpose:** Compute structural similarity using kernel functions.

**Implemented Kernels:**

1. **Weisfeiler-Lehman (WL) Kernel**
   - Iteratively refines node labels
   - Captures local graph structure
   - Fast computation
   - Good for structural similarity

2. **Random Walk Kernel**
   - Compares random walk sequences
   - Considers graph topology
   - Good for global structure

3. **Shortest Path Kernel**
   - Compares shortest path distributions
   - Captures distance relationships
   - Robust to local variations

**Ensemble Strategy:**
- Weighted combination of multiple kernels
- Default weights: WL(0.4), RW(0.3), SP(0.3)
- Normalized to [0, 1] range

#### 3.2 GNN-Based Methods

**Purpose:** Learn semantic similarity through deep learning.

**Architecture:**
```
Graph Input
    ↓
GCN/GAT Layers (3-5 layers)
    ↓
Batch Normalization + ReLU
    ↓
Dropout
    ↓
Global Pooling (Mean/Max/Attention)
    ↓
Graph Embedding (64-dim)
    ↓
Similarity Metric (Cosine/Learned)
    ↓
Similarity Score [0, 1]
```

**Siamese Architecture:**
- Shared encoder for both graphs
- Contrastive/triplet loss training
- Learns semantic similarity patterns

**GNN Types:**
- **GCN** (Graph Convolutional Network): Fast, simple aggregation
- **GAT** (Graph Attention Network): Attention-weighted neighbors

### 4. Detection Layer

**Purpose:** Orchestrate the full detection pipeline.

**Components:**
- `PlagiarismDetector`: Main orchestrator
- Threshold-based classification
- Segment-level match extraction
- Batch processing support

**Detection Modes:**
1. **Kernel-only**: Fast, interpretable
2. **GNN-only**: Learned semantic similarity
3. **Ensemble**: Best of both worlds

**Output:**
- Overall similarity score
- Binary plagiarism decision
- Individual kernel scores
- Sentence-level matches
- Processing metrics

### 5. Reporting Layer

**Purpose:** Generate comprehensive reports and visualizations.

**Components:**
- `ReportGenerator`: Report creation
- `GraphVisualizer`: Graph visualization
- HTML/PDF report generation
- Interactive visualizations

## Data Flow

### Single Document Comparison

```python
doc1, doc2 (text)
    ↓
DocumentParser.parse_document()
    ↓
Document objects (with sentences, dependencies)
    ↓
GraphBuilder.build_graph()
    ↓
DocumentGraph objects (nodes, edges, embeddings)
    ↓
GraphKernelSimilarity.compute_similarity()
GNNSimilarity.compute_similarity()
    ↓
SimilarityScore (ensemble)
    ↓
PlagiarismDetector.detect_plagiarism()
    ↓
PlagiarismReport (with matches, scores, metadata)
```

### Batch Document Comparison

```python
documents[] (texts)
    ↓
DocumentParser.parse_batch()
    ↓
Documents[]
    ↓
GraphBuilder.build_batch()
    ↓
DocumentGraphs[]
    ↓
Similarity.compute_batch()
    ↓
Similarity Matrix (n×n)
    ↓
PlagiarismDetector.identify_suspicious_pairs()
    ↓
Suspicious pairs list
```

## Key Design Decisions

### 1. Graph-Based Representation

**Rationale:**
- Captures semantic relationships beyond word-level matching
- Robust to paraphrasing and reordering
- Enables structural comparison
- Supports multilingual detection

### 2. Dual Similarity Approach

**Graph Kernels:**
- Pros: Fast, interpretable, no training needed
- Cons: May miss complex semantic patterns

**GNN Models:**
- Pros: Learn semantic patterns, flexible
- Cons: Require training data, slower

**Ensemble:**
- Combines strengths of both approaches
- Fallback if one method fails

### 3. Sentence-Level Granularity

**Rationale:**
- Fine-grained detection
- Enables segment matching
- Reasonable computational cost
- Natural linguistic unit

### 4. Multilingual Support

**Strategy:**
- Multilingual sentence embeddings
- Language-agnostic graph structure
- Cross-lingual detection capability

## Performance Considerations

### Computational Complexity

**Time Complexity:**
- Parsing: O(n) per document (n = text length)
- Graph building: O(s²) worst case (s = sentences)
- Kernel computation: O(s²) to O(s³) depending on kernel
- GNN inference: O(s × d) (d = embedding dimension)

**Space Complexity:**
- Document storage: O(s × d) (embeddings)
- Graph storage: O(s²) (dense edges)
- Similarity matrix: O(n²) (n = number of documents)

### Optimization Strategies

1. **Batch Processing:** Process multiple documents together
2. **Embedding Caching:** Cache sentence embeddings
3. **Sparse Graphs:** Limit edge creation (max_edge_distance)
4. **GPU Acceleration:** Use CUDA for GNN inference
5. **Approximate Kernels:** Trade accuracy for speed

### Scalability

**Current Limitations:**
- Memory-bound for large document collections
- Quadratic comparison complexity

**Future Improvements:**
- Approximate nearest neighbor search
- Hierarchical clustering
- Distributed processing
- Online detection

## Extension Points

1. **New Kernels:** Add custom graph kernels
2. **GNN Architectures:** Implement new GNN types
3. **Edge Strategies:** Custom edge creation logic
4. **Languages:** Add new language models
5. **Features:** Additional node/edge features
6. **Pooling Methods:** Custom graph pooling
7. **Similarity Metrics:** Custom distance functions

## Research Directions

1. **Cross-lingual Detection:** Improve multilingual performance
2. **Hierarchical Graphs:** Document, paragraph, sentence levels
3. **Temporal Analysis:** Track plagiarism over time
4. **Explainability:** Better interpretation of results
5. **Active Learning:** Iterative model improvement
6. **Zero-shot Detection:** No training data required
