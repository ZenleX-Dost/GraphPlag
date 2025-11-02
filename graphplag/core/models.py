"""
Data models for GraphPlag system.

Contains core data structures used throughout the plagiarism detection pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import numpy as np


class LanguageCode(Enum):
    """Supported language codes."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    ARABIC = "ar"
    UNKNOWN = "unknown"


@dataclass
class Sentence:
    """Represents a single sentence with linguistic analysis."""
    text: str
    index: int
    tokens: List[str] = field(default_factory=list)
    lemmas: List[str] = field(default_factory=list)
    pos_tags: List[str] = field(default_factory=list)
    dependencies: List[tuple] = field(default_factory=list)  # [(head_idx, dep_type, dependent_idx)]
    embedding: Optional[np.ndarray] = None
    
    def __repr__(self) -> str:
        return f"Sentence(index={self.index}, text='{self.text[:50]}...')"


@dataclass
class Document:
    """Represents a document with its sentences and metadata."""
    text: str
    sentences: List[Sentence] = field(default_factory=list)
    language: LanguageCode = LanguageCode.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    
    def __len__(self) -> int:
        return len(self.sentences)
    
    def __repr__(self) -> str:
        return f"Document(id={self.doc_id}, sentences={len(self.sentences)}, language={self.language.value})"


@dataclass
class Dependency:
    """Represents a syntactic dependency relation."""
    head: int  # Token index of head
    dependent: int  # Token index of dependent
    relation: str  # Dependency relation type (e.g., 'nsubj', 'dobj')
    weight: float = 1.0
    
    def __repr__(self) -> str:
        return f"Dependency({self.head} --{self.relation}-> {self.dependent})"


@dataclass
class GraphNode:
    """Represents a node in the document graph."""
    node_id: int
    sentence: Sentence
    features: np.ndarray
    node_type: str = "sentence"
    
    def __repr__(self) -> str:
        return f"GraphNode(id={self.node_id}, type={self.node_type})"


@dataclass
class GraphEdge:
    """Represents an edge in the document graph."""
    source: int
    target: int
    edge_type: str
    weight: float = 1.0
    features: Optional[Dict[str, Any]] = None
    
    def __repr__(self) -> str:
        return f"GraphEdge({self.source} --{self.edge_type}-> {self.target}, weight={self.weight:.3f})"


@dataclass
class DocumentGraph:
    """Represents a document as a graph structure."""
    document: Document
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    graph_data: Any = None  # NetworkX or PyG graph object
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"DocumentGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"


@dataclass
class SimilarityScore:
    """Represents similarity computation results."""
    score: float
    method: str
    confidence: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"SimilarityScore(score={self.score:.4f}, method={self.method})"


@dataclass
class PlagiarismMatch:
    """Represents a detected plagiarism match between document segments."""
    doc1_segment: tuple  # (start_idx, end_idx)
    doc2_segment: tuple  # (start_idx, end_idx)
    similarity: float
    method: str
    
    def __repr__(self) -> str:
        return f"PlagiarismMatch(sim={self.similarity:.3f}, segments={self.doc1_segment}->{self.doc2_segment})"


@dataclass
class PlagiarismReport:
    """Comprehensive plagiarism detection report."""
    document1: Document
    document2: Document
    similarity_score: float
    is_plagiarism: bool
    threshold: float
    method: str
    matches: List[PlagiarismMatch] = field(default_factory=list)
    kernel_scores: Dict[str, float] = field(default_factory=dict)
    gnn_score: Optional[float] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        status = "PLAGIARISM DETECTED" if self.is_plagiarism else "NO PLAGIARISM"
        return f"PlagiarismReport({status}, score={self.similarity_score:.3f}, matches={len(self.matches)})"
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"{'='*60}",
            f"PLAGIARISM DETECTION REPORT",
            f"{'='*60}",
            f"Document 1: {self.document1.doc_id or 'N/A'}",
            f"Document 2: {self.document2.doc_id or 'N/A'}",
            f"",
            f"Overall Similarity: {self.similarity_score:.2%}",
            f"Detection Method: {self.method}",
            f"Threshold: {self.threshold:.2%}",
            f"Result: {'PLAGIARISM DETECTED' if self.is_plagiarism else 'NO PLAGIARISM'}",
            f"",
            f"Matches Found: {len(self.matches)}",
        ]
        
        if self.kernel_scores:
            lines.append(f"\nKernel Scores:")
            for kernel, score in self.kernel_scores.items():
                lines.append(f"  - {kernel}: {score:.4f}")
        
        if self.gnn_score is not None:
            lines.append(f"\nGNN Score: {self.gnn_score:.4f}")
        
        lines.append(f"\nProcessing Time: {self.processing_time:.3f}s")
        lines.append(f"{'='*60}")
        
        return "\n".join(lines)
