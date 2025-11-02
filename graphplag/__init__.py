"""
GraphPlag: Semantic Graph-Based Plagiarism Detection System

A research-oriented plagiarism detection system using graph representations
and semantic similarity computation through Graph Kernels and GNNs.
"""

__version__ = "0.1.0"
__author__ = "GraphPlag Team"

from graphplag.core.document_parser import DocumentParser
from graphplag.core.graph_builder import GraphBuilder
from graphplag.detection.detector import PlagiarismDetector

__all__ = [
    "DocumentParser",
    "GraphBuilder",
    "PlagiarismDetector",
]
