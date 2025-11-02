"""Similarity computation modules initialization."""

from graphplag.similarity.graph_kernels import GraphKernelSimilarity
from graphplag.similarity.gnn_models import GNNSimilarity

__all__ = [
    "GraphKernelSimilarity",
    "GNNSimilarity",
]
