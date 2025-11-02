"""
Unit tests for similarity computation modules.
"""

import pytest
import numpy as np

from graphplag.core.document_parser import DocumentParser
from graphplag.core.graph_builder import GraphBuilder
from graphplag.similarity.graph_kernels import GraphKernelSimilarity
from graphplag.core.models import SimilarityScore


@pytest.fixture
def parser():
    return DocumentParser(language='en')


@pytest.fixture
def builder():
    return GraphBuilder()


@pytest.fixture
def kernel_similarity():
    return GraphKernelSimilarity(kernel_types=['wl', 'sp'])


@pytest.fixture
def similar_documents(parser):
    """Create two similar documents."""
    doc1 = parser.parse_document(
        "Machine learning is a field of AI. It uses algorithms to learn from data.",
        doc_id="doc1"
    )
    doc2 = parser.parse_document(
        "Machine learning is an AI field. It employs algorithms to learn from data.",
        doc_id="doc2"
    )
    return doc1, doc2


@pytest.fixture
def dissimilar_documents(parser):
    """Create two dissimilar documents."""
    doc1 = parser.parse_document(
        "The weather is sunny today. I enjoy outdoor activities.",
        doc_id="doc1"
    )
    doc2 = parser.parse_document(
        "Quantum computing uses quantum mechanics. It processes information differently.",
        doc_id="doc2"
    )
    return doc1, doc2


def test_kernel_similarity_initialization(kernel_similarity):
    """Test kernel similarity initializes correctly."""
    assert kernel_similarity is not None
    assert len(kernel_similarity.kernels) > 0


def test_compute_similarity_similar_docs(builder, kernel_similarity, similar_documents):
    """Test similarity computation for similar documents."""
    doc1, doc2 = similar_documents
    
    graph1 = builder.build_graph(doc1)
    graph2 = builder.build_graph(doc2)
    
    result = kernel_similarity.compute_similarity(graph1, graph2, method='ensemble')
    
    assert isinstance(result, SimilarityScore)
    assert 0.0 <= result.score <= 1.0
    assert result.score > 0.5  # Should be relatively high for similar docs


def test_compute_similarity_dissimilar_docs(builder, kernel_similarity, dissimilar_documents):
    """Test similarity computation for dissimilar documents."""
    doc1, doc2 = dissimilar_documents
    
    graph1 = builder.build_graph(doc1)
    graph2 = builder.build_graph(doc2)
    
    result = kernel_similarity.compute_similarity(graph1, graph2, method='ensemble')
    
    assert isinstance(result, SimilarityScore)
    assert 0.0 <= result.score <= 1.0
    # Dissimilar documents should have lower score, but not necessarily < 0.5


def test_wl_kernel(builder, kernel_similarity, similar_documents):
    """Test Weisfeiler-Lehman kernel."""
    doc1, doc2 = similar_documents
    
    graph1 = builder.build_graph(doc1)
    graph2 = builder.build_graph(doc2)
    
    score = kernel_similarity.compute_wl_kernel(graph1.graph_data, graph2.graph_data)
    
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_shortest_path_kernel(builder, kernel_similarity, similar_documents):
    """Test Shortest Path kernel."""
    doc1, doc2 = similar_documents
    
    graph1 = builder.build_graph(doc1)
    graph2 = builder.build_graph(doc2)
    
    score = kernel_similarity.compute_shortest_path_kernel(
        graph1.graph_data, graph2.graph_data
    )
    
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_ensemble_score(builder, kernel_similarity, similar_documents):
    """Test ensemble kernel score."""
    doc1, doc2 = similar_documents
    
    graph1 = builder.build_graph(doc1)
    graph2 = builder.build_graph(doc2)
    
    result = kernel_similarity.ensemble_kernel_score(graph1, graph2)
    
    assert isinstance(result, SimilarityScore)
    assert result.method == "kernel_ensemble"
    assert 'individual_scores' in result.details
    assert len(result.details['individual_scores']) > 0


def test_batch_similarity(builder, kernel_similarity, parser):
    """Test batch similarity computation."""
    texts = [
        "First document about AI.",
        "Second document about AI.",
        "Third document about cooking."
    ]
    
    documents = [parser.parse_document(text, doc_id=f"doc{i}") 
                 for i, text in enumerate(texts)]
    graphs = [builder.build_graph(doc) for doc in documents]
    
    similarity_matrix = kernel_similarity.compute_batch(graphs, method='ensemble')
    
    assert isinstance(similarity_matrix, np.ndarray)
    assert similarity_matrix.shape == (3, 3)
    
    # Diagonal should be 1.0 (self-similarity)
    assert np.allclose(np.diag(similarity_matrix), 1.0)
    
    # Matrix should be symmetric
    assert np.allclose(similarity_matrix, similarity_matrix.T)
    
    # All values should be in [0, 1]
    assert np.all(similarity_matrix >= 0.0)
    assert np.all(similarity_matrix <= 1.0)


def test_self_similarity(builder, kernel_similarity, parser):
    """Test that self-similarity is high."""
    doc = parser.parse_document(
        "This is a test document for self-similarity.",
        doc_id="doc1"
    )
    
    graph = builder.build_graph(doc)
    
    result = kernel_similarity.compute_similarity(graph, graph, method='ensemble')
    
    # Self-similarity should be very high (close to 1.0)
    assert result.score > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
