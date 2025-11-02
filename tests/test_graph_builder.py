"""
Unit tests for GraphBuilder module.
"""

import pytest
import numpy as np
import networkx as nx

from graphplag.core.document_parser import DocumentParser
from graphplag.core.graph_builder import GraphBuilder
from graphplag.core.models import DocumentGraph, GraphNode, GraphEdge


@pytest.fixture
def parser():
    """Create a DocumentParser instance."""
    return DocumentParser(language='en')


@pytest.fixture
def builder():
    """Create a GraphBuilder instance."""
    return GraphBuilder(embedding_model="paraphrase-multilingual-mpnet-base-v2")


@pytest.fixture
def sample_document(parser):
    """Create a sample document for testing."""
    text = """
    Machine learning is a field of artificial intelligence. 
    It focuses on building systems that learn from data.
    These systems can improve their performance over time.
    """
    return parser.parse_document(text, doc_id="sample_doc")


def test_builder_initialization(builder):
    """Test builder initializes correctly."""
    assert builder.encoder is not None
    assert builder.edge_strategy in ['sequential', 'dependency', 'hybrid']


def test_build_graph(builder, sample_document):
    """Test building a graph from a document."""
    graph = builder.build_graph(sample_document)
    
    assert isinstance(graph, DocumentGraph)
    assert len(graph.nodes) == len(sample_document.sentences)
    assert len(graph.edges) > 0
    assert graph.graph_data is not None


def test_node_features(builder, sample_document):
    """Test that nodes have valid features."""
    graph = builder.build_graph(sample_document)
    
    for node in graph.nodes:
        assert isinstance(node, GraphNode)
        assert node.features is not None
        assert isinstance(node.features, np.ndarray)
        assert len(node.features) > 0  # Should have embedding dimensions


def test_edge_creation_sequential(sample_document):
    """Test sequential edge creation strategy."""
    builder = GraphBuilder(edge_strategy="sequential", max_edge_distance=2)
    graph = builder.build_graph(sample_document)
    
    # Should have edges between sequential sentences
    assert len(graph.edges) > 0
    
    # Check edge properties
    for edge in graph.edges:
        assert isinstance(edge, GraphEdge)
        assert edge.source >= 0
        assert edge.target >= 0
        assert edge.weight > 0


def test_edge_creation_dependency(sample_document):
    """Test dependency-based edge creation strategy."""
    builder = GraphBuilder(edge_strategy="dependency")
    graph = builder.build_graph(sample_document)
    
    # May or may not have edges depending on content overlap
    assert isinstance(graph.edges, list)


def test_edge_creation_hybrid(sample_document):
    """Test hybrid edge creation strategy."""
    builder = GraphBuilder(edge_strategy="hybrid")
    graph = builder.build_graph(sample_document)
    
    # Should have edges from both strategies
    assert len(graph.edges) > 0


def test_networkx_graph_format(builder, sample_document):
    """Test NetworkX graph format."""
    graph = builder.build_graph(sample_document, graph_type="networkx")
    
    assert isinstance(graph.graph_data, nx.Graph)
    assert len(graph.graph_data.nodes()) == len(sample_document.sentences)
    
    # Check node attributes
    for node_id in graph.graph_data.nodes():
        node_data = graph.graph_data.nodes[node_id]
        assert 'features' in node_data
        assert 'sentence_text' in node_data


def test_sentence_embeddings(builder, sample_document):
    """Test sentence embeddings are generated correctly."""
    graph = builder.build_graph(sample_document)
    
    # All sentences should have embeddings
    for sentence in sample_document.sentences:
        assert sentence.embedding is not None
        assert isinstance(sentence.embedding, np.ndarray)
    
    # Embeddings should have same dimension
    embedding_dims = [sent.embedding.shape[0] for sent in sample_document.sentences]
    assert len(set(embedding_dims)) == 1  # All same dimension


def test_batch_building(builder, parser):
    """Test building graphs for multiple documents."""
    texts = [
        "First document about AI.",
        "Second document about machine learning.",
        "Third document about data science."
    ]
    
    documents = [parser.parse_document(text, doc_id=f"doc{i}") 
                 for i, text in enumerate(texts)]
    
    graphs = builder.build_batch(documents)
    
    assert len(graphs) == 3
    for graph in graphs:
        assert isinstance(graph, DocumentGraph)
        assert len(graph.nodes) > 0


def test_graph_metadata(builder, sample_document):
    """Test graph metadata is populated."""
    graph = builder.build_graph(sample_document)
    
    assert 'embedding_model' in graph.metadata
    assert 'edge_strategy' in graph.metadata
    assert 'num_nodes' in graph.metadata
    assert 'num_edges' in graph.metadata
    
    assert graph.metadata['num_nodes'] == len(graph.nodes)
    assert graph.metadata['num_edges'] == len(graph.edges)


def test_empty_document(builder, parser):
    """Test handling of empty document."""
    document = parser.parse_document("", doc_id="empty")
    graph = builder.build_graph(document)
    
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
