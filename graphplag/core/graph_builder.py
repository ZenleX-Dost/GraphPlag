"""
Graph Builder Module

Transforms parsed documents into semantic graph representations.
Creates dependency graphs with sentence embeddings as node features.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer

from graphplag.core.models import (
    Document,
    DocumentGraph,
    GraphNode,
    GraphEdge,
    Sentence
)


class GraphBuilder:
    """
    Build graph representations of documents.
    
    Transforms documents into graphs where:
    - Nodes represent sentences with embedding features
    - Edges represent syntactic/semantic relationships
    """
    
    def __init__(
        self,
        embedding_model: str = "paraphrase-multilingual-mpnet-base-v2",
        edge_strategy: str = "sequential",
        max_edge_distance: int = 3
    ):
        """
        Initialize the GraphBuilder.
        
        Args:
            embedding_model: Name of sentence transformer model
            edge_strategy: Strategy for creating edges ('sequential', 'dependency', 'hybrid')
            max_edge_distance: Maximum distance for sequential edges
        """
        self.embedding_model_name = embedding_model
        self.edge_strategy = edge_strategy
        self.max_edge_distance = max_edge_distance
        
        # Load sentence transformer model
        print(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        
    def build_graph(
        self,
        document: Document,
        graph_type: str = "networkx"
    ) -> DocumentGraph:
        """
        Build a graph representation of a document.
        
        Args:
            document: Document object to convert
            graph_type: Type of graph ('networkx' or 'pyg')
            
        Returns:
            DocumentGraph object
        """
        # Generate sentence embeddings
        sentence_texts = [sent.text for sent in document.sentences]
        embeddings = self.encoder.encode(
            sentence_texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Store embeddings in sentences
        for sent, emb in zip(document.sentences, embeddings):
            sent.embedding = emb
        
        # Create graph nodes
        nodes = []
        for idx, sent in enumerate(document.sentences):
            node = GraphNode(
                node_id=idx,
                sentence=sent,
                features=sent.embedding,
                node_type="sentence"
            )
            nodes.append(node)
        
        # Create graph edges based on strategy
        edges = self._create_edges(document.sentences, self.edge_strategy)
        
        # Build NetworkX graph
        if graph_type == "networkx":
            graph_data = self._build_networkx_graph(nodes, edges)
        elif graph_type == "pyg":
            graph_data = self._build_pyg_graph(nodes, edges)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        
        # Create DocumentGraph object
        doc_graph = DocumentGraph(
            document=document,
            nodes=nodes,
            edges=edges,
            graph_data=graph_data,
            metadata={
                "embedding_model": self.embedding_model_name,
                "edge_strategy": self.edge_strategy,
                "num_nodes": len(nodes),
                "num_edges": len(edges)
            }
        )
        
        return doc_graph
    
    def _create_edges(
        self,
        sentences: List[Sentence],
        strategy: str
    ) -> List[GraphEdge]:
        """
        Create edges between sentence nodes based on strategy.
        
        Args:
            sentences: List of sentences
            strategy: Edge creation strategy
            
        Returns:
            List of GraphEdge objects
        """
        if strategy == "sequential":
            return self._create_sequential_edges(sentences)
        elif strategy == "dependency":
            return self._create_dependency_edges(sentences)
        elif strategy == "hybrid":
            seq_edges = self._create_sequential_edges(sentences)
            dep_edges = self._create_dependency_edges(sentences)
            return seq_edges + dep_edges
        else:
            raise ValueError(f"Unknown edge strategy: {strategy}")
    
    def _create_sequential_edges(self, sentences: List[Sentence]) -> List[GraphEdge]:
        """
        Create edges between sequential sentences.
        
        Creates edges between sentences within max_edge_distance.
        """
        edges = []
        n = len(sentences)
        
        for i in range(n):
            for j in range(i + 1, min(i + self.max_edge_distance + 1, n)):
                # Forward edge
                edge = GraphEdge(
                    source=i,
                    target=j,
                    edge_type="sequential",
                    weight=1.0 / (j - i),  # Closer sentences have higher weight
                    features={"distance": j - i}
                )
                edges.append(edge)
                
                # Backward edge for undirected graph
                reverse_edge = GraphEdge(
                    source=j,
                    target=i,
                    edge_type="sequential",
                    weight=1.0 / (j - i),
                    features={"distance": j - i}
                )
                edges.append(reverse_edge)
        
        return edges
    
    def _create_dependency_edges(self, sentences: List[Sentence]) -> List[GraphEdge]:
        """
        Create edges based on syntactic dependencies.
        
        For now, creates edges between sentences that share common entities
        or have high lexical overlap (simplified approach).
        """
        edges = []
        n = len(sentences)
        
        # Calculate lexical overlap between sentence pairs
        for i in range(n):
            tokens_i = set(sentences[i].lemmas)
            
            for j in range(i + 1, n):
                tokens_j = set(sentences[j].lemmas)
                
                # Calculate Jaccard similarity
                intersection = tokens_i & tokens_j
                union = tokens_i | tokens_j
                
                if len(union) > 0:
                    similarity = len(intersection) / len(union)
                    
                    # Only create edge if similarity is above threshold
                    if similarity > 0.1:
                        edge = GraphEdge(
                            source=i,
                            target=j,
                            edge_type="semantic",
                            weight=similarity,
                            features={"overlap": len(intersection)}
                        )
                        edges.append(edge)
                        
                        # Reverse edge
                        reverse_edge = GraphEdge(
                            source=j,
                            target=i,
                            edge_type="semantic",
                            weight=similarity,
                            features={"overlap": len(intersection)}
                        )
                        edges.append(reverse_edge)
        
        return edges
    
    def _build_networkx_graph(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge]
    ) -> nx.Graph:
        """
        Build a NetworkX graph.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            
        Returns:
            NetworkX Graph object
        """
        G = nx.Graph()
        
        # Add nodes with features
        for node in nodes:
            G.add_node(
                node.node_id,
                features=node.features,
                sentence_text=node.sentence.text,
                node_type=node.node_type
            )
        
        # Add edges with attributes
        for edge in edges:
            if not G.has_edge(edge.source, edge.target):
                G.add_edge(
                    edge.source,
                    edge.target,
                    edge_type=edge.edge_type,
                    weight=edge.weight,
                    features=edge.features
                )
        
        return G
    
    def _build_pyg_graph(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge]
    ) -> Any:
        """
        Build a PyTorch Geometric graph.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            
        Returns:
            PyG Data object
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError("PyTorch Geometric is required for PyG graphs. Install with: pip install torch-geometric")
        
        # Prepare node features
        x = torch.tensor(
            np.stack([node.features for node in nodes]),
            dtype=torch.float
        )
        
        # Prepare edge indices
        edge_index = torch.tensor(
            [[edge.source for edge in edges],
             [edge.target for edge in edges]],
            dtype=torch.long
        )
        
        # Prepare edge attributes
        edge_attr = torch.tensor(
            [edge.weight for edge in edges],
            dtype=torch.float
        ).unsqueeze(1)
        
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data
    
    def create_node_features(self, sentence: Sentence) -> np.ndarray:
        """
        Create feature vector for a sentence node.
        
        Args:
            sentence: Sentence object
            
        Returns:
            Feature vector as numpy array
        """
        # Generate embedding if not already present
        if sentence.embedding is None:
            sentence.embedding = self.encoder.encode(
                sentence.text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
        
        return sentence.embedding
    
    def extract_dependencies(self, sentence: Sentence) -> List[tuple]:
        """
        Extract dependency relations from a sentence.
        
        Args:
            sentence: Sentence with dependency information
            
        Returns:
            List of (head, relation, dependent) tuples
        """
        return sentence.dependencies
    
    def build_batch(
        self,
        documents: List[Document],
        graph_type: str = "networkx"
    ) -> List[DocumentGraph]:
        """
        Build graphs for multiple documents.
        
        Args:
            documents: List of Document objects
            graph_type: Type of graph to build
            
        Returns:
            List of DocumentGraph objects
        """
        return [self.build_graph(doc, graph_type) for doc in documents]
    
    def __repr__(self) -> str:
        return f"GraphBuilder(model='{self.embedding_model_name}', strategy='{self.edge_strategy}')"
