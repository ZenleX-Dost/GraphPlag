"""
Graph Kernel Similarity Module

Implements graph kernel methods for computing similarity between document graphs.
Uses the GraKeL library for standard graph kernels.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx
from grakel import Graph as GrakelGraph
from grakel.kernels import (
    WeisfeilerLehman,
    RandomWalk,
    ShortestPath,
    VertexHistogram
)

from graphplag.core.models import DocumentGraph, SimilarityScore


class GraphKernelSimilarity:
    """
    Compute similarity between document graphs using graph kernels.
    
    Supports multiple kernel types:
    - Weisfeiler-Lehman (WL) kernel
    - Random Walk kernel
    - Shortest Path kernel
    - Vertex Histogram kernel
    """
    
    def __init__(
        self,
        kernel_types: Optional[List[str]] = None,
        normalize: bool = True,
        wl_iterations: int = 5
    ):
        """
        Initialize graph kernel similarity computer.
        
        Args:
            kernel_types: List of kernel types to use (default: ['wl', 'rw', 'sp'])
            normalize: Whether to normalize kernel values
            wl_iterations: Number of iterations for WL kernel
        """
        self.kernel_types = kernel_types or ['wl', 'rw', 'sp']
        self.normalize = normalize
        self.wl_iterations = wl_iterations
        
        # Initialize kernels
        self.kernels = self._initialize_kernels()
    
    def _initialize_kernels(self) -> Dict:
        """Initialize kernel objects."""
        kernels = {}
        
        if 'wl' in self.kernel_types:
            kernels['wl'] = WeisfeilerLehman(
                n_iter=self.wl_iterations,
                normalize=self.normalize,
                base_graph_kernel=VertexHistogram
            )
        
        if 'rw' in self.kernel_types:
            kernels['rw'] = RandomWalk(
                normalize=self.normalize,
                lambda_decay=0.1
            )
        
        if 'sp' in self.kernel_types:
            kernels['sp'] = ShortestPath(
                normalize=self.normalize
            )
        
        return kernels
    
    def compute_similarity(
        self,
        graph1: DocumentGraph,
        graph2: DocumentGraph,
        method: str = 'ensemble'
    ) -> SimilarityScore:
        """
        Compute similarity between two document graphs.
        
        Args:
            graph1: First document graph
            graph2: Second document graph
            method: Kernel method ('wl', 'rw', 'sp', or 'ensemble')
            
        Returns:
            SimilarityScore object
        """
        if method == 'ensemble':
            return self.ensemble_kernel_score(graph1, graph2)
        elif method in self.kernels:
            score = self._compute_single_kernel(
                graph1.graph_data,
                graph2.graph_data,
                method
            )
            return SimilarityScore(
                score=score,
                method=f"kernel_{method}",
                details={"kernel_type": method}
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compute_wl_kernel(
        self,
        graph1: nx.Graph,
        graph2: nx.Graph
    ) -> float:
        """
        Compute Weisfeiler-Lehman kernel similarity.
        
        Args:
            graph1: First NetworkX graph
            graph2: Second NetworkX graph
            
        Returns:
            Similarity score
        """
        return self._compute_single_kernel(graph1, graph2, 'wl')
    
    def compute_random_walk_kernel(
        self,
        graph1: nx.Graph,
        graph2: nx.Graph
    ) -> float:
        """
        Compute Random Walk kernel similarity.
        
        Args:
            graph1: First NetworkX graph
            graph2: Second NetworkX graph
            
        Returns:
            Similarity score
        """
        return self._compute_single_kernel(graph1, graph2, 'rw')
    
    def compute_shortest_path_kernel(
        self,
        graph1: nx.Graph,
        graph2: nx.Graph
    ) -> float:
        """
        Compute Shortest Path kernel similarity.
        
        Args:
            graph1: First NetworkX graph
            graph2: Second NetworkX graph
            
        Returns:
            Similarity score
        """
        return self._compute_single_kernel(graph1, graph2, 'sp')
    
    def _compute_single_kernel(
        self,
        graph1: nx.Graph,
        graph2: nx.Graph,
        kernel_type: str
    ) -> float:
        """
        Compute a single kernel similarity.
        
        Args:
            graph1: First graph
            graph2: Second graph
            kernel_type: Type of kernel
            
        Returns:
            Similarity score
        """
        # Convert NetworkX graphs to GraKeL format
        grakel_graphs = self._convert_to_grakel([graph1, graph2])
        
        # Compute kernel matrix
        kernel = self.kernels[kernel_type]
        K = kernel.fit_transform(grakel_graphs)
        
        # Extract similarity score (off-diagonal element)
        similarity = K[0, 1]
        
        # Normalize to [0, 1] range if not already normalized
        if not self.normalize and K[0, 0] > 0 and K[1, 1] > 0:
            similarity = similarity / np.sqrt(K[0, 0] * K[1, 1])
        
        # Ensure score is in valid range
        similarity = np.clip(similarity, 0.0, 1.0)
        
        return float(similarity)
    
    def ensemble_kernel_score(
        self,
        graph1: DocumentGraph,
        graph2: DocumentGraph,
        weights: Optional[Dict[str, float]] = None
    ) -> SimilarityScore:
        """
        Compute ensemble score using multiple kernels.
        
        Args:
            graph1: First document graph
            graph2: Second document graph
            weights: Optional weights for each kernel (default: equal weights)
            
        Returns:
            SimilarityScore with ensemble result
        """
        if weights is None:
            # Equal weights for all kernels
            weights = {k: 1.0 / len(self.kernels) for k in self.kernels.keys()}
        
        scores = {}
        weighted_sum = 0.0
        
        for kernel_type in self.kernels.keys():
            score = self._compute_single_kernel(
                graph1.graph_data,
                graph2.graph_data,
                kernel_type
            )
            scores[kernel_type] = score
            weighted_sum += weights.get(kernel_type, 0.0) * score
        
        return SimilarityScore(
            score=weighted_sum,
            method="kernel_ensemble",
            details={
                "individual_scores": scores,
                "weights": weights
            }
        )
    
    def _convert_to_grakel(
        self,
        nx_graphs: List[nx.Graph]
    ) -> List[GrakelGraph]:
        """
        Convert NetworkX graphs to GraKeL format.
        
        Args:
            nx_graphs: List of NetworkX graphs
            
        Returns:
            List of GraKeL Graph objects
        """
        grakel_graphs = []
        
        for G in nx_graphs:
            # Create node labels (use node IDs as labels)
            node_labels = {node: node for node in G.nodes()}
            
            # Create edge list
            edges = list(G.edges())
            
            # Create edge labels/weights
            edge_labels = {}
            for u, v in edges:
                edge_data = G.get_edge_data(u, v)
                weight = edge_data.get('weight', 1.0) if edge_data else 1.0
                edge_labels[(u, v)] = weight
            
            # Create GraKeL graph
            grakel_graph = GrakelGraph(
                edges,
                node_labels=node_labels,
                edge_labels=edge_labels
            )
            grakel_graphs.append(grakel_graph)
        
        return grakel_graphs
    
    def compute_batch(
        self,
        graphs: List[DocumentGraph],
        method: str = 'ensemble'
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix for multiple graphs.
        
        Args:
            graphs: List of document graphs
            method: Kernel method to use
            
        Returns:
            Similarity matrix (n x n)
        """
        n = len(graphs)
        similarity_matrix = np.zeros((n, n))
        
        # Compute pairwise similarities
        for i in range(n):
            similarity_matrix[i, i] = 1.0  # Self-similarity
            
            for j in range(i + 1, n):
                score = self.compute_similarity(
                    graphs[i],
                    graphs[j],
                    method=method
                ).score
                
                similarity_matrix[i, j] = score
                similarity_matrix[j, i] = score  # Symmetric
        
        return similarity_matrix
    
    def __repr__(self) -> str:
        return f"GraphKernelSimilarity(kernels={list(self.kernels.keys())}, normalize={self.normalize})"
