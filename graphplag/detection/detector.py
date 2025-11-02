"""
Plagiarism Detector Module

Main orchestrator for plagiarism detection using graph-based methods.
"""

from typing import List, Optional, Union
import time
import numpy as np

from graphplag.core.document_parser import DocumentParser
from graphplag.core.graph_builder import GraphBuilder
from graphplag.core.models import Document, DocumentGraph, PlagiarismReport, PlagiarismMatch
from graphplag.similarity.graph_kernels import GraphKernelSimilarity
from graphplag.similarity.gnn_models import GNNSimilarity


class PlagiarismDetector:
    """
    Main plagiarism detection system.
    
    Orchestrates the pipeline: parsing -> graph building -> similarity computation -> detection.
    """
    
    def __init__(
        self,
        method: str = "ensemble",
        threshold: float = 0.7,
        language: str = "en",
        embedding_model: str = "paraphrase-multilingual-mpnet-base-v2",
        kernel_types: Optional[List[str]] = None,
        gnn_model_path: Optional[str] = None
    ):
        """
        Initialize plagiarism detector.
        
        Args:
            method: Detection method ('kernel', 'gnn', or 'ensemble')
            threshold: Similarity threshold for plagiarism detection
            language: Language for document parsing
            embedding_model: Sentence embedding model
            kernel_types: Graph kernel types to use
            gnn_model_path: Path to trained GNN model
        """
        self.method = method.lower()
        self.threshold = threshold
        
        # Initialize components
        self.parser = DocumentParser(language=language)
        self.graph_builder = GraphBuilder(embedding_model=embedding_model)
        
        # Initialize similarity computers
        if method in ['kernel', 'ensemble']:
            self.kernel_similarity = GraphKernelSimilarity(
                kernel_types=kernel_types or ['wl', 'rw', 'sp']
            )
        else:
            self.kernel_similarity = None
        
        if method in ['gnn', 'ensemble']:
            self.gnn_similarity = GNNSimilarity(model_path=gnn_model_path)
        else:
            self.gnn_similarity = None
    
    def detect_plagiarism(
        self,
        doc1: Union[str, Document],
        doc2: Union[str, Document],
        doc1_id: Optional[str] = None,
        doc2_id: Optional[str] = None
    ) -> PlagiarismReport:
        """
        Detect plagiarism between two documents.
        
        Args:
            doc1: First document (text or Document object)
            doc2: Second document (text or Document object)
            doc1_id: Optional ID for first document
            doc2_id: Optional ID for second document
            
        Returns:
            PlagiarismReport with detection results
        """
        start_time = time.time()
        
        # Parse documents if needed
        if isinstance(doc1, str):
            document1 = self.parser.parse_document(doc1, doc_id=doc1_id)
        else:
            document1 = doc1
        
        if isinstance(doc2, str):
            document2 = self.parser.parse_document(doc2, doc_id=doc2_id)
        else:
            document2 = doc2
        
        # Build graphs
        graph1 = self.graph_builder.build_graph(document1)
        graph2 = self.graph_builder.build_graph(document2)
        
        # Compute similarity
        kernel_scores = {}
        gnn_score = None
        final_score = 0.0
        
        if self.method == 'kernel':
            similarity_result = self.kernel_similarity.compute_similarity(
                graph1, graph2, method='ensemble'
            )
            final_score = similarity_result.score
            kernel_scores = similarity_result.details.get('individual_scores', {})
        
        elif self.method == 'gnn':
            similarity_result = self.gnn_similarity.compute_similarity(graph1, graph2)
            final_score = similarity_result.score
            gnn_score = final_score
        
        elif self.method == 'ensemble':
            # Compute both kernel and GNN scores
            kernel_result = self.kernel_similarity.compute_similarity(
                graph1, graph2, method='ensemble'
            )
            kernel_score = kernel_result.score
            kernel_scores = kernel_result.details.get('individual_scores', {})
            
            gnn_result = self.gnn_similarity.compute_similarity(graph1, graph2)
            gnn_score = gnn_result.score
            
            # Ensemble: average of kernel and GNN scores
            final_score = (kernel_score + gnn_score) / 2.0
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Determine plagiarism
        is_plagiarism = final_score >= self.threshold
        
        # Identify specific matches (simplified version)
        matches = self._identify_matches(graph1, graph2, final_score) if is_plagiarism else []
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create report
        report = PlagiarismReport(
            document1=document1,
            document2=document2,
            similarity_score=final_score,
            is_plagiarism=is_plagiarism,
            threshold=self.threshold,
            method=self.method,
            matches=matches,
            kernel_scores=kernel_scores,
            gnn_score=gnn_score,
            processing_time=processing_time,
            metadata={
                "num_sentences_doc1": len(document1.sentences),
                "num_sentences_doc2": len(document2.sentences),
                "embedding_model": self.graph_builder.embedding_model_name
            }
        )
        
        return report
    
    def _identify_matches(
        self,
        graph1: DocumentGraph,
        graph2: DocumentGraph,
        overall_similarity: float
    ) -> List[PlagiarismMatch]:
        """
        Identify specific plagiarized segments between documents.
        
        This is a simplified implementation. A full implementation would use
        more sophisticated subgraph matching algorithms.
        
        Args:
            graph1: First document graph
            graph2: Second document graph
            overall_similarity: Overall similarity score
            
        Returns:
            List of plagiarism matches
        """
        matches = []
        
        # Compute sentence-level similarity using embeddings
        for i, node1 in enumerate(graph1.nodes):
            emb1 = node1.features
            
            for j, node2 in enumerate(graph2.nodes):
                emb2 = node2.features
                
                # Compute cosine similarity between sentence embeddings
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                
                # If similarity is high, consider it a match
                if similarity > 0.8:
                    match = PlagiarismMatch(
                        doc1_segment=(i, i+1),
                        doc2_segment=(j, j+1),
                        similarity=float(similarity),
                        method="sentence_embedding"
                    )
                    matches.append(match)
        
        return matches
    
    def batch_compare(
        self,
        documents: List[Union[str, Document]],
        doc_ids: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Compare multiple documents and create similarity matrix.
        
        Args:
            documents: List of documents to compare
            doc_ids: Optional list of document IDs
            
        Returns:
            Similarity matrix (n x n)
        """
        n = len(documents)
        
        # Parse documents if needed
        parsed_docs = []
        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                doc_id = doc_ids[i] if doc_ids else f"doc_{i}"
                parsed_docs.append(self.parser.parse_document(doc, doc_id=doc_id))
            else:
                parsed_docs.append(doc)
        
        # Build graphs
        graphs = [self.graph_builder.build_graph(doc) for doc in parsed_docs]
        
        # Compute similarity matrix
        similarity_matrix = np.eye(n)  # Initialize with 1s on diagonal
        
        for i in range(n):
            for j in range(i + 1, n):
                # Compute similarity
                if self.method == 'kernel':
                    result = self.kernel_similarity.compute_similarity(
                        graphs[i], graphs[j], method='ensemble'
                    )
                    score = result.score
                elif self.method == 'gnn':
                    result = self.gnn_similarity.compute_similarity(graphs[i], graphs[j])
                    score = result.score
                elif self.method == 'ensemble':
                    kernel_result = self.kernel_similarity.compute_similarity(
                        graphs[i], graphs[j], method='ensemble'
                    )
                    gnn_result = self.gnn_similarity.compute_similarity(graphs[i], graphs[j])
                    score = (kernel_result.score + gnn_result.score) / 2.0
                else:
                    score = 0.0
                
                similarity_matrix[i, j] = score
                similarity_matrix[j, i] = score
        
        return similarity_matrix
    
    def identify_suspicious_pairs(
        self,
        documents: List[Union[str, Document]],
        doc_ids: Optional[List[str]] = None,
        threshold: Optional[float] = None
    ) -> List[tuple]:
        """
        Identify pairs of documents with suspicious similarity.
        
        Args:
            documents: List of documents to analyze
            doc_ids: Optional list of document IDs
            threshold: Similarity threshold (uses detector's threshold if not provided)
            
        Returns:
            List of (doc1_idx, doc2_idx, similarity_score) tuples
        """
        threshold = threshold or self.threshold
        
        # Compute similarity matrix
        similarity_matrix = self.batch_compare(documents, doc_ids)
        
        # Find pairs above threshold
        suspicious_pairs = []
        n = len(documents)
        
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= threshold:
                    suspicious_pairs.append((i, j, similarity_matrix[i, j]))
        
        # Sort by similarity (highest first)
        suspicious_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return suspicious_pairs
    
    def __repr__(self) -> str:
        return f"PlagiarismDetector(method='{self.method}', threshold={self.threshold})"
