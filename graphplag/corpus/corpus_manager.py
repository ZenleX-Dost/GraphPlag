"""
Corpus Manager - Main Interface for Document Corpus

Coordinates PostgreSQL, Milvus, and Elasticsearch for document storage and retrieval.
"""

from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

from graphplag.corpus.postgres_client import PostgresClient
from graphplag.corpus.milvus_client import MilvusClient
from graphplag.corpus.elasticsearch_client import ElasticsearchClient
from graphplag.embeddings.gnn_embedder import GNNEmbedder
from graphplag.utils.file_parser import FileParser

logger = logging.getLogger(__name__)


class CorpusManager:
    """
    Main manager for document corpus operations.
    
    Handles:
    - Document upload and storage
    - Embedding generation and indexing
    - Full-text and vector search
    - Hybrid search with result merging
    """
    
    def __init__(
        self,
        postgres_url: str,
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        elasticsearch_host: str = "localhost",
        elasticsearch_port: int = 9200
    ):
        """
        Initialize corpus manager.
        
        Args:
            postgres_url: PostgreSQL connection URL
            milvus_host: Milvus host
            milvus_port: Milvus port
            elasticsearch_host: Elasticsearch host
            elasticsearch_port: Elasticsearch port
        """
        logger.info("Initializing CorpusManager...")
        
        # Initialize clients
        self.pg = PostgresClient(postgres_url)
        self.milvus = MilvusClient(milvus_host, milvus_port)
        self.es = ElasticsearchClient(elasticsearch_host, elasticsearch_port)
        
        # Initialize utilities
        self.embedder = GNNEmbedder()
        self.file_parser = FileParser()
        
        logger.info("CorpusManager initialized successfully")
    
    def add_document(
        self,
        file_path: str,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add document to corpus.
        
        Process:
        1. Parse file → extract text
        2. Store in PostgreSQL
        3. Generate embedding → store in Milvus
        4. Index in Elasticsearch
        
        Args:
            file_path: Path to document file
            metadata: Optional metadata (tags, category)
            
        Returns:
            doc_id of added document
        """
        try:
            path = Path(file_path)
            file_name = path.name
            
            logger.info(f"Adding document to corpus: {file_name}")
            
            # Step 1: Parse file
            content = self.file_parser.parse_file(file_path)
            if not content or not content.strip():
                raise ValueError(f"No content extracted from {file_name}")
            
            logger.info(f"Extracted {len(content)} characters from {file_name}")
            
            # Step 2: Store in PostgreSQL
            doc_id = self.pg.add_document(
                file_name=file_name,
                content=content,
                file_path=str(path.absolute()),
                metadata=metadata
            )
            
            # Step 3: Generate and store embedding
            logger.info(f"Generating embedding for doc_id: {doc_id}")
            embedding = self.embedder.embed(content)
            self.milvus.add_embedding(doc_id, embedding)
            
            # Step 4: Index in Elasticsearch
            logger.info(f"Indexing in Elasticsearch: doc_id {doc_id}")
            self.es.index_document(doc_id, file_name, content)
            
            logger.info(f"✅ Successfully added document to corpus: {file_name} (doc_id: {doc_id})")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document to corpus: {e}")
            raise
    
    def search_corpus(
        self,
        query_text: str,
        top_k: int = 10,
        search_mode: str = "hybrid"
    ) -> List[Dict]:
        """
        Search corpus for similar documents.
        
        Args:
            query_text: Query text
            top_k: Number of results
            search_mode: 'vector', 'fulltext', or 'hybrid'
            
        Returns:
            List of match dicts with doc_id, score, and metadata
        """
        try:
            logger.info(f"Searching corpus with mode: {search_mode}")
            
            if search_mode == "vector":
                # Vector search only
                embedding = self.embedder.embed(query_text)
                vector_matches = self.milvus.search_similar(embedding, top_k)
                matches = self._format_matches(vector_matches)
                
            elif search_mode == "fulltext":
                # Full-text search only
                text_matches = self.es.search(query_text, top_k)
                matches = self._format_matches(text_matches)
                
            else:  # hybrid
                # Both vector and full-text search
                embedding = self.embedder.embed(query_text)
                vector_matches = self.milvus.search_similar(embedding, top_k * 2)
                text_matches = self.es.search(query_text, top_k * 2)
                
                # Merge and rank results
                matches = self._merge_results(vector_matches, text_matches, top_k)
            
            # Enrich with document metadata
            for match in matches:
                doc = self.pg.get_document(match['doc_id'])
                if doc:
                    match['file_name'] = doc['file_name']
                    match['tags'] = doc.get('tags', [])
                    match['category'] = doc.get('category', 'general')
                    match['content_preview'] = doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content']
            
            logger.info(f"Found {len(matches)} matching documents")
            return matches
            
        except Exception as e:
            logger.error(f"Error searching corpus: {e}")
            return []
    
    def _format_matches(self, raw_matches: List[Tuple[int, float]]) -> List[Dict]:
        """Format raw matches into dict format."""
        return [
            {'doc_id': doc_id, 'score': score}
            for doc_id, score in raw_matches
        ]
    
    def _merge_results(
        self,
        vector_matches: List[Tuple[int, float]],
        text_matches: List[Tuple[int, float]],
        top_k: int
    ) -> List[Dict]:
        """
        Merge vector and text search results.
        
        Uses weighted combination (60% vector + 40% text).
        """
        # Create score dictionaries
        vector_scores = {doc_id: score for doc_id, score in vector_matches}
        text_scores = {doc_id: score for doc_id, score in text_matches}
        
        # Normalize text scores to 0-1 range
        if text_scores:
            max_text_score = max(text_scores.values())
            if max_text_score > 0:
                text_scores = {doc_id: score / max_text_score for doc_id, score in text_scores.items()}
        
        # Merge with weighted combination
        merged_scores = {}
        all_doc_ids = set(vector_scores.keys()) | set(text_scores.keys())
        
        for doc_id in all_doc_ids:
            vec_score = vector_scores.get(doc_id, 0.0)
            txt_score = text_scores.get(doc_id, 0.0)
            
            # Weighted combination: 60% vector + 40% text
            combined_score = 0.6 * vec_score + 0.4 * txt_score
            merged_scores[doc_id] = combined_score
        
        # Sort by combined score
        sorted_matches = sorted(
            merged_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [
            {'doc_id': doc_id, 'score': score}
            for doc_id, score in sorted_matches
        ]
    
    def get_document(self, doc_id: int) -> Optional[Dict]:
        """Get document by ID."""
        return self.pg.get_document(doc_id)
    
    def get_all_documents(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get all corpus documents."""
        return self.pg.get_all_documents(limit, offset)
    
    def delete_document(self, doc_id: int) -> bool:
        """
        Delete document from corpus.
        
        Removes from PostgreSQL, Milvus, and Elasticsearch.
        """
        try:
            logger.info(f"Deleting document from corpus: {doc_id}")
            
            # Delete from all stores
            self.pg.delete_document(doc_id)
            self.milvus.delete_embedding(doc_id)
            self.es.delete_document(doc_id)
            
            logger.info(f"✅ Successfully deleted document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def get_corpus_stats(self) -> Dict:
        """
        Get corpus statistics.
        
        Returns:
            Dict with total documents, size, categories, etc.
        """
        try:
            pg_stats = self.pg.get_corpus_stats()
            
            stats = {
                'total_documents': pg_stats.get('total_documents', 0),
                'total_size_bytes': pg_stats.get('total_size', 0),
                'avg_size_bytes': pg_stats.get('avg_size', 0),
                'categories': pg_stats.get('categories', {}),
                'milvus_embeddings': self.milvus.get_count(),
                'elasticsearch_indexed': self.es.get_count()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting corpus stats: {e}")
            return {}
    
    def close(self):
        """Close all connections."""
        logger.info("Closing CorpusManager connections...")
        self.pg.close()
        self.milvus.close()
        self.es.close()
        logger.info("CorpusManager closed")
