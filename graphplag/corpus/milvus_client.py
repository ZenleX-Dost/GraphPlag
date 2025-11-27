"""
Milvus Client for Vector Storage

Handles document embedding storage and similarity search.
"""

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MilvusClient:
    """Milvus vector database client for semantic search."""
    
    COLLECTION_NAME = "document_embeddings"
    EMBEDDING_DIM = 768  # paraphrase-multilingual-mpnet-base-v2 dimension
    
    def __init__(self, host: str = "localhost", port: str = "19530"):
        """
        Initialize Milvus client.
        
        Args:
            host: Milvus host
            port: Milvus port
        """
        self.host = host
        self.port = port
        self.collection = None
        self._connect()
        self._init_collection()
    
    def _connect(self):
        """Connect to Milvus server."""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _init_collection(self):
        """Initialize or load collection."""
        try:
            # Check if collection exists
            if utility.has_collection(self.COLLECTION_NAME):
                self.collection = Collection(self.COLLECTION_NAME)
                logger.info(f"Loaded existing collection: {self.COLLECTION_NAME}")
            else:
                # Create new collection
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="doc_id", dtype=DataType.INT64),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.EMBEDDING_DIM)
                ]
                
                schema = CollectionSchema(fields=fields, description="Document embeddings")
                self.collection = Collection(name=self.COLLECTION_NAME, schema=schema)
                
                # Create index for similarity search
                index_params = {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
                self.collection.create_index(field_name="embedding", index_params=index_params)
                
                logger.info(f"Created new collection:  {self.COLLECTION_NAME}")
            
            # Load collection into memory
            self.collection.load()
            
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise
    
    def add_embedding(self, doc_id: int, embedding: np.ndarray) -> bool:
        """
        Add document embedding to Milvus.
        
        Args:
            doc_id: Document ID
            embedding: Embedding vector
            
        Returns:
            True if successful
        """
        try:
            # Ensure embedding is correct shape
            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, -1)
            
            # Insert data
            data = [
                [doc_id],  # doc_id field
                embedding.tolist()  # embedding field
            ]
            
            self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"Added embedding for doc_id: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding embedding: {e}")
            return False
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, distance) tuples
        """
        try:
            # Ensure embedding is correct shape
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Search parameters
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            # Perform search
            results = self.collection.search(
                data=query_embedding.tolist(),
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["doc_id"]
            )
            
            # Extract results
            matches = []
            for hits in results:
                for hit in hits:
                    doc_id = hit.entity.get('doc_id')
                    distance = hit.distance
                    # Convert L2 distance to similarity score (0-1)
                    similarity = 1 / (1 + distance)
                    matches.append((doc_id, similarity))
            
            logger.info(f"Found {len(matches)} similar documents")
            return matches
            
        except Exception as e:
            logger.error(f"Error searching Milvus: {e}")
            return []
    
    def delete_embedding(self, doc_id: int) -> bool:
        """
        Delete document embedding.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful
        """
        try:
            expr = f"doc_id == {doc_id}"
            self.collection.delete(expr)
            self.collection.flush()
            
            logger.info(f"Deleted embedding for doc_id: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embedding: {e}")
            return False
    
    def get_count(self) -> int:
        """Get number of embeddings in collection."""
        try:
            return self.collection.num_entities
        except:
            return 0
    
    def close(self):
        """Close Milvus connection."""
        try:
            if self.collection:
                self.collection.release()
            connections.disconnect(alias="default")
            logger.info("Milvus connection closed")
        except Exception as e:
            logger.error(f"Error closing Milvus connection: {e}")
