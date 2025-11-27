"""
Elasticsearch Client for Full-Text Search

Handles document indexing and text-based search.
"""

from elasticsearch import Elasticsearch
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """Elasticsearch client for full-text document search."""
    
    INDEX_NAME = "documents"
    
    def __init__(self, host: str = "localhost", port: int = 9200):
        """
        Initialize Elasticsearch client.
        
        Args:
            host: Elasticsearch host
            port: Elasticsearch port
        """
        self.es = Elasticsearch([f"http://{host}:{port}"])
        self._init_index()
    
    def _init_index(self):
        """Initialize index with mapping."""
        try:
            # Check if index exists
            if not self.es.indices.exists(index=self.INDEX_NAME):
                # Create index with mapping
                mapping = {
                    "mappings": {
                        "properties": {
                            "doc_id": {"type": "integer"},
                            "file_name": {"type": "keyword"},
                            "content": {
                                "type": "text",
                                "analyzer": "standard",
                                "similarity": "BM25"
                            },
                            "timestamp": {"type": "date"}
                        }
                    }
                }
                
                self.es.indices.create(index=self.INDEX_NAME, body=mapping)
                logger.info(f"Created Elasticsearch index: {self.INDEX_NAME}")
            else:
                logger.info(f"Elasticsearch index already exists: {self.INDEX_NAME}")
                
        except Exception as e:
            logger.error(f"Error initializing Elasticsearch index: {e}")
            raise
    
    def index_document(
        self,
        doc_id: int,
        file_name: str,
        content: str
    ) -> bool:
        """
        Index a document.
        
        Args:
            doc_id: Document ID
            file_name: File name
            content: Document content
            
        Returns:
            True if successful
        """
        try:
            doc = {
                "doc_id": doc_id,
                "file_name": file_name,
                "content": content,
                "timestamp": "now"
            }
            
            self.es.index(
                index=self.INDEX_NAME,
                id=str(doc_id),
                body=doc
            )
            
            # Refresh index to make document available immediately
            self.es.indices.refresh(index=self.INDEX_NAME)
            
            logger.info(f"Indexed document: {file_name} (doc_id: {doc_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            return False
    
    def search(
        self,
        query_text: str,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Search for documents.
        
        Args:
            query_text: Query text
            top_k: Number of results
            
        Returns:
            List of (doc_id, score) tuples
        """
        try:
            # Build search query
            query = {
                "query": {
                    "match": {
                        "content": {
                            "query": query_text,
                            "fuzziness": "AUTO"
                        }
                    }
                },
                "size": top_k
            }
            
            # Execute search
            response = self.es.search(index=self.INDEX_NAME, body=query)
            
            # Extract results
            matches = []
            for hit in response['hits']['hits']:
                doc_id = hit['_source']['doc_id']
                score = hit['_score']
                matches.append((doc_id, score))
            
            logger.info(f"Found {len(matches)} matching documents")
            return matches
            
        except Exception as e:
            logger.error(f"Error searching Elasticsearch: {e}")
            return []
    
    def delete_document(self, doc_id: int) -> bool:
        """
        Delete a document from index.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful
        """
        try:
            self.es.delete(index=self.INDEX_NAME, id=str(doc_id))
            self.es.indices.refresh(index=self.INDEX_NAME)
            
            logger.info(f"Deleted document from index: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def get_count(self) -> int:
        """Get number of documents in index."""
        try:
            response = self.es.count(index=self.INDEX_NAME)
            return response['count']
        except:
            return 0
    
    def close(self):
        """Close Elasticsearch connection."""
        try:
            self.es.close()
            logger.info("Elasticsearch connection closed")
        except Exception as e:
            logger.error(f"Error closing Elasticsearch connection: {e}")
