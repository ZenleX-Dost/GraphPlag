"""
PostgreSQL Client for Corpus Management

Handles document metadata and content storage.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class PostgresClient:
    """PostgreSQL database client for corpus documents."""
    
    def __init__(self, connection_url: str):
        """
        Initialize PostgreSQL client.
        
        Args:
            connection_url: PostgreSQL connection URL
        """
        self.connection_url = connection_url
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(self.connection_url)
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def _get_cursor(self):
        """Get database cursor."""
        if self.conn is None or self.conn.closed:
            self._connect()
        return self.conn.cursor(cursor_factory=RealDictCursor)
    
    def add_document(
        self,
        file_name: str,
        content: str,
        file_path: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add document to database.
        
        Args:
            file_name: Name of the file
            content: Document text content
            file_path: Optional file path
            metadata: Optional metadata dict
            
        Returns:
            doc_id of inserted document
        """
        # Generate file hash for deduplication
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Check if document already exists
        existing = self.get_document_by_hash(file_hash)
        if existing:
            logger.info(f"Document already exists with doc_id: {existing['doc_id']}")
            return existing['doc_id']
        
        # Get file size
        file_size = len(content.encode('utf-8'))
        
        # Determine content type from extension
        content_type = 'text/plain'
        if file_name.endswith('.pdf'):
            content_type = 'application/pdf'
        elif file_name.endswith('.docx'):
            content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        
        try:
            cursor = self._get_cursor()
            
            # Insert document
            cursor.execute("""
                INSERT INTO documents (file_name, file_path, file_size, file_hash, content_type, content, added_to_corpus)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING doc_id
            """, (file_name, file_path, file_size, file_hash, content_type, content, True))
            
            result = cursor.fetchone()
            doc_id = result['doc_id']
            
            # Add to corpus table if metadata provided
            if metadata:
                tags = metadata.get('tags', [])
                category = metadata.get('category', 'general')
                
                cursor.execute("""
                    INSERT INTO document_corpus (doc_id, tags, category)
                    VALUES (%s, %s, %s)
                """, (doc_id, tags, category))
            
            self.conn.commit()
            logger.info(f"Added document: {file_name} (doc_id: {doc_id})")
            return doc_id
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding document: {e}")
            raise
    
    def get_document(self, doc_id: int) -> Optional[Dict]:
        """
        Get document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document dict or None
        """
        try:
            cursor = self._get_cursor()
            cursor.execute("""
                SELECT d.*, dc.tags, dc.category
                FROM documents d
                LEFT JOIN document_corpus dc ON d.doc_id = dc.doc_id
                WHERE d.doc_id = %s
            """, (doc_id,))
            
            result = cursor.fetchone()
            return dict(result) if result else None
            
        except Exception as e:
            logger.error(f"Error getting document: {e}")
            return None
    
    def get_document_by_hash(self, file_hash: str) -> Optional[Dict]:
        """Get document by file hash."""
        try:
            cursor = self._get_cursor()
            cursor.execute("""
                SELECT doc_id, file_name, file_hash
                FROM documents
                WHERE file_hash = %s
            """, (file_hash,))
            
            result = cursor.fetchone()
            return dict(result) if result else None
            
        except Exception as e:
            logger.error(f"Error getting document by hash: {e}")
            return None
    
    def get_all_documents(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Get all corpus documents.
        
        Args:
            limit: Maximum number of documents
            offset: Offset for pagination
            
        Returns:
            List of document dicts
        """
        try:
            cursor = self._get_cursor()
            cursor.execute("""
                SELECT d.doc_id, d.file_name, d.file_size, d.created_at, 
                       dc.tags, dc.category, dc.added_at
                FROM documents d
                INNER JOIN document_corpus dc ON d.doc_id = dc.doc_id
                WHERE d.added_to_corpus = TRUE
                ORDER BY dc.added_at DESC
                LIMIT %s OFFSET %s
            """, (limit, offset))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return []
    
    def delete_document(self, doc_id: int) -> bool:
        """
        Delete document from corpus.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            cursor = self._get_cursor()
            
            # Delete from corpus table (cascade will handle documents table)
            cursor.execute("DELETE FROM document_corpus WHERE doc_id = %s", (doc_id,))
            cursor.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
            
            self.conn.commit()
            logger.info(f"Deleted document: {doc_id}")
            return True
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error deleting document: {e}")
            return False
    
    def get_corpus_stats(self) -> Dict:
        """
        Get corpus statistics.
        
        Returns:
            Dict with stats
        """
        try:
            cursor = self._get_cursor()
            
            # Get counts
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_documents,
                    SUM(file_size) as total_size,
                    AVG(file_size) as avg_size
                FROM documents
                WHERE added_to_corpus = TRUE
            """)
            
            stats = dict(cursor.fetchone())
            
            # Get category breakdown
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM document_corpus
                GROUP BY category
            """)
            
            categories = {row['category']: row['count'] for row in cursor.fetchall()}
            stats['categories'] = categories
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting corpus stats: {e}")
            return {}
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("PostgreSQL connection closed")
