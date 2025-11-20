#!/usr/bin/env python3
"""
Setup Milvus vector database collections and indices.
Run after Milvus container is ready.
"""

import sys
import time
import logging
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, create_collection, utility

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_milvus():
    """Create Milvus collections and indices."""
    
    # Connect to Milvus
    max_retries = 5
    for attempt in range(max_retries):
        try:
            logger.info(f"Connecting to Milvus (attempt {attempt + 1}/{max_retries})...")
            connections.connect(
                alias="default",
                host="milvus",
                port=19530,
                timeout=10
            )
            logger.info("✓ Connected to Milvus")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Connection failed: {str(e)}. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                logger.error("Failed to connect to Milvus after multiple attempts")
                sys.exit(1)
    
    # Define schema for document embeddings
    fields = [
        FieldSchema(name="doc_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="file_hash", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="embedding_model", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="created_at", dtype=DataType.INT64),
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description="Document embeddings for plagiarism detection",
        enable_dynamic_field=True  # Allow additional fields
    )
    
    # Create collection
    collection_name = "document_embeddings"
    
    if utility.has_collection(collection_name):
        logger.info(f"Collection '{collection_name}' already exists, dropping...")
        utility.drop_collection(collection_name)
    
    logger.info(f"Creating collection '{collection_name}'...")
    collection = create_collection(
        name=collection_name,
        schema=schema,
        consistency_level="Strong"
    )
    logger.info(f"✓ Collection '{collection_name}' created")
    
    # Create indices for fast search
    logger.info("Creating indices...")
    
    # Vector index (HNSW - best for general purpose)
    index_params = {
        "index_type": "HNSW",
        "metric_type": "IP",  # Inner Product for cosine similarity
        "params": {
            "M": 30,  # Number of connections per element
            "efConstruction": 200,  # Size of dynamic candidate list
        }
    }
    
    collection.create_index(
        field_name="embedding",
        index_params=index_params,
        index_name="embedding_index"
    )
    logger.info("✓ Vector index (HNSW) created")
    
    # Scalar indices
    collection.create_index(
        field_name="doc_id",
        index_name="doc_id_index"
    )
    collection.create_index(
        field_name="file_hash",
        index_name="file_hash_index"
    )
    collection.create_index(
        field_name="created_at",
        index_name="created_at_index"
    )
    logger.info("✓ Scalar indices created")
    
    # Load collection into memory
    logger.info("Loading collection into memory...")
    collection.load()
    logger.info("✓ Collection loaded into memory")
    
    # Get collection info
    info = utility.get_collection_info(collection_name)
    logger.info(f"Collection info: {info}")
    
    # Create additional collection for document chunks (for large document handling)
    logger.info("\nCreating chunk collection for large documents...")
    
    chunk_fields = [
        FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="doc_id", dtype=DataType.INT64),
        FieldSchema(name="chunk_index", dtype=DataType.INT32),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="created_at", dtype=DataType.INT64),
    ]
    
    chunk_schema = CollectionSchema(
        fields=chunk_fields,
        description="Document chunks for handling large files",
        enable_dynamic_field=True
    )
    
    chunk_collection_name = "document_chunks"
    
    if utility.has_collection(chunk_collection_name):
        logger.info(f"Collection '{chunk_collection_name}' already exists, dropping...")
        utility.drop_collection(chunk_collection_name)
    
    chunk_collection = create_collection(
        name=chunk_collection_name,
        schema=chunk_schema,
        consistency_level="Strong"
    )
    logger.info(f"✓ Collection '{chunk_collection_name}' created")
    
    # Create index for chunk collection
    chunk_collection.create_index(
        field_name="embedding",
        index_params=index_params,
        index_name="chunk_embedding_index"
    )
    chunk_collection.create_index(
        field_name="doc_id",
        index_name="doc_id_index"
    )
    logger.info("✓ Chunk collection indices created")
    
    # Load chunk collection
    chunk_collection.load()
    logger.info("✓ Chunk collection loaded into memory")
    
    # List all collections
    logger.info("\nAvailable collections:")
    collections = utility.list_collections()
    for col_name in collections:
        logger.info(f"  - {col_name}")
    
    # Disconnect
    connections.disconnect(alias="default")
    logger.info("\n✓ Setup complete! Disconnected from Milvus")

if __name__ == "__main__":
    setup_milvus()
