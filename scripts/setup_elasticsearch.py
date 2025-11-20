#!/usr/bin/env python3
"""
Setup Elasticsearch indices and mappings.
Run after Elasticsearch container is ready.
"""

import sys
import time
import logging
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_elasticsearch():
    """Create Elasticsearch indices and mappings."""
    
    # Connect to Elasticsearch
    max_retries = 5
    es = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Connecting to Elasticsearch (attempt {attempt + 1}/{max_retries})...")
            es = Elasticsearch(
                hosts=["http://elasticsearch:9200"],
                timeout=10,
                retry_on_timeout=True,
                max_retries=3
            )
            
            # Test connection
            info = es.info()
            logger.info(f"✓ Connected to Elasticsearch {info['version']['number']}")
            break
        except ConnectionError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Connection failed: {str(e)}. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                logger.error("Failed to connect to Elasticsearch after multiple attempts")
                sys.exit(1)
    
    # Define mappings for documents index
    documents_mapping = {
        "settings": {
            "number_of_shards": 5,
            "number_of_replicas": 1,
            "index": {
                "refresh_interval": "1s",
                "analysis": {
                    "analyzer": {
                        "text_analyzer": {
                            "type": "standard",
                            "stopwords": "_english_"
                        }
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "doc_id": {
                    "type": "keyword"
                },
                "file_name": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword"
                        }
                    }
                },
                "content": {
                    "type": "text",
                    "analyzer": "text_analyzer",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "title": {
                    "type": "text",
                    "analyzer": "text_analyzer",
                    "boost": 2.0
                },
                "author": {
                    "type": "keyword"
                },
                "created_at": {
                    "type": "date"
                },
                "updated_at": {
                    "type": "date"
                },
                "file_size": {
                    "type": "integer"
                },
                "file_hash": {
                    "type": "keyword"
                },
                "ai_score": {
                    "type": "float"
                },
                "tags": {
                    "type": "keyword"
                }
            }
        }
    }
    
    # Create documents index
    index_name = "documents"
    if es.indices.exists(index=index_name):
        logger.info(f"Index '{index_name}' already exists, deleting...")
        es.indices.delete(index=index_name)
    
    logger.info(f"Creating index '{index_name}'...")
    es.indices.create(index=index_name, body=documents_mapping)
    logger.info(f"✓ Index '{index_name}' created")
    
    # Create plagiarism_matches index
    matches_mapping = {
        "settings": {
            "number_of_shards": 3,
            "number_of_replicas": 1
        },
        "mappings": {
            "properties": {
                "job_id": {
                    "type": "keyword"
                },
                "analysis_id": {
                    "type": "keyword"
                },
                "source_doc_id": {
                    "type": "keyword"
                },
                "matched_doc_id": {
                    "type": "keyword"
                },
                "matched_file_name": {
                    "type": "text"
                },
                "similarity_score": {
                    "type": "float"
                },
                "match_type": {
                    "type": "keyword"
                },
                "created_at": {
                    "type": "date"
                }
            }
        }
    }
    
    matches_index = "plagiarism_matches"
    if es.indices.exists(index=matches_index):
        logger.info(f"Index '{matches_index}' already exists, deleting...")
        es.indices.delete(index=matches_index)
    
    logger.info(f"Creating index '{matches_index}'...")
    es.indices.create(index=matches_index, body=matches_mapping)
    logger.info(f"✓ Index '{matches_index}' created")
    
    # Create analysis_logs index
    logs_mapping = {
        "settings": {
            "number_of_shards": 2,
            "number_of_replicas": 1
        },
        "mappings": {
            "properties": {
                "job_id": {
                    "type": "keyword"
                },
                "level": {
                    "type": "keyword"
                },
                "message": {
                    "type": "text"
                },
                "timestamp": {
                    "type": "date"
                },
                "component": {
                    "type": "keyword"
                }
            }
        }
    }
    
    logs_index = "analysis_logs"
    if es.indices.exists(index=logs_index):
        logger.info(f"Index '{logs_index}' already exists, deleting...")
        es.indices.delete(index=logs_index)
    
    logger.info(f"Creating index '{logs_index}'...")
    es.indices.create(index=logs_index, body=logs_mapping)
    logger.info(f"✓ Index '{logs_index}' created")
    
    # Set up index aliases
    logger.info("\nSetting up aliases...")
    
    # Create write alias for documents (for seamless index rotation)
    es.indices.put_alias(index=index_name, name="documents_write")
    logger.info("✓ Alias 'documents_write' created")
    
    # Create read alias
    es.indices.put_alias(index=index_name, name="documents_read")
    logger.info("✓ Alias 'documents_read' created")
    
    # Get cluster health
    logger.info("\nCluster health:")
    health = es.cluster.health()
    logger.info(f"  Status: {health['status']}")
    logger.info(f"  Active shards: {health['active_shards']}")
    logger.info(f"  Relocating shards: {health['relocating_shards']}")
    logger.info(f"  Initializing shards: {health['initializing_shards']}")
    logger.info(f"  Unassigned shards: {health['unassigned_shards']}")
    
    # List indices
    logger.info("\nAvailable indices:")
    indices = es.indices.get_alias(index="*")
    for idx_name in sorted(indices.keys()):
        idx_info = es.indices.stats(index=idx_name)
        doc_count = idx_info['indices'][idx_name]['total']['docs']['count']
        logger.info(f"  - {idx_name} ({doc_count} documents)")
    
    logger.info("\n✓ Setup complete!")

if __name__ == "__main__":
    setup_elasticsearch()
