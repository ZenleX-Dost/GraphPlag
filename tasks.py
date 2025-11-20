"""
Celery tasks for distributed document processing.
These tasks run in parallel on worker nodes.
"""

import os
import logging
from celery import Celery
from celery.signals import task_prerun, task_postrun
from datetime import datetime
import json

# Configure Celery
celery_app = Celery(
    'graphplag_tasks',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes hard limit
    task_soft_time_limit=25 * 60,  # 25 minutes soft limit
    worker_prefetch_multiplier=1,  # One task at a time
    worker_max_tasks_per_child=1000,
)

# Logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


# Signal handlers
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra):
    """Log task start."""
    job_id = args[0] if args else kwargs.get('job_id', 'unknown')
    logger.info(f"Task {task.name} [{task_id}] started for job {job_id}")


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, result=None, **extra):
    """Log task completion."""
    job_id = args[0] if args else 'unknown'
    logger.info(f"Task {task.name} [{task_id}] completed for job {job_id}")


# === Import here to avoid circular imports ===
# These are moved to task functions to prevent import issues


@celery_app.task(
    name='tasks.parse_document',
    bind=True,
    max_retries=3
)
def parse_document(self, job_id: str, file_path: str):
    """
    Parse document from file and extract text.
    Retries on failure.
    """
    try:
        from graphplag.parser.pdf_parser import PDFParser
        
        logger.info(f"[{job_id}] Parsing document: {file_path}")
        
        parser = PDFParser()
        text = parser.parse(file_path)
        
        if not text:
            raise ValueError("Document parsing returned empty text")
        
        logger.info(f"[{job_id}] Successfully parsed {len(text)} characters")
        
        return {
            'job_id': job_id,
            'text': text,
            'file_path': file_path,
            'char_count': len(text)
        }
    
    except Exception as exc:
        logger.error(f"[{job_id}] Parse error: {str(exc)}")
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)


@celery_app.task(
    name='tasks.detect_ai_content',
    bind=True
)
def detect_ai_content(self, job_id: str, text: str):
    """
    Detect if content is AI-generated.
    Uses multiple detection methods for robustness.
    """
    try:
        from graphplag.detection.ai_detector import AIDetector
        
        logger.info(f"[{job_id}] Detecting AI content")
        
        detector = AIDetector()
        ai_score = detector.detect_ai_content(text)
        
        logger.info(f"[{job_id}] AI detection score: {ai_score:.2%}")
        
        return {
            'job_id': job_id,
            'ai_score': float(ai_score),
            'is_ai': ai_score > 0.5,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as exc:
        logger.error(f"[{job_id}] AI detection error: {str(exc)}")
        return {
            'job_id': job_id,
            'ai_score': 0.0,
            'is_ai': False,
            'error': str(exc)
        }


@celery_app.task(
    name='tasks.build_graph',
    bind=True,
    max_retries=3
)
def build_graph(self, job_id: str, text: str):
    """
    Build AST/semantic graph from text.
    Used for structural similarity detection.
    """
    try:
        from graphplag.graph.graph_builder import GraphBuilder
        import networkx as nx
        
        logger.info(f"[{job_id}] Building document graph")
        
        builder = GraphBuilder()
        graph = builder.build(text)
        
        num_nodes = len(graph.nodes)
        num_edges = len(graph.edges)
        
        logger.info(f"[{job_id}] Graph built: {num_nodes} nodes, {num_edges} edges")
        
        # Serialize graph
        graph_data = nx.node_link_data(graph)
        
        return {
            'job_id': job_id,
            'graph': graph_data,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'graph_density': 2 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        }
    
    except Exception as exc:
        logger.error(f"[{job_id}] Graph building error: {str(exc)}")
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)


@celery_app.task(
    name='tasks.generate_embedding',
    bind=True
)
def generate_embedding(self, job_id: str, graph_data: dict, text: str):
    """
    Generate vector embedding from graph and text.
    Used for similarity search in vector database.
    """
    try:
        import networkx as nx
        import numpy as np
        
        logger.info(f"[{job_id}] Generating embedding")
        
        # Try to use GNN embedder if available
        try:
            from graphplag.embeddings.gnn_embedder import GNNEmbedder
            
            # Reconstruct graph
            graph = nx.node_link_graph(graph_data)
            
            embedder = GNNEmbedder()
            embedding = embedder.embed(graph)
            
            logger.info(f"[{job_id}] Generated GNN embedding: {len(embedding)} dimensions")
        
        except (ImportError, Exception) as e:
            # Fallback to sentence embeddings
            logger.warning(f"[{job_id}] GNN embedder unavailable, using fallback: {str(e)}")
            
            try:
                from sentence_transformers import SentenceTransformer
                
                model = SentenceTransformer('all-MiniLM-L6-v2')
                embedding = model.encode(text[:10000])  # Use first 10k chars
                
                logger.info(f"[{job_id}] Generated fallback embedding: {len(embedding)} dimensions")
            
            except Exception as fallback_e:
                logger.error(f"[{job_id}] Embedding generation failed: {str(fallback_e)}")
                # Return zero vector as last resort
                embedding = np.zeros(384).tolist()
        
        return {
            'job_id': job_id,
            'embedding': embedding if isinstance(embedding, list) else embedding.tolist(),
            'embedding_dim': len(embedding)
        }
    
    except Exception as exc:
        logger.error(f"[{job_id}] Embedding error: {str(exc)}")
        import numpy as np
        return {
            'job_id': job_id,
            'embedding': np.zeros(384).tolist(),
            'embedding_dim': 384,
            'error': str(exc)
        }


@celery_app.task(
    name='tasks.search_vector_db',
    bind=True
)
def search_vector_db(self, job_id: str, embedding: list, top_k: int = 100):
    """
    Search Milvus vector database for similar documents.
    Returns top_k results with similarity scores.
    """
    try:
        from pymilvus import connections, Collection
        
        logger.info(f"[{job_id}] Searching vector database (top {top_k})")
        
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=os.getenv('MILVUS_HOST', 'milvus'),
            port=int(os.getenv('MILVUS_PORT', '19530'))
        )
        
        collection = Collection("document_embeddings")
        
        # Search
        search_params = {"metric_type": "IP", "params": {"nprobe": 64}}
        results = collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["doc_id", "file_name", "embedding_model"]
        )
        
        # Format results
        matches = []
        if results and len(results) > 0:
            for hit in results[0]:
                matches.append({
                    'doc_id': str(hit.get('entity').get('doc_id')),
                    'file_name': hit.get('entity').get('file_name'),
                    'similarity_score': float(hit.distance),
                    'source': 'vector'
                })
        
        logger.info(f"[{job_id}] Vector search returned {len(matches)} matches")
        
        # Close connection
        connections.disconnect(alias="default")
        
        return {
            'job_id': job_id,
            'vector_results': matches,
            'num_results': len(matches)
        }
    
    except Exception as exc:
        logger.error(f"[{job_id}] Vector search error: {str(exc)}")
        return {
            'job_id': job_id,
            'vector_results': [],
            'num_results': 0,
            'error': str(exc)
        }


@celery_app.task(
    name='tasks.search_fulltext',
    bind=True
)
def search_fulltext(self, job_id: str, text: str, top_k: int = 100):
    """
    Search Elasticsearch for keyword-based matches.
    Returns top_k results with relevance scores.
    """
    try:
        from elasticsearch import Elasticsearch
        
        logger.info(f"[{job_id}] Searching full-text index (top {top_k})")
        
        es = Elasticsearch([os.getenv('ELASTICSEARCH_URL', 'http://elasticsearch:9200')])
        
        # Build query from text keywords
        keywords = text.split()[:20]  # Use first 20 words
        query = {
            "query": {
                "multi_match": {
                    "query": ' '.join(keywords),
                    "fields": ["content^2", "title^3"],
                    "fuzziness": "AUTO"
                }
            },
            "size": top_k
        }
        
        response = es.search(index="documents", body=query)
        
        # Format results
        matches = []
        for hit in response.get('hits', {}).get('hits', []):
            matches.append({
                'doc_id': hit.get('_source', {}).get('doc_id'),
                'file_name': hit.get('_source', {}).get('file_name'),
                'similarity_score': float(hit.get('_score', 0)),
                'source': 'fulltext'
            })
        
        logger.info(f"[{job_id}] Fulltext search returned {len(matches)} matches")
        
        return {
            'job_id': job_id,
            'fulltext_results': matches,
            'num_results': len(matches)
        }
    
    except Exception as exc:
        logger.error(f"[{job_id}] Fulltext search error: {str(exc)}")
        return {
            'job_id': job_id,
            'fulltext_results': [],
            'num_results': 0,
            'error': str(exc)
        }


@celery_app.task(
    name='tasks.compute_similarity',
    bind=True
)
def compute_similarity(self, job_id: str, text: str, doc_id: str):
    """
    Compute detailed similarity with a specific document.
    Uses multiple methods for accurate plagiarism detection.
    """
    try:
        from graphplag.detection.similarity_detector import SimilarityDetector
        
        logger.info(f"[{job_id}] Computing similarity with doc {doc_id}")
        
        detector = SimilarityDetector()
        
        # Fetch matched document (would query database in real scenario)
        # For now, return placeholder
        
        similarity_score = 0.0  # Placeholder
        
        return {
            'job_id': job_id,
            'doc_id': doc_id,
            'similarity_score': similarity_score,
            'plagiarism_percentage': similarity_score * 100
        }
    
    except Exception as exc:
        logger.error(f"[{job_id}] Similarity computation error: {str(exc)}")
        return {
            'job_id': job_id,
            'doc_id': doc_id,
            'similarity_score': 0.0,
            'error': str(exc)
        }


@celery_app.task(
    name='tasks.aggregate_results',
    bind=True
)
def aggregate_results(self, job_id: str, vector_results: list, fulltext_results: list, ai_score: float):
    """
    Aggregate results from vector and fulltext searches.
    Merge, deduplicate, and rank by combined score.
    """
    try:
        logger.info(f"[{job_id}] Aggregating {len(vector_results)} vector + {len(fulltext_results)} fulltext results")
        
        combined = {}
        
        # Add vector results (60% weight)
        for result in vector_results:
            doc_id = result['doc_id']
            if doc_id not in combined:
                combined[doc_id] = {
                    'doc_id': doc_id,
                    'file_name': result.get('file_name', 'unknown'),
                    'vector_score': result.get('similarity_score', 0),
                    'fulltext_score': 0,
                    'combined_score': 0
                }
            else:
                combined[doc_id]['vector_score'] = result.get('similarity_score', 0)
        
        # Add fulltext results (25% weight)
        for result in fulltext_results:
            doc_id = result['doc_id']
            score = result.get('similarity_score', 0) / 10.0
            
            if doc_id not in combined:
                combined[doc_id] = {
                    'doc_id': doc_id,
                    'file_name': result.get('file_name', 'unknown'),
                    'vector_score': 0,
                    'fulltext_score': score,
                    'combined_score': 0
                }
            else:
                combined[doc_id]['fulltext_score'] = score
        
        # Compute combined scores (60% vector + 25% fulltext + 15% baseline)
        for doc_id, data in combined.items():
            combined_score = (
                data['vector_score'] * 0.6 +
                data['fulltext_score'] * 0.25 +
                0.15  # Baseline for documents in database
            )
            data['combined_score'] = min(1.0, combined_score)
        
        # Sort by combined score
        ranked = sorted(
            combined.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        logger.info(f"[{job_id}] Aggregated to {len(ranked)} unique documents")
        
        return {
            'job_id': job_id,
            'results': ranked,
            'num_results': len(ranked),
            'ai_score': ai_score
        }
    
    except Exception as exc:
        logger.error(f"[{job_id}] Aggregation error: {str(exc)}")
        return {
            'job_id': job_id,
            'results': [],
            'num_results': 0,
            'error': str(exc)
        }


@celery_app.task(
    name='tasks.store_results',
    bind=True
)
def store_results(self, job_id: str, results: dict):
    """
    Store analysis results in PostgreSQL.
    Called after all processing is complete.
    """
    try:
        import asyncpg
        
        logger.info(f"[{job_id}] Storing results in database")
        
        async def store():
            pool = await asyncpg.create_pool(
                dsn=os.getenv('POSTGRES_URL', 'postgresql://user:pass@postgres:5432/graphplag')
            )
            
            async with pool.acquire() as conn:
                # Store analysis
                await conn.execute(
                    """
                    INSERT INTO analyses (job_id, ai_score, num_matches, created_at)
                    VALUES ($1, $2, $3, $4)
                    """,
                    job_id,
                    results.get('ai_score', 0),
                    len(results.get('results', [])),
                    datetime.now()
                )
                
                # Store matches
                for i, match in enumerate(results.get('results', [])[:100]):
                    await conn.execute(
                        """
                        INSERT INTO matches (job_id, rank, doc_id, file_name, similarity_score)
                        VALUES ($1, $2, $3, $4, $5)
                        """,
                        job_id,
                        i + 1,
                        match.get('doc_id'),
                        match.get('file_name'),
                        match.get('combined_score', 0)
                    )
            
            await pool.close()
        
        import asyncio
        asyncio.run(store())
        
        logger.info(f"[{job_id}] Results stored successfully")
        
        return {
            'job_id': job_id,
            'status': 'stored',
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as exc:
        logger.error(f"[{job_id}] Storage error: {str(exc)}")
        return {
            'job_id': job_id,
            'status': 'failed',
            'error': str(exc)
        }
