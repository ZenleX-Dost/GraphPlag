from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import json
import time
from typing import Optional
import logging
from datetime import datetime
import uuid

# Task queue
from celery import Celery
from celery.result import AsyncResult

# Databases
from pymilvus import Collection, connections
import elasticsearch
import asyncpg

# Utils
from graphplag.parser.pdf_parser import PDFParser
from graphplag.graph.graph_builder import GraphBuilder
from graphplag.detection.similarity_detector import SimilarityDetector
from graphplag.detection.ai_detector import AIDetector

# Setup logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="GraphPlag Scalable API",
    description="Production-grade plagiarism detection system for massive document databases",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Celery setup
celery_app = Celery(
    'graphplag',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
)

# Initialize components
parser = PDFParser()
graph_builder = GraphBuilder()
similarity_detector = SimilarityDetector()
ai_detector = AIDetector()

# Database connections (lazy-initialized)
_milvus_conn = None
_es_client = None
_pg_pool = None


async def get_milvus():
    global _milvus_conn
    if _milvus_conn is None:
        connections.connect(
            alias="default",
            host=os.getenv('MILVUS_HOST', 'milvus'),
            port=int(os.getenv('MILVUS_PORT', '19530'))
        )
        _milvus_conn = Collection("document_embeddings")
    return _milvus_conn


async def get_elasticsearch():
    global _es_client
    if _es_client is None:
        _es_client = elasticsearch.Elasticsearch(
            hosts=[os.getenv('ELASTICSEARCH_URL', 'http://elasticsearch:9200')]
        )
    return _es_client


async def get_postgres_pool():
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = await asyncpg.create_pool(
            dsn=os.getenv('POSTGRES_URL', 'postgresql://user:pass@postgres:5432/graphplag'),
            min_size=5,
            max_size=20
        )
    return _pg_pool


# === Models ===
class AnalysisRequest:
    def __init__(self, file_path: str, file_name: str, job_id: str):
        self.file_path = file_path
        self.file_name = file_name
        self.job_id = job_id
        self.timestamp = datetime.now()


class AnalysisResult:
    def __init__(self, job_id: str, status: str, data: dict = None, error: str = None):
        self.job_id = job_id
        self.status = status
        self.data = data or {}
        self.error = error
        self.timestamp = datetime.now()

    def to_dict(self):
        return {
            'job_id': self.job_id,
            'status': self.status,
            'data': self.data,
            'error': self.error,
            'timestamp': self.timestamp.isoformat()
        }


# === Celery Tasks ===
@celery_app.task(name='tasks.process_and_search')
def process_and_search(job_id: str, file_path: str, file_name: str):
    """
    Celery task for processing document and searching against database.
    This runs in parallel on distributed workers.
    """
    try:
        logger.info(f"[{job_id}] Starting processing: {file_name}")
        
        # Step 1: Parse document
        try:
            text = parser.parse(file_path)
            logger.info(f"[{job_id}] Parsed document: {len(text)} chars")
        except Exception as e:
            logger.error(f"[{job_id}] Parse error: {str(e)}")
            return {
                'job_id': job_id,
                'status': 'failed',
                'error': f'Failed to parse document: {str(e)}'
            }
        
        # Step 2: Detect AI content
        ai_score = ai_detector.detect_ai_content(text)
        logger.info(f"[{job_id}] AI detection score: {ai_score}")
        
        # Step 3: Build graph representation
        try:
            graph = graph_builder.build(text)
            logger.info(f"[{job_id}] Built graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        except Exception as e:
            logger.error(f"[{job_id}] Graph building error: {str(e)}")
            return {
                'job_id': job_id,
                'status': 'failed',
                'error': f'Failed to build graph: {str(e)}'
            }
        
        # Step 4: Generate embedding from graph
        try:
            from graphplag.embeddings.gnn_embedder import GNNEmbedder
            embedder = GNNEmbedder()
            embedding = embedder.embed(graph)
            logger.info(f"[{job_id}] Generated embedding: dim={len(embedding)}")
        except Exception as e:
            logger.error(f"[{job_id}] Embedding error: {str(e)}")
            embedding = None
        
        # Step 5: Vector search in Milvus (parallel with fulltext)
        vector_results = []
        if embedding is not None:
            try:
                # Import here to avoid connection issues during task startup
                import asyncio
                loop = asyncio.get_event_loop()
                
                milvus = loop.run_until_complete(get_milvus())
                
                # Search top 100 similar documents
                search_params = {"metric_type": "IP", "params": {"nprobe": 64}}
                vector_results = milvus.search(
                    data=[embedding],
                    anns_field="embedding",
                    param=search_params,
                    limit=100,
                    output_fields=["doc_id", "file_name", "similarity"]
                )
                
                logger.info(f"[{job_id}] Vector search returned {len(vector_results)} results")
            except Exception as e:
                logger.error(f"[{job_id}] Vector search error: {str(e)}")
        
        # Step 6: Full-text search in Elasticsearch (parallel with vector)
        fulltext_results = []
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            es = loop.run_until_complete(get_elasticsearch())
            
            # Search for key phrases
            query = {
                "query": {
                    "multi_match": {
                        "query": text[:500],  # Use first 500 chars as query
                        "fields": ["content", "title^2"]
                    }
                },
                "size": 100
            }
            
            es_response = es.search(index="documents", body=query)
            fulltext_results = es_response.get('hits', {}).get('hits', [])
            
            logger.info(f"[{job_id}] Fulltext search returned {len(fulltext_results)} results")
        except Exception as e:
            logger.error(f"[{job_id}] Fulltext search error: {str(e)}")
        
        # Step 7: Aggregate and rank results
        ranked_results = aggregate_results(
            vector_results, 
            fulltext_results,
            similarity_detector,
            text
        )
        
        logger.info(f"[{job_id}] Aggregated to {len(ranked_results)} unique results")
        
        # Step 8: Store results in PostgreSQL
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            pg_pool = loop.run_until_complete(get_postgres_pool())
            
            async def store_results():
                async with pg_pool.acquire() as conn:
                    # Store analysis record
                    await conn.execute(
                        """
                        INSERT INTO analyses (job_id, file_name, ai_score, num_matches, created_at)
                        VALUES ($1, $2, $3, $4, $5)
                        """,
                        job_id, file_name, ai_score, len(ranked_results), datetime.now()
                    )
                    
                    # Store matches
                    for i, result in enumerate(ranked_results):
                        await conn.execute(
                            """
                            INSERT INTO matches (job_id, rank, matched_file, similarity_score, ai_score)
                            VALUES ($1, $2, $3, $4, $5)
                            """,
                            job_id, i+1, result.get('file_name', 'unknown'),
                            result.get('similarity_score', 0), result.get('ai_score', 0)
                        )
            
            loop.run_until_complete(store_results())
            logger.info(f"[{job_id}] Stored results in PostgreSQL")
        except Exception as e:
            logger.error(f"[{job_id}] Database storage error: {str(e)}")
        
        return {
            'job_id': job_id,
            'status': 'completed',
            'file_name': file_name,
            'ai_score': ai_score,
            'num_matches': len(ranked_results),
            'top_matches': ranked_results[:10],
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"[{job_id}] Unexpected error: {str(e)}", exc_info=True)
        return {
            'job_id': job_id,
            'status': 'failed',
            'error': f'Unexpected error: {str(e)}'
        }


def aggregate_results(vector_results, fulltext_results, similarity_detector, text):
    """
    Aggregate vector search and fulltext search results.
    Merge, deduplicate, and rank by combined score.
    """
    combined = {}
    
    # Add vector search results (60% weight)
    if vector_results:
        for result in vector_results[0]:  # Results in nested list
            doc_id = str(result.get('doc_id'))
            if doc_id not in combined:
                combined[doc_id] = {
                    'doc_id': doc_id,
                    'file_name': result.get('file_name', 'unknown'),
                    'vector_score': float(result.get('distance', 0)),
                    'fulltext_score': 0,
                    'similarity_score': 0
                }
            else:
                combined[doc_id]['vector_score'] = float(result.get('distance', 0))
    
    # Add fulltext results (25% weight)
    for result in fulltext_results:
        doc_id = str(result.get('_source', {}).get('doc_id', result.get('_id')))
        score = result.get('_score', 0) / 10.0  # Normalize ES score
        
        if doc_id not in combined:
            combined[doc_id] = {
                'doc_id': doc_id,
                'file_name': result.get('_source', {}).get('file_name', 'unknown'),
                'vector_score': 0,
                'fulltext_score': score,
                'similarity_score': 0
            }
        else:
            combined[doc_id]['fulltext_score'] = score
    
    # Compute final similarity scores (15% weight)
    for doc_id, data in combined.items():
        try:
            # In a real system, fetch the matched document and compute similarity
            # For now, use weighted combination of vector and fulltext scores
            similarity_score = (
                data['vector_score'] * 0.6 +
                data['fulltext_score'] * 0.25 +
                0.15  # Baseline for documents found in DB
            )
            data['similarity_score'] = min(1.0, similarity_score)
            data['ai_score'] = 0.0  # Would fetch from matched document's analysis
        except Exception as e:
            logger.error(f"Error computing similarity for {doc_id}: {str(e)}")
            data['similarity_score'] = 0.0
            data['ai_score'] = 0.0
    
    # Sort by combined score
    ranked = sorted(
        combined.values(),
        key=lambda x: x['similarity_score'],
        reverse=True
    )
    
    return ranked


# === API Endpoints ===

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }


@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Upload a document and analyze for plagiarism against database.
    Returns job_id for tracking progress.
    
    Response:
    {
        "job_id": "uuid",
        "status": "queued",
        "message": "Analysis started. Check /status/{job_id} for progress.",
        "estimated_time": 30
    }
    """
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.txt', '.md'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: {allowed_extensions}"
            )
        
        # Create temp file
        job_id = str(uuid.uuid4())
        upload_dir = "/app/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, f"{job_id}_{file.filename}")
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        logger.info(f"[{job_id}] File saved: {file_path} ({len(contents)} bytes)")
        
        # Queue task
        task = process_and_search.delay(job_id, file_path, file.filename)
        
        return JSONResponse({
            'job_id': job_id,
            'task_id': task.id,
            'status': 'queued',
            'message': 'Analysis started. Check /status/{job_id} for progress.',
            'estimated_time': 30,
            'timestamp': datetime.now().isoformat()
        }, status_code=202)
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Get current status of analysis job.
    
    Response:
    {
        "job_id": "uuid",
        "status": "processing|completed|failed",
        "progress": 45,
        "message": "Searching database...",
        "eta_seconds": 15
    }
    """
    try:
        pg_pool = await get_postgres_pool()
        
        async with pg_pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                SELECT job_id, status, progress, message, eta_seconds
                FROM job_status
                WHERE job_id = $1
                """,
                job_id
            )
        
        if not record:
            return JSONResponse({
                'job_id': job_id,
                'status': 'not_found',
                'message': 'Job not found'
            }, status_code=404)
        
        return {
            'job_id': record['job_id'],
            'status': record['status'],
            'progress': record['progress'],
            'message': record['message'],
            'eta_seconds': record['eta_seconds'],
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{job_id}")
async def stream_results(job_id: str):
    """
    Stream results of analysis as Server-Sent Events.
    Provides live updates as processing completes.
    
    Stream format:
    data: {"status": "processing", "current_step": "Parsing PDF...", "progress": 10}
    data: {"status": "processing", "current_step": "Building graph...", "progress": 30}
    data: {"status": "completed", "matches": [...], "ai_score": 0.35}
    """
    async def event_generator():
        try:
            pg_pool = await get_postgres_pool()
            last_check = 0
            
            while True:
                async with pg_pool.acquire() as conn:
                    record = await conn.fetchrow(
                        """
                        SELECT job_id, status, data
                        FROM analysis_results
                        WHERE job_id = $1
                        """,
                        job_id
                    )
                
                if record:
                    result_data = json.loads(record['data']) if isinstance(record['data'], str) else record['data']
                    yield f"data: {json.dumps(result_data)}\n\n"
                    
                    if record['status'] in ['completed', 'failed']:
                        break
                
                # Check every 2 seconds
                await asyncio.sleep(2)
                last_check += 2
                
                # Timeout after 5 minutes
                if last_check > 300:
                    yield f"data: {json.dumps({'error': 'Analysis timeout'})}\n\n"
                    break
        
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/database-stats")
async def database_stats():
    """
    Get current database statistics.
    
    Response:
    {
        "total_documents": 10500000,
        "total_embeddings": 10500000,
        "avg_ai_score": 0.15,
        "milvus_size_gb": 250,
        "elasticsearch_indices": 12,
        "postgres_tables": 8
    }
    """
    try:
        milvus = await get_milvus()
        es = await get_elasticsearch()
        pg_pool = await get_postgres_pool()
        
        # Get Milvus stats
        milvus_count = milvus.num_entities
        
        # Get Elasticsearch stats
        es_stats = es.indices.stats()
        es_indices = len(es_stats.get('indices', {}))
        
        # Get PostgreSQL stats
        async with pg_pool.acquire() as conn:
            doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
            avg_ai = await conn.fetchval("SELECT AVG(ai_score) FROM analyses WHERE ai_score IS NOT NULL")
        
        return {
            'total_documents': doc_count,
            'total_embeddings': milvus_count,
            'avg_ai_score': float(avg_ai) if avg_ai else 0,
            'elasticsearch_indices': es_indices,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Startup/Shutdown
@app.on_event("startup")
async def startup():
    """Initialize database connections on startup."""
    logger.info("Starting up API server...")
    try:
        await get_milvus()
        await get_elasticsearch()
        await get_postgres_pool()
        logger.info("All databases connected successfully")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """Close connections on shutdown."""
    logger.info("Shutting down API server...")
    global _pg_pool
    if _pg_pool:
        await _pg_pool.close()


if __name__ == "__main__":
    uvicorn.run(
        "app_scalable:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        reload=False
    )
