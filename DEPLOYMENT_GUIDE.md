# GraphPlag Scalable Deployment Guide

## Overview
This guide walks you through deploying GraphPlag as a production-scale system capable of comparing documents against a massive database (millions of documents).

## Architecture Components

### 1. **Tier 1: API Gateway (FastAPI)**
- **File**: `app_scalable.py`
- **Port**: 8000
- **Endpoints**:
  - `POST /analyze` - Upload document, get job ID
  - `GET /status/{job_id}` - Check processing status
  - `GET /results/{job_id}` - Stream results (Server-Sent Events)
  - `GET /health` - Health check
  - `GET /database-stats` - Database statistics

### 2. **Tier 2: Job Queue (Celery + Redis)**
- **Files**: `tasks.py`, `docker-compose-scalable.yml`
- **Components**:
  - Redis: Message broker and cache
  - 4 Celery workers: Parallel task processing
  - Flower: Task monitoring dashboard (port 5555)

**Task Pipeline**:
1. `parse_document` - Extract text from PDF/DOCX
2. `detect_ai_content` - Check if AI-generated
3. `build_graph` - Build semantic graph
4. `generate_embedding` - Create vector representation
5. `search_vector_db` - Find similar documents via vectors
6. `search_fulltext` - Find similar documents via keywords
7. `aggregate_results` - Merge and rank results
8. `store_results` - Save to database

### 3. **Tier 3: Databases**

#### **Vector Database (Milvus)**
- Stores document embeddings
- Sub-50ms search on 10M documents
- HNSW index for fast similarity search

#### **Full-Text Search (Elasticsearch)**
- Keyword-based plagiarism detection
- Scoring and relevance ranking

#### **Relational Database (PostgreSQL + pgvector)**
- Metadata: documents, analyses, matches
- Embedding linkage
- Job tracking and results storage

#### **Cache Layer (Redis)**
- Recent results caching
- Task queue management
- Session storage

---

## Quick Start

### Prerequisites
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- 16+ GB RAM
- 50+ GB free disk space (for databases)
- Python 3.10+

### Step 1: Install Requirements

```powershell
# Navigate to project directory
cd c:\Users\Amine EL-Hend\Documents\GitHub\GraphPlag

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-scalable.txt

# Verify installations
pip list | findstr celery, fastapi, pymilvus
```

### Step 2: Start Docker Compose

```powershell
# Build images
docker-compose -f docker-compose-scalable.yml build

# Start services
docker-compose -f docker-compose-scalable.yml up -d

# Verify services
docker-compose -f docker-compose-scalable.yml ps

# Check logs
docker-compose -f docker-compose-scalable.yml logs -f api
```

Expected startup time: 2-3 minutes

### Step 3: Initialize Databases

```powershell
# Wait for PostgreSQL to be ready (~30 seconds)
Start-Sleep -Seconds 30

# Initialize schema
docker exec graphplag_postgres psql -U user -d graphplag -f /docker-entrypoint-initdb.d/init.sql

# Create Milvus collection
python scripts/setup_milvus.py

# Create Elasticsearch indices
python scripts/setup_elasticsearch.py
```

### Step 4: Test the System

```powershell
# Test API health
curl http://localhost:8000/health

# Test document upload
$file = @{file=@'c:\path\to\test.pdf'}
$response = Invoke-WebRequest -Uri "http://localhost:8000/analyze" -Method Post -Form $file -ContentType "multipart/form-data"
$jobId = $response.Content | ConvertFrom-Json | Select-Object -ExpandProperty job_id

# Check status
curl "http://localhost:8000/status/$jobId"

# Stream results
curl "http://localhost:8000/results/$jobId"
```

### Step 5: Monitor Processing

- **Flower (Celery Dashboard)**: http://localhost:5555
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

---

## API Usage Examples

### Example 1: Upload and Analyze Document

```python
import requests
import json
import time

# Upload document
with open('suspect_document.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/analyze', files=files)

result = response.json()
job_id = result['job_id']
print(f"Analysis started. Job ID: {job_id}")

# Poll status
while True:
    status_resp = requests.get(f'http://localhost:8000/status/{job_id}')
    status = status_resp.json()
    
    if status['status'] in ['completed', 'failed']:
        break
    
    print(f"Progress: {status['progress']}% - {status['message']}")
    time.sleep(2)

# Get results
results_resp = requests.get(f'http://localhost:8000/results/{job_id}')
results = results_resp.json()

print(f"\nAnalysis Results:")
print(f"AI Score: {results.get('ai_score', 0):.2%}")
print(f"Matches Found: {results.get('num_matches', 0)}")
print(f"Top 5 Matches:")
for i, match in enumerate(results.get('top_matches', [])[:5], 1):
    print(f"  {i}. {match['file_name']} - {match['similarity_score']:.2%}")
```

### Example 2: Stream Results in Real-Time

```python
import requests
import json

response = requests.get(
    f'http://localhost:8000/results/{job_id}',
    stream=True
)

for line in response.iter_lines():
    if line:
        data = json.loads(line.decode('utf-8').replace('data: ', ''))
        print(f"Status: {data.get('status')} - {data.get('current_step')}")
        if data.get('status') == 'completed':
            print(f"Final Results: {data}")
            break
```

### Example 3: Get Database Statistics

```python
response = requests.get('http://localhost:8000/database-stats')
stats = response.json()

print(f"Total Documents: {stats['total_documents']:,}")
print(f"Total Embeddings: {stats['total_embeddings']:,}")
print(f"Average AI Score: {stats['avg_ai_score']:.2%}")
print(f"Elasticsearch Indices: {stats['elasticsearch_indices']}")
```

---

## Batch Processing with Spark

For initial database population or bulk reprocessing:

### Setup Spark Job

```python
# spark_jobs/batch_indexing.py
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec
import os

spark = SparkSession.builder \
    .appName("GraphPlagBatchIndexing") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Read documents
docs = spark.read.json("/data/documents/*.json")

# Process in parallel
processed = docs.repartition(16).map(lambda row: {
    'doc_id': row['id'],
    'embedding': generate_embedding(row['content']),
    'ai_score': detect_ai(row['content'])
})

# Store to Milvus and Elasticsearch
# ... implementation ...

spark.stop()
```

### Submit Spark Job

```powershell
docker exec graphplag_spark-master spark-submit \
  --master spark://spark-master:7077 \
  --deploy-mode cluster \
  --executor-memory 2g \
  --num-executors 3 \
  /app/spark_jobs/batch_indexing.py
```

---

## Configuration

### Environment Variables

Create `.env` file in project root:

```env
# FastAPI
LOG_LEVEL=INFO
WORKERS=4

# Redis
REDIS_URL=redis://redis:6379/0
REDIS_CACHE_TTL=86400

# PostgreSQL
POSTGRES_URL=postgresql://user:pass@postgres:5432/graphplag
POSTGRES_POOL_SIZE=20

# Milvus
MILVUS_HOST=milvus
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=document_embeddings

# Elasticsearch
ELASTICSEARCH_URL=http://elasticsearch:9200
ES_INDEX_SHARDS=5
ES_INDEX_REPLICAS=1

# Celery
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1
CELERY_TASK_TIMEOUT=1800

# Search Parameters
VECTOR_SEARCH_TOP_K=100
FULLTEXT_SEARCH_TOP_K=100
SIMILARITY_THRESHOLD=0.3
```

### Database Scaling

For production (millions of documents):

```sql
-- Add partitioning to large tables
ALTER TABLE matches PARTITION BY RANGE (YEAR(created_at)) (
    PARTITION p_2024 VALUES LESS THAN (2025),
    PARTITION p_2025 VALUES LESS THAN (2026)
);

-- Create additional indices
CREATE INDEX idx_fast_search ON matches(combined_similarity_score DESC, created_at DESC);

-- Enable pgvector for hybrid search
ALTER TABLE documents ADD COLUMN embedding vector(384);
CREATE INDEX ON documents USING ivfflat (embedding vector_ip_ops);
```

---

## Monitoring

### Key Metrics to Track

1. **API Performance**
   - Request latency (p50, p95, p99)
   - Throughput (requests/sec)
   - Error rate

2. **Task Queue**
   - Task queue depth
   - Task processing time
   - Worker utilization

3. **Database Performance**
   - Query latency
   - Index hit rate
   - Storage usage
   - Replication lag (if applicable)

4. **System Resources**
   - CPU usage per service
   - Memory usage
   - Network I/O
   - Disk space remaining

### Grafana Dashboards

Pre-built dashboards available:
- API Performance Dashboard
- Celery Tasks Dashboard
- Database Performance Dashboard
- System Resources Dashboard

Import from `monitoring/dashboards/`

---

## Troubleshooting

### Common Issues

#### 1. **Celery Tasks Not Processing**

```powershell
# Check Redis connection
docker exec graphplag_redis redis-cli ping

# Check Celery worker logs
docker-compose -f docker-compose-scalable.yml logs worker-1

# Check task queue depth
docker exec graphplag_redis redis-cli LLEN celery
```

#### 2. **Milvus Search Too Slow**

```python
# Optimize Milvus parameters
from pymilvus import Collection

collection = Collection("document_embeddings")

# Increase nprobe for better recall
collection.load()
search_params = {
    "metric_type": "IP",
    "params": {"nprobe": 128}  # Increase from 64
}
```

#### 3. **PostgreSQL Disk Full**

```powershell
# Check disk usage
docker exec graphplag_postgres du -sh /var/lib/postgresql/data

# Archive old analyses
docker exec graphplag_postgres psql -U user -d graphplag -c "
  DELETE FROM analyses
  WHERE created_at < NOW() - INTERVAL '90 days'
  AND status = 'completed'
"

# Vacuum database
docker exec graphplag_postgres psql -U user -d graphplag -c "VACUUM ANALYZE;"
```

#### 4. **High Memory Usage**

```powershell
# Check memory per service
docker stats

# Reduce Redis maxmemory or enable eviction
docker exec graphplag_redis redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Reduce FastAPI workers if needed (in docker-compose)
# Change: uvicorn app_scalable:app --workers 4
# To:     uvicorn app_scalable:app --workers 2
```

---

## Performance Tuning

### For 10M+ Documents

1. **Vector Database Optimization**
   ```python
   # Use larger batch size for indexing
   batch_size = 10000
   
   # Use GPU-accelerated search (if available)
   # Requires CUDA installation
   ```

2. **Elasticsearch Tuning**
   ```yaml
   # Increase refresh interval for bulk indexing
   PUT /documents/_settings
   {
     "refresh_interval": "30s"  # From default 1s
   }
   ```

3. **PostgreSQL Tuning**
   ```sql
   -- Increase shared buffers
   ALTER SYSTEM SET shared_buffers = '4GB';
   
   -- Increase work_mem for sorts
   ALTER SYSTEM SET work_mem = '256MB';
   
   -- Restart PostgreSQL
   ```

4. **Celery Worker Tuning**
   ```powershell
   # Increase concurrency
   # In docker-compose: --concurrency=8 (from 4)
   
   # Use process pool instead of threads
   # Already configured in Dockerfile.worker
   ```

---

## Scaling to Production

### Kubernetes Deployment

For production cloud deployment:

```bash
# Apply Kubernetes manifests (provided in k8s/ directory)
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/worker-deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/autoscaling.yaml

# Verify deployment
kubectl get pods -n graphplag
kubectl get services -n graphplag
```

### Cloud Providers

- **AWS**: Use RDS for PostgreSQL, ElastiCache for Redis, SageMaker for Milvus
- **GCP**: Use Cloud SQL, Memorystore, Vertex AI
- **Azure**: Use Azure Database for PostgreSQL, Azure Cache for Redis

---

## Cost Analysis

For 10M documents, 1000 queries/day:

| Component | Estimated Cost/Month |
|-----------|----------------------|
| Compute (Kubernetes) | $700 |
| Vector DB (Milvus) | $491.50 |
| PostgreSQL | $450 |
| Elasticsearch | $300 |
| Redis | $200 |
| Storage (S3) | $23 |
| **Total** | **~$3,200** |

**Cost per query**: $0.11

---

## Next Steps

1. ✅ Start Docker Compose stack
2. ✅ Initialize databases
3. ✅ Upload initial document corpus
4. ✅ Test API endpoints
5. ✅ Monitor performance (Flower, Prometheus, Grafana)
6. ✅ Optimize parameters based on metrics
7. ✅ Plan cloud migration
8. ✅ Set up CI/CD pipelines

---

## Support & Documentation

- **Celery**: https://docs.celeryproject.io/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Milvus**: https://milvus.io/docs
- **Elasticsearch**: https://www.elastic.co/guide/
- **PostgreSQL**: https://www.postgresql.org/docs/

---

## License

GraphPlag - Production Plagiarism Detection System
