# Scaling GraphPlag: From 1-to-1 to 1-to-Millions

## Current State vs. Scalable Architecture

### ❌ Current Architecture (1-to-1 Comparison)
```
User Input
    ↓
[Document Parser] → [Graph Builder] → [Similarity Computer]
    ↓
PlagiarismReport (Single Comparison)
```

**Limitations**:
- ❌ Only compares 2 documents at a time
- ❌ All computation on single machine
- ❌ Limited by available RAM/GPU
- ❌ No distributed caching
- ❌ Sequential processing
- ❌ Cannot scale beyond single server

### ✅ Scalable Architecture (1-to-Millions)
```
┌─────────────────────────────────────────────────────────────┐
│                     User Upload (Query Doc)                  │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
                  ┌─────────────────┐
                  │  API Gateway    │
                  │  (FastAPI)      │
                  └────────┬────────┘
                           ↓
        ┌──────────────────────────────────────┐
        │  Document Processing Pipeline        │
        │  (Pre-compute + Store in DB)        │
        └────────┬─────────────────────────────┘
                 ↓
        ┌─────────────────────────────────────┐
        │   Distributed Cache Layer           │
        │   - Redis (Embeddings)              │
        │   - Elasticsearch (Full-text)       │
        │   - Vector DB (Graph kernels)       │
        └────────┬────────────────────────────┘
                 ↓
        ┌─────────────────────────────────────┐
        │  Distributed Similarity Search      │
        │  - Spark Jobs (Batch)               │
        │  - Stream Processing (Real-time)    │
        │  - Approximate NN Search            │
        └────────┬────────────────────────────┘
                 ↓
        ┌─────────────────────────────────────┐
        │  Results Aggregation & Ranking      │
        └────────┬────────────────────────────┘
                 ↓
          PlagiarismReport (Top 1000 matches)
```

---

## Architecture: Three-Tier Approach

### Tier 1: Data Ingestion & Preprocessing (Batch)
### Tier 2: Online Query Processing (Real-time)
### Tier 3: Database & Indexing (Storage)

---

# TIER 1: BATCH PREPROCESSING (Apache Spark)

## Why Spark Instead of Hadoop?

| Feature | Hadoop | Spark | Choice |
|---------|--------|-------|--------|
| Speed | Disk-based (slow) | In-memory (100x faster) | **Spark** |
| Programming | Java/MapReduce (complex) | Python/Scala (simple) | **Spark** |
| Real-time | No | Yes (Streaming) | **Spark** |
| Graph processing | Limited | GraphX built-in | **Spark** |
| ML Integration | MLlib (basic) | MLlib + PyTorch | **Spark** |
| Community | Declining | Very active | **Spark** |

### Spark Architecture for GraphPlag

```
Batch Job: Process 1M Documents in Parallel
┌────────────────────────────────────────────────┐
│          Spark Driver (Orchestrator)           │
└────────────┬─────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────┐
│        RDD: Documents [1000, 1001, ...]        │
└────────────┬─────────────────────────────────┘
             ↓
   ┌─────────┴─────────┬─────────────┬─────────────┐
   ↓                   ↓             ↓             ↓
[Worker 1]        [Worker 2]    [Worker 3]   [Worker 4]
Parse + Build     Parse +       Parse +      Parse +
Graph             Build Graph   Build Graph  Build Graph
│                 │             │            │
└─────────────────┴─────────────┴────────────┘
             ↓
    Compute All Embeddings
    (In-Memory Caching)
             ↓
    Store in Vector Database
    + Elasticsearch Index
```

### Spark Job Implementation

```python
# spark_batch_processor.py
from pyspark.sql import SparkSession
from pyspark.ml.feature import BucketedRandomProjectionLSH
from graphplag.core.document_parser import DocumentParser
from graphplag.core.graph_builder import GraphBuilder

class SparkGraphPlagProcessor:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.parser = DocumentParser()
        self.builder = GraphBuilder()
    
    def process_document_batch(self, document_paths: List[str]):
        """
        Distributed processing of documents
        """
        # Create RDD from document paths
        docs_rdd = self.spark.sparkContext.parallelize(
            document_paths, 
            numPartitions=100  # Distribute across 100 partitions
        )
        
        # Transform: Read → Parse → Build Graph → Compute Embedding
        embeddings_rdd = docs_rdd.map(self._process_document)
        
        # Cache in memory for repeated access
        embeddings_rdd.cache()
        
        # Store to Vector Database (Milvus/Weaviate)
        embeddings_rdd.foreachPartition(
            lambda partition: self._store_to_vector_db(partition)
        )
        
        # Store to Elasticsearch for full-text search
        embeddings_rdd.foreachPartition(
            lambda partition: self._store_to_elasticsearch(partition)
        )
        
        return embeddings_rdd.count()
    
    def _process_document(self, doc_path: str) -> Dict:
        """Single document processing (runs on worker)"""
        doc = self.parser.parse_document_from_file(doc_path)
        graph = self.builder.build_graph(doc)
        embedding = graph.get_kernel_embedding()  # Pre-computed
        
        return {
            'doc_id': doc.id,
            'path': doc_path,
            'embedding': embedding,
            'text': doc.text,
            'metadata': doc.metadata
        }
    
    def _store_to_vector_db(self, partition):
        """Store embeddings to vector database"""
        from milvus import Milvus  # Vector DB client
        client = Milvus(uri="milvus://milvus-service:19530")
        
        for record in partition:
            client.insert(
                collection_name="document_embeddings",
                records=[{
                    'id': record['doc_id'],
                    'embedding': record['embedding'],
                    'metadata': record['metadata']
                }]
            )
    
    def _store_to_elasticsearch(self, partition):
        """Full-text indexing"""
        from elasticsearch import Elasticsearch
        es = Elasticsearch(["elasticsearch-service:9200"])
        
        for record in partition:
            es.index(
                index="documents",
                id=record['doc_id'],
                body={
                    'text': record['text'],
                    'metadata': record['metadata']
                }
            )
```

### Submit Spark Job (Docker)

```bash
# docker-compose.yml - Spark cluster definition
version: '3'
services:
  spark-master:
    image: bitnami/spark:3.4.0
    environment:
      SPARK_MODE: master
    ports:
      - "7077:7077"
      - "8080:8080"
    volumes:
      - ./documents:/data/documents:ro
  
  spark-worker-1:
    image: bitnami/spark:3.4.0
    environment:
      SPARK_MODE: worker
      SPARK_MASTER_URL: spark://spark-master:7077
    depends_on:
      - spark-master
  
  spark-worker-2:
    image: bitnami/spark:3.4.0
    environment:
      SPARK_MODE: worker
      SPARK_MASTER_URL: spark://spark-master:7077
    depends_on:
      - spark-master
  
  spark-worker-3:
    image: bitnami/spark:3.4.0
    environment:
      SPARK_MODE: worker
      SPARK_MASTER_URL: spark://spark-master:7077
    depends_on:
      - spark-master
```

```bash
# Submit batch job
spark-submit \
  --master spark://spark-master:7077 \
  --num-executors 10 \
  --executor-cores 4 \
  --executor-memory 8g \
  spark_batch_processor.py \
  --documents-path /data/documents \
  --batch-size 100000
```

---

# TIER 2: ONLINE QUERY PROCESSING (Real-time API)

## FastAPI + Async Processing

### Architecture

```
User Query (Upload Document)
        ↓
┌─────────────────────────────────┐
│  FastAPI Endpoint               │
│  - Parse upload                 │
│  - Store in temporary storage   │
│  - Queue job                    │
└────────┬────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  Job Queue (Redis)              │
│  - Task priority queue          │
│  - Rate limiting                │
└────────┬────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  Celery Workers (Parallel)      │
│  Worker 1: Parse + Build Graph  │
│  Worker 2: Compute Embedding    │
│  Worker 3: Vector Search        │
│  Worker 4: Full-text Search     │
│  Worker 5: Results Ranking      │
└────────┬────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  Results Aggregation            │
│  - Merge results                │
│  - Rank by similarity           │
│  - Generate report              │
└────────┬────────────────────────┘
         ↓
    Return Results (Streaming)
```

### Implementation

```python
# app_scalable.py - Distributed query processing
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from celery import Celery
from redis import Redis
import asyncio

app = FastAPI()
celery_app = Celery('graphplag', broker='redis://redis:6379/0')
redis_client = Redis(host='redis', port=6379)

@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload document → Compare against DB → Stream results
    """
    # Store file temporarily
    doc_path = await save_upload(file)
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Queue processing task
    task = celery_app.apply_async(
        'tasks.process_and_search',
        args=(doc_path,),
        task_id=job_id,
        priority=7  # High priority
    )
    
    # Return immediately with job ID
    return {
        "job_id": job_id,
        "status": "processing",
        "status_url": f"/status/{job_id}",
        "results_url": f"/results/{job_id}"
    }

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """
    Get results (streaming) with Server-Sent Events
    """
    async def result_generator():
        # Get from cache if available
        cached = redis_client.get(f"results:{job_id}")
        if cached:
            yield cached
            return
        
        # Otherwise, stream from job
        while True:
            status = redis_client.hgetall(f"job:{job_id}")
            
            if status.get('status') == 'completed':
                yield json.dumps(status['results'])
                break
            elif status.get('status') == 'failed':
                yield json.dumps({'error': status['error']})
                break
            
            await asyncio.sleep(0.5)
    
    return StreamingResponse(result_generator(), media_type="application/json")
```

### Celery Tasks (Distributed Workers)

```python
# tasks.py - Worker tasks
from celery import Celery, group, chain
from graphplag.core.document_parser import DocumentParser
from graphplag.similarity.vector_search import VectorSearcher

celery_app = Celery('graphplag', broker='redis://redis:6379/0')

@celery_app.task(bind=True)
def process_and_search(self, doc_path: str):
    """
    Main orchestration task:
    1. Parse document
    2. Search vector DB (parallel)
    3. Search full-text (parallel)
    4. Aggregate results
    """
    job_id = self.request.id
    
    try:
        # Update status
        redis_client.hset(f"job:{job_id}", "status", "parsing")
        
        # Task 1: Parse and build graph
        parser = DocumentParser()
        document = parser.parse_document_from_file(doc_path)
        
        # Update status
        redis_client.hset(f"job:{job_id}", "status", "searching")
        
        # Task 2 & 3: Parallel searching
        vector_results = vector_search.async_search(
            document.embedding,
            top_k=1000
        )
        fulltext_results = fulltext_search.async_search(
            document.text,
            top_k=1000
        )
        
        # Task 4: Aggregate and rank
        redis_client.hset(f"job:{job_id}", "status", "ranking")
        
        final_results = aggregate_and_rank(
            vector_results,
            fulltext_results,
            document
        )
        
        # Store results
        redis_client.hset(f"job:{job_id}", "status", "completed")
        redis_client.hset(f"job:{job_id}", "results", 
                         json.dumps(final_results))
        
        return {
            "job_id": job_id,
            "status": "completed",
            "results_count": len(final_results),
            "top_match_similarity": final_results[0]['similarity'] if final_results else 0
        }
    
    except Exception as e:
        redis_client.hset(f"job:{job_id}", "status", "failed")
        redis_client.hset(f"job:{job_id}", "error", str(e))
        raise

def aggregate_and_rank(vector_results, fulltext_results, query_doc):
    """
    Merge results from multiple search methods
    """
    # Normalize scores
    combined = {}
    
    for result in vector_results:
        doc_id = result['doc_id']
        combined[doc_id] = {
            'vector_score': normalize(result['similarity']),
            'doc': result['doc'],
            'fulltext_score': 0
        }
    
    for result in fulltext_results:
        doc_id = result['doc_id']
        if doc_id not in combined:
            combined[doc_id] = {'vector_score': 0, 'fulltext_score': 0, 'doc': result['doc']}
        combined[doc_id]['fulltext_score'] = normalize(result['score'])
    
    # Weighted ensemble score
    for doc_id in combined:
        combined[doc_id]['final_score'] = (
            0.6 * combined[doc_id]['vector_score'] +
            0.4 * combined[doc_id]['fulltext_score']
        )
    
    # Sort and return top 100
    ranked = sorted(combined.items(), 
                   key=lambda x: x[1]['final_score'], 
                   reverse=True)
    
    return [
        {
            'doc_id': doc_id,
            'similarity': scores['final_score'],
            'matched_doc': scores['doc']
        }
        for doc_id, scores in ranked[:100]
    ]
```

---

# TIER 3: DATABASES & INDEXING

## Three-Layer Storage Strategy

### Layer 1: Vector Database (Approximate Nearest Neighbor Search)

**Why Milvus/Weaviate instead of Elasticsearch alone?**

```
Query Embedding (768 dimensions)
        ↓
Traditional approach (Elasticsearch):
- Compute distance to ALL 1M documents: ~1000ms ❌
- Memory intensive: ~1GB per embedding
- Not optimized for semantic similarity

HNSW approach (Milvus/Weaviate):
- Create hierarchical graph structure once
- Query in ~50ms ✅
- Memory efficient: ~100MB for 1M embeddings
- Approximate but 99.9% accurate
```

### Layer 2: Full-Text Search (Elasticsearch)

**For keyword-based plagiarism detection**

```python
# Full-text index setup
mapping = {
    "mappings": {
        "properties": {
            "text": {
                "type": "text",
                "analyzer": "standard"
            },
            "metadata": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "language": {"type": "keyword"}
        }
    }
}

# Query (Find documents containing exact phrases)
query = {
    "query": {
        "bool": {
            "must": [
                {"match": {"text": "suspicious phrase"}}
            ],
            "filter": [
                {"term": {"language": "en"}}
            ]
        }
    }
}
```

### Layer 3: Document Storage (PostgreSQL + S3)

**Why separate storage?**

```python
# PostgreSQL: Metadata + Structured Data
CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,
    doc_id VARCHAR(255) UNIQUE,
    title VARCHAR(500),
    author VARCHAR(255),
    language VARCHAR(10),
    created_at TIMESTAMP,
    file_hash VARCHAR(64),  -- For deduplication
    vector_embedding VECTOR(768),  -- pgvector extension
    INDEX idx_embedding ON documents USING ivfflat (vector_embedding)
);

# S3: Raw files (PDFs, DOCX, etc.)
# CloudFront CDN for fast retrieval

# Reasoning:
- PostgreSQL: Fast queries, transactions, relationships
- S3: Cheap storage, scalable, versioning
- CDN: Fast delivery to users globally
```

---

## Complete Docker Compose Setup

```yaml
# docker-compose-scalable.yml
version: '3.8'

services:
  # API Gateway
  api:
    build: .
    command: uvicorn app_scalable:app --host 0.0.0.0 --port 8000 --workers 4
    ports:
      - "8000:8000"
    environment:
      REDIS_URL: redis://redis:6379/0
      POSTGRES_URL: postgresql://user:pass@postgres:5432/graphplag
      MILVUS_URL: milvus://milvus:19530
      ELASTICSEARCH_URL: http://elasticsearch:9200
    depends_on:
      - redis
      - postgres
      - milvus
      - elasticsearch
    volumes:
      - ./uploads:/app/uploads
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
  
  # Job Queue & Caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
  
  # Celery Workers
  worker:
    build: .
    command: celery -A tasks worker --loglevel=info --concurrency=4
    environment:
      CELERY_BROKER_URL: redis://redis:6379/0
      CELERY_RESULT_BACKEND: redis://redis:6379/1
      POSTGRES_URL: postgresql://user:pass@postgres:5432/graphplag
      MILVUS_URL: milvus://milvus:19530
    depends_on:
      - redis
      - postgres
      - milvus
    deploy:
      replicas: 4  # Scale to 4 workers
  
  # Vector Database
  milvus:
    image: milvusdb/milvus:v0.24.0
    ports:
      - "19530:19530"
      - "9091:9091"
    environment:
      COMMON_STORAGETYPE: minio
    depends_on:
      - minio
    volumes:
      - milvus_data:/var/lib/milvus
  
  # Object Storage for Milvus
  minio:
    image: minio/minio:latest
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: minio server /minio_data
    volumes:
      - minio_data:/minio_data
  
  # Full-Text Search
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.9.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
  
  # Relational Database
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: graphplag
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
  
  # Spark Cluster (for batch processing)
  spark-master:
    image: bitnami/spark:3.4.0
    environment:
      SPARK_MODE: master
    ports:
      - "7077:7077"
      - "8080:8080"
    volumes:
      - ./documents:/data/documents:ro
  
  spark-worker-1:
    image: bitnami/spark:3.4.0
    environment:
      SPARK_MODE: worker
      SPARK_MASTER_URL: spark://spark-master:7077
    depends_on:
      - spark-master
  
  spark-worker-2:
    image: bitnami/spark:3.4.0
    environment:
      SPARK_MODE: worker
      SPARK_MASTER_URL: spark://spark-master:7077
    depends_on:
      - spark-master
  
  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    depends_on:
      - prometheus

volumes:
  redis_data:
  postgres_data:
  milvus_data:
  minio_data:
  elasticsearch_data:
```

---

## Performance Expectations

### Scenario: Compare 1 Document Against 10 Million Documents

```
With Distributed Architecture:
┌─────────────────────────────────────────────┐
│ Upload Document        (2 sec)              │
├─────────────────────────────────────────────┤
│ Parse + Build Graph    (0.5 sec)            │
├─────────────────────────────────────────────┤
│ Parallel Search (async):                    │
│   - Vector DB Search   (0.05 sec)           │
│   - Full-text Search   (0.1 sec)            │
│   - Run on 4 workers   (0.15 sec total)     │
├─────────────────────────────────────────────┤
│ Aggregate + Rank       (0.5 sec)            │
├─────────────────────────────────────────────┤
│ Generate Report        (1 sec)              │
└─────────────────────────────────────────────┘
Total: ~4 seconds for 10M documents! ✅

Without Distributed Architecture:
- 1 Query vs 10M documents = ~10,000 seconds ❌
```

---

## Kubernetes Deployment (Production)

For true production scale, use Kubernetes:

```yaml
# kubernetes/deployment.yaml
apiVersion: v1
kind: Service
metadata:
  name: graphplag-api
spec:
  selector:
    app: graphplag
  ports:
    - port: 80
      targetPort: 8000
  type: LoadBalancer
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graphplag-api
spec:
  replicas: 5  # Auto-scale based on load
  selector:
    matchLabels:
      app: graphplag
  template:
    metadata:
      labels:
        app: graphplag
    spec:
      containers:
      - name: api
        image: graphplag:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: REDIS_URL
          value: redis://redis:6379/0
        - name: POSTGRES_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: graphplag-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: graphplag-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Cost Analysis

### Cloud Deployment Costs (AWS Example - Monthly)

```
For 1M documents, 1000 queries/day:

Infrastructure:
├─ Compute (Kubernetes)
│  └─ 5x m5.xlarge instances × $0.192/hr × 730 hrs = $700
├─ Vector Database (Milvus)
│  └─ Storage: 10M × 768 × 4 bytes = 30GB = $1.50
│  └─ Compute: 1x m5.2xlarge = $490
├─ PostgreSQL Database
│  └─ RDS r5.xlarge + storage = $450
├─ Elasticsearch
│  └─ m5.large × 2 = $300
├─ Redis Cache
│  └─ cache.r6g.xlarge = $200
└─ S3 Storage
   └─ 10M files × 100KB avg = 1TB = $23

Total: ~$3,200/month for 10M documents

Cost per query: $3,200 / 30,000 = $0.11/query
```

---

## Migration Path

### Phase 1: Single-Instance Optimization (Week 1-2)
- ✅ Add caching (Redis)
- ✅ Add vector indexing (Milvus local)
- ✅ Optimize code
- **Result**: 5-10x speedup

### Phase 2: Containerization (Week 3-4)
- ✅ Docker setup
- ✅ Docker Compose orchestration
- ✅ Local Spark cluster
- **Result**: Easy scaling locally

### Phase 3: Cloud Deployment (Week 5-8)
- ✅ AWS/GCP migration
- ✅ Kubernetes setup
- ✅ Auto-scaling groups
- ✅ CDN integration
- **Result**: Global production system

### Phase 4: Advanced Features (Week 9+)
- ✅ Distributed training
- ✅ Multi-region replication
- ✅ Machine learning optimization
- ✅ Advanced monitoring/alerting

---

## Key Takeaways

| Problem | Solution | Technology |
|---------|----------|-----------|
| Scale to 10M documents | Batch processing | Apache Spark |
| Sub-second search | Approximate NN | Milvus + HNSW |
| Full-text search | Inverted index | Elasticsearch |
| Real-time API | Async workers | FastAPI + Celery |
| Distributed queue | Job queue | Redis + Celery |
| Relational data | Transactions | PostgreSQL |
| Metadata caching | In-memory | Redis |
| Global scale | Cloud CDN | AWS CloudFront |
| Auto-scaling | Orchestration | Kubernetes |
| Monitoring | Metrics | Prometheus + Grafana |

**Why NOT pure Hadoop?**
- ❌ Slower than Spark (disk-based)
- ❌ Complex Java ecosystem
- ❌ No real-time streaming
- ❌ Outdated architecture (HDFS unnecessary)

**Why Spark + Docker + Kubernetes?**
- ✅ Python-native (easier for your team)
- ✅ In-memory processing (100x faster)
- ✅ Real-time + batch in one system
- ✅ Modern DevOps practices
- ✅ Industry standard (Netflix, Uber, Alibaba)

---

## Next Steps

1. **Add Vector Database** - Replace single-machine similarity with Milvus
2. **Implement Celery** - Queue document processing tasks
3. **Add Elasticsearch** - Full-text search capability
4. **Docker Setup** - Containerize everything
5. **Deploy to Cloud** - AWS/GCP Kubernetes cluster
6. **Monitor & Scale** - Auto-scaling based on demand

Your GraphPlag system can scale from 1-to-1 comparisons to searching **billions of documents in seconds**! 🚀

