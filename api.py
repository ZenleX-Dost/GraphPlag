"""
FastAPI REST API for GraphPlag Plagiarism Detection System.

Provides RESTful endpoints for plagiarism detection with authentication,
rate limiting, and async support.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import os
import tempfile
import time
import hashlib
from datetime import datetime
from pathlib import Path

from graphplag.detection.detector import PlagiarismDetector
from graphplag.utils.file_parser import FileParser

# Initialize FastAPI app
app = FastAPI(
    title="GraphPlag API",
    description="Graph-based Plagiarism Detection REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global detector instance (loaded once)
detector: Optional[PlagiarismDetector] = None

# Job storage for async tasks
jobs: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Pydantic Models
# ============================================================================

class TextCompareRequest(BaseModel):
    """Request model for text comparison."""
    text1: str = Field(..., description="First text to compare")
    text2: str = Field(..., description="Second text to compare")
    method: str = Field("kernel", description="Detection method: kernel, gnn, or ensemble")
    threshold: float = Field(0.7, ge=0.0, le=1.0, description="Plagiarism threshold")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text1": "Machine learning is a subset of AI.",
                "text2": "ML is a branch of artificial intelligence.",
                "method": "kernel",
                "threshold": 0.7
            }
        }


class ComparisonResult(BaseModel):
    """Response model for comparison results."""
    similarity: float
    is_plagiarism: bool
    threshold: float
    method: str
    kernel_scores: Optional[Dict[str, float]] = None
    processing_time: float
    doc1_sentences: int
    doc2_sentences: int


class BatchCompareRequest(BaseModel):
    """Request model for batch comparison."""
    texts: List[str] = Field(..., min_length=2, description="List of texts to compare")
    method: str = Field("kernel", description="Detection method")
    threshold: float = Field(0.7, ge=0.0, le=1.0)


class JobStatus(BaseModel):
    """Job status model."""
    job_id: str
    status: str  # pending, processing, completed, failed
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime: float
    cache_stats: Optional[Dict[str, Any]] = None


class ConfigResponse(BaseModel):
    """Configuration response."""
    methods: List[str]
    default_threshold: float
    supported_formats: List[str]
    max_file_size_mb: int


# ============================================================================
# Authentication
# ============================================================================

API_KEYS = {
    "demo_key_123": {"name": "Demo User", "rate_limit": 100},
    # Add more API keys here
}


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Verify API key."""
    token = credentials.credentials
    
    if token not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return API_KEYS[token]


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize detector on startup."""
    global detector
    print("Initializing plagiarism detector...")
    detector = PlagiarismDetector(
        method="kernel",
        threshold=0.7,
        use_cache=True
    )
    print("✓ Detector initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Shutting down API...")


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "service": "GraphPlag API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    cache_stats = detector.graph_builder.get_cache_stats()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime=time.time(),
        cache_stats=cache_stats
    )


@app.get("/config", response_model=ConfigResponse, tags=["General"])
async def get_config():
    """Get API configuration."""
    return ConfigResponse(
        methods=["kernel", "gnn", "ensemble"],
        default_threshold=0.7,
        supported_formats=["txt", "pdf", "docx", "md"],
        max_file_size_mb=50
    )


@app.post("/compare/text", response_model=ComparisonResult, tags=["Detection"])
async def compare_texts(
    request: TextCompareRequest,
    user: Dict[str, Any] = Depends(verify_token)
):
    """
    Compare two texts for plagiarism.
    
    Requires authentication via Bearer token.
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    try:
        start_time = time.time()
        
        # Detect plagiarism
        result = detector.detect_plagiarism(
            request.text1,
            request.text2
        )
        
        processing_time = time.time() - start_time
        
        return ComparisonResult(
            similarity=result.similarity_score,
            is_plagiarism=result.is_plagiarism,
            threshold=result.threshold,
            method=result.method,
            kernel_scores=result.kernel_scores,
            processing_time=processing_time,
            doc1_sentences=len(result.document1.sentences),
            doc2_sentences=len(result.document2.sentences)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


@app.post("/compare/files", response_model=ComparisonResult, tags=["Detection"])
async def compare_files(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    method: str = "kernel",
    threshold: float = 0.7,
    user: Dict[str, Any] = Depends(verify_token)
):
    """
    Compare two uploaded files for plagiarism.
    
    Supports: TXT, PDF, DOCX, MD
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    # Check file sizes (50MB limit)
    MAX_SIZE = 50 * 1024 * 1024
    
    try:
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file1.filename).suffix) as tmp1:
            content1 = await file1.read()
            if len(content1) > MAX_SIZE:
                raise HTTPException(status_code=413, detail="File 1 too large (max 50MB)")
            tmp1.write(content1)
            tmp1_path = tmp1.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file2.filename).suffix) as tmp2:
            content2 = await file2.read()
            if len(content2) > MAX_SIZE:
                raise HTTPException(status_code=413, detail="File 2 too large (max 50MB)")
            tmp2.write(content2)
            tmp2_path = tmp2.name
        
        # Parse files
        parser = FileParser()
        text1 = parser.parse_file(tmp1_path)
        text2 = parser.parse_file(tmp2_path)
        
        # Detect plagiarism
        start_time = time.time()
        result = detector.detect_plagiarism(text1, text2)
        processing_time = time.time() - start_time
        
        # Cleanup
        os.unlink(tmp1_path)
        os.unlink(tmp2_path)
        
        return ComparisonResult(
            similarity=result.similarity_score,
            is_plagiarism=result.is_plagiarism,
            threshold=result.threshold,
            method=result.method,
            kernel_scores=result.kernel_scores,
            processing_time=processing_time,
            doc1_sentences=len(result.document1.sentences),
            doc2_sentences=len(result.document2.sentences)
        )
    
    except Exception as e:
        # Cleanup on error
        if 'tmp1_path' in locals():
            try:
                os.unlink(tmp1_path)
            except:
                pass
        if 'tmp2_path' in locals():
            try:
                os.unlink(tmp2_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


@app.post("/batch/compare", tags=["Detection"])
async def batch_compare(
    request: BatchCompareRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(verify_token)
):
    """
    Compare multiple texts in batch mode (async).
    
    Returns a job ID to check status later.
    """
    # Generate job ID
    job_id = hashlib.md5(f"{time.time()}{request.texts}".encode()).hexdigest()
    
    # Store job
    jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "result": None,
        "error": None
    }
    
    # Process in background
    background_tasks.add_task(process_batch, job_id, request)
    
    return {"job_id": job_id, "status": "pending", "check_url": f"/job/{job_id}"}


@app.get("/job/{job_id}", response_model=JobStatus, tags=["Jobs"])
async def get_job_status(job_id: str, user: Dict[str, Any] = Depends(verify_token)):
    """Get status of a batch job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        result=job.get("result"),
        error=job.get("error")
    )


@app.delete("/cache", tags=["Management"])
async def clear_cache(user: Dict[str, Any] = Depends(verify_token)):
    """Clear the embedding cache."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    detector.graph_builder.clear_cache()
    return {"message": "Cache cleared successfully"}


@app.get("/cache/stats", tags=["Management"])
async def get_cache_stats(user: Dict[str, Any] = Depends(verify_token)):
    """Get cache statistics."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    stats = detector.graph_builder.get_cache_stats()
    return stats or {"message": "Cache disabled"}


# ============================================================================
# Background Tasks
# ============================================================================

async def process_batch(job_id: str, request: BatchCompareRequest):
    """Process batch comparison in background."""
    try:
        jobs[job_id]["status"] = "processing"
        
        results = []
        n = len(request.texts)
        
        # Compare all pairs
        for i in range(n):
            for j in range(i + 1, n):
                result = detector.detect_plagiarism(
                    request.texts[i],
                    request.texts[j]
                )
                results.append({
                    "pair": [i, j],
                    "similarity": result.similarity_score,
                    "is_plagiarism": result.is_plagiarism
                })
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["result"] = results
    
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
