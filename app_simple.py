from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import json
from celery_app import analyze_document_task

app = FastAPI(title="GraphPlag API", version="1.0.0")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Upload a document for plagiarism analysis"""
    try:
        # Save uploaded file
        contents = await file.read()
        
        # Dispatch to Celery
        task = analyze_document_task.delay(
            filename=file.filename,
            content=contents.decode('utf-8', errors='ignore')[:1000]  # First 1000 chars
        )
        
        return {
            "job_id": task.id,
            "filename": file.filename,
            "status": "processing"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get job status"""
    from celery_app import celery_app
    task = celery_app.AsyncResult(job_id)
    return {
        "job_id": job_id,
        "status": task.status,
        "result": task.result if task.ready() else None
    }

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get analysis results"""
    from celery_app import celery_app
    task = celery_app.AsyncResult(job_id)
    
    if not task.ready():
        return {"status": "pending", "job_id": job_id}
    
    return {
        "job_id": job_id,
        "status": "complete",
        "results": task.result
    }

@app.get("/")
async def root():
    return {
        "service": "GraphPlag",
        "version": "1.0.0",
        "api_docs": "/docs",
        "status": "running"
    }
