from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="GraphPlag API")

class AnalysisRequest(BaseModel):
    text: str
    document_id: str = "doc1"

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    return {"job_id": "job-1", "status": "processing"}

@app.get("/status/{job_id}")
async def status(job_id: str):
    return {"job_id": job_id, "status": "completed", "score": 0.85}

@app.get("/")
async def root():
    return {"name": "GraphPlag API", "version": "1.0.0"}
