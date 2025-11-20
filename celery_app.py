from celery import Celery
import os

# Get Redis URL from environment or use default
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://plagplag:plagplag@localhost:5432/plagdb")

celery_app = Celery(
    "graphplag",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

@celery_app.task(name='analyze_document')
def analyze_document_task(filename: str, content: str):
    """Analyze document for plagiarism"""
    # Simple plagiarism check - count repeated words
    words = content.lower().split()
    word_freq = {}
    
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Find suspicious words (appearing 5+ times)
    suspicious = {w: c for w, c in word_freq.items() if c >= 5 and len(w) > 3}
    
    plagiarism_score = min(100, len(suspicious) * 5)  # Simple scoring
    
    return {
        "filename": filename,
        "plagiarism_score": plagiarism_score,
        "suspicious_terms": len(suspicious),
        "status": "complete"
    }
