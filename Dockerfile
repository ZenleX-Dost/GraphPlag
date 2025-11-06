# GraphPlag - Graph-based Plagiarism Detection API
# Optimized for Railway deployment (< 4GB image size)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables to reduce image size
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements
COPY requirements.txt .

# Install NumPy 1.x first (GraKeL compatibility)
RUN pip install --no-cache-dir "numpy<2.0.0"

# Install Python dependencies (minimal set for API)
# Remove unnecessary packages for deployment
RUN pip install --no-cache-dir \
    scipy>=1.7.0 \
    spacy>=3.5.0 \
    sentence-transformers>=2.2.0 \
    torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu \
    networkx>=3.0 \
    grakel>=0.1.9 \
    scikit-learn>=1.0.0 \
    langdetect>=1.0.9 \
    pyyaml>=6.0 \
    tqdm>=4.64.0 \
    PyPDF2>=3.0.0 \
    python-docx>=1.0.0 \
    markdown>=3.4.0

# Install API dependencies
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    python-multipart==0.0.6 \
    reportlab==4.0.8 \
    openpyxl>=3.1.2

# Copy only necessary application files
COPY graphplag/ ./graphplag/
COPY api.py .
COPY patch_grakel.py .
COPY .env.example .

# Apply GraKeL patches
RUN python patch_grakel.py || echo "Warning: GraKeL patching skipped"

# Create cache directory
RUN mkdir -p .cache/embeddings .cache/sentences

# Remove build dependencies to reduce size
RUN apt-get purge -y --auto-remove build-essential git \
    && rm -rf /root/.cache \
    && find /usr/local -type f -name '*.pyc' -delete \
    && find /usr/local -type d -name '__pycache__' -delete

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
