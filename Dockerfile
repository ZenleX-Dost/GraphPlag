# GraphPlag - Graph-based Plagiarism Detection
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install NumPy 1.x first (GraKeL compatibility)
RUN pip install --no-cache-dir "numpy<2.0.0"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install API dependencies
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    python-multipart==0.0.6 \
    reportlab==4.0.8 \
    openpyxl==3.1.2

# Copy application code
COPY . .

# Apply GraKeL patches
RUN python patch_grakel.py || echo "Warning: GraKeL patching skipped"

# Create cache directory
RUN mkdir -p .cache/embeddings .cache/sentences

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
