#!/bin/bash
# Deployment script for GraphPlag

set -e

echo "================================"
echo "GraphPlag Deployment Script"
echo "================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed"
    exit 1
fi

echo "✓ Docker and Docker Compose are installed"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p cache/embeddings cache/sentences logs models
echo "✓ Directories created"
echo ""

# Copy environment file if not exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠ Please update .env with your configuration"
    echo ""
fi

# Build Docker image
echo "Building Docker image..."
docker-compose build
echo "✓ Docker image built"
echo ""

# Start services
echo "Starting services..."
docker-compose up -d
echo "✓ Services started"
echo ""

# Wait for service to be healthy
echo "Waiting for service to be healthy..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker-compose exec -T graphplag-api curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ Service is healthy"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "✗ Service health check failed"
    echo "Check logs with: docker-compose logs"
    exit 1
fi

echo ""
echo "================================"
echo "✓ Deployment successful!"
echo "================================"
echo ""
echo "API is running at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Health Check: http://localhost:8000/health"
echo ""
echo "Useful commands:"
echo "  - View logs: docker-compose logs -f"
echo "  - Stop services: docker-compose down"
echo "  - Restart: docker-compose restart"
echo ""
