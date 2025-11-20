#!/bin/bash

# Fast deployment script - minimal setup
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "GraphPlag Fast Deployment"
echo "========================="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not installed"
    exit 1
fi

echo "[OK] Docker found"

# Start services
echo "Starting minimal services (API + Redis + PostgreSQL + Workers)..."
docker-compose -f docker-compose-fast.yml up -d

echo ""
echo "Waiting for services to start..."
sleep 5

# Check if services are running
docker-compose -f docker-compose-fast.yml ps

echo ""
echo "==================================="
echo "Services Ready!"
echo "==================================="
echo ""
echo "API Docs:     http://localhost:8000/docs"
echo "Flower UI:    http://localhost:5555"
echo ""
echo "Database:     PostgreSQL on localhost:5432"
echo "Cache:        Redis on localhost:6379"
echo ""
echo "To stop:      docker-compose -f docker-compose-fast.yml down"
