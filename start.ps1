#!/usr/bin/env powershell
# GraphPlag Scalable System - Quick Start Script

param(
    [string]$Action = "start"
)

$ErrorActionPreference = "Stop"
$projectRoot = "c:\Users\Amine EL-Hend\Documents\GitHub\GraphPlag"

Write-Host ""
Write-Host "GraphPlag Scalable System - Starting Deployment" -ForegroundColor Green
Write-Host ""

function Check-Prerequisites {
    Write-Host "Checking Prerequisites..." -ForegroundColor Cyan
    
    # Check Docker
    try {
        $dockerVersion = docker --version
        Write-Host "[OK] Docker: $dockerVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "[ERROR] Docker not found. Please install Docker first." -ForegroundColor Red
        exit 1
    }
    
    # Check Python
    try {
        $pythonVersion = python --version
        Write-Host "[OK] Python: $pythonVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "[ERROR] Python not found. Please install Python 3.10+ first." -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Prerequisites OK`n" -ForegroundColor Green
}

function Start-Services {
    Write-Host "Starting Docker Services..." -ForegroundColor Cyan
    Write-Host "Building Docker images..." -ForegroundColor Yellow
    
    docker-compose -f "$projectRoot\docker-compose-scalable.yml" build --no-cache
    
    Write-Host "Starting containers..." -ForegroundColor Yellow
    docker-compose -f "$projectRoot\docker-compose-scalable.yml" up -d
    
    Write-Host "[OK] Services started`n" -ForegroundColor Green
    
    Write-Host "Waiting for API to be ready..." -ForegroundColor Yellow
    $maxAttempts = 30
    $attempt = 0
    
    while ($attempt -lt $maxAttempts) {
        $attempt++
        
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Host "[OK] API is healthy`n" -ForegroundColor Green
                break
            }
        }
        catch {
            Write-Host "Waiting... ($attempt/$maxAttempts)" -ForegroundColor Yellow
            Start-Sleep -Seconds 2
        }
    }
}

function Initialize-Databases {
    Write-Host "Initializing Databases..." -ForegroundColor Cyan
    
    Write-Host "Waiting for PostgreSQL..." -ForegroundColor Yellow
    Start-Sleep -Seconds 15
    
    Write-Host "Setting up Milvus..." -ForegroundColor Yellow
    python "$projectRoot\scripts\setup_milvus.py"
    
    Write-Host "Setting up Elasticsearch..." -ForegroundColor Yellow
    python "$projectRoot\scripts\setup_elasticsearch.py"
    
    Write-Host "[OK] Databases initialized`n" -ForegroundColor Green
}

function Show-Dashboards {
    Write-Host "Services Ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Access these dashboards:" -ForegroundColor Cyan
    Write-Host "  FastAPI Docs:  http://localhost:8000/docs" -ForegroundColor White
    Write-Host "  Flower Tasks:  http://localhost:5555" -ForegroundColor White
    Write-Host "  Prometheus:    http://localhost:9090" -ForegroundColor White
    Write-Host "  Grafana:       http://localhost:3000 (admin/admin)" -ForegroundColor White
    Write-Host ""
    Write-Host "Database Connections:" -ForegroundColor Cyan
    Write-Host "  PostgreSQL:    localhost:5432 (user:pass)" -ForegroundColor White
    Write-Host "  Redis:         localhost:6379" -ForegroundColor White
    Write-Host "  Elasticsearch: http://localhost:9200" -ForegroundColor White
    Write-Host "  Milvus:        localhost:19530" -ForegroundColor White
    Write-Host ""
}

function Show-Status {
    Write-Host "Service Status:" -ForegroundColor Cyan
    docker-compose -f "$projectRoot\docker-compose-scalable.yml" ps
}

function Stop-Services {
    Write-Host "Stopping Services..." -ForegroundColor Cyan
    docker-compose -f "$projectRoot\docker-compose-scalable.yml" down -v
    Write-Host "[OK] Services stopped" -ForegroundColor Green
}

function Test-API {
    Write-Host "Testing API..." -ForegroundColor Cyan
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -Method Get
        $data = $response.Content | ConvertFrom-Json
        Write-Host "[OK] API Health: $($data.status)" -ForegroundColor Green
    }
    catch {
        Write-Host "[ERROR] Health check failed: $_" -ForegroundColor Red
        exit 1
    }
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/database-stats" -Method Get
        $stats = $response.Content | ConvertFrom-Json
        Write-Host "[OK] Database connected" -ForegroundColor Green
        Write-Host "  Documents: $($stats.total_documents)" -ForegroundColor White
        Write-Host "  Embeddings: $($stats.total_embeddings)" -ForegroundColor White
    }
    catch {
        Write-Host "[WARNING] Database stats unavailable (may still be initializing)" -ForegroundColor Yellow
    }
    
    Write-Host ""
}

# Main execution
switch ($Action.ToLower()) {
    "start" {
        Check-Prerequisites
        Start-Services
        Initialize-Databases
        Test-API
        Show-Dashboards
        Write-Host "System is READY! Open the dashboards above in your browser." -ForegroundColor Green
    }
    "stop" {
        Stop-Services
    }
    "status" {
        Show-Status
    }
    "test" {
        Test-API
    }
    default {
        Write-Host "Usage: .\start.ps1 [action]" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Actions:" -ForegroundColor Cyan
        Write-Host "  start  - Start services and initialize (default)" -ForegroundColor White
        Write-Host "  stop   - Stop all services" -ForegroundColor White
        Write-Host "  status - Show service status" -ForegroundColor White
        Write-Host "  test   - Run API tests" -ForegroundColor White
    }
}

Write-Host ""
