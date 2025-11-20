#!/usr/bin/env powershell
<#
.SYNOPSIS
GraphPlag Scalable Deployment Quick Start Script
.DESCRIPTION
Automates the setup and initialization of the GraphPlag scalable system.
#>

param(
    [string]$Action = "start",
    [string]$Environment = "development"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommandPath

function Write-Header {
    param([string]$text)
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "| $text" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$text)
    Write-Host "✓ $text" -ForegroundColor Green
}

function Write-Warning {
    param([string]$text)
    Write-Host "⚠ $text" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$text)
    Write-Host "✗ $text" -ForegroundColor Red
}

function Check-Prerequisites {
    Write-Header "Checking Prerequisites"
    
    $required = @{
        "Docker" = "docker --version"
        "Python" = "python --version"
        "Git" = "git --version"
    }
    
    foreach ($tool in $required.GetEnumerator()) {
        try {
            $output = Invoke-Expression $tool.Value 2>&1
            Write-Success "$($tool.Key): $output"
        }
        catch {
            Write-Error "$($tool.Key) not found. Please install it first."
            exit 1
        }
    }
}

function Start-Services {
    Write-Header "Starting Docker Services"
    
    Write-Host "Building Docker images..." -ForegroundColor Cyan
    docker-compose -f "$projectRoot\docker-compose-scalable.yml" build --no-cache
    
    Write-Host "Starting services..." -ForegroundColor Cyan
    docker-compose -f "$projectRoot\docker-compose-scalable.yml" up -d
    
    Write-Success "Services started"
    
    Write-Host "`nWaiting for services to be ready..." -ForegroundColor Cyan
    $maxAttempts = 30
    $attempt = 0
    
    while ($attempt -lt $maxAttempts) {
        $attempt++
        
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Success "API is healthy"
                break
            }
        }
        catch {
            Write-Host "Attempt $attempt/$maxAttempts..." -ForegroundColor Yellow
            Start-Sleep -Seconds 2
        }
    }
    
    if ($attempt -eq $maxAttempts) {
        Write-Warning "API took longer than expected to start. Continuing anyway..."
    }
}

function Initialize-Databases {
    Write-Header "Initializing Databases"
    
    # Wait for PostgreSQL
    Write-Host "Waiting for PostgreSQL..." -ForegroundColor Cyan
    Start-Sleep -Seconds 15
    
    # Setup Milvus
    Write-Host "Setting up Milvus vector database..." -ForegroundColor Cyan
    & python "$projectRoot\scripts\setup_milvus.py"
    
    # Setup Elasticsearch
    Write-Host "Setting up Elasticsearch indices..." -ForegroundColor Cyan
    & python "$projectRoot\scripts\setup_elasticsearch.py"
    
    Write-Success "Databases initialized"
}

function Show-Dashboard {
    Write-Header "Service Dashboards"
    
    Write-Host "Services are ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "API & Monitoring Endpoints:" -ForegroundColor Cyan
    Write-Host "  • FastAPI:       http://localhost:8000" -ForegroundColor Yellow
    Write-Host "  • API Docs:      http://localhost:8000/docs" -ForegroundColor Yellow
    Write-Host "  • Flower (Tasks): http://localhost:5555" -ForegroundColor Yellow
    Write-Host "  • Prometheus:    http://localhost:9090" -ForegroundColor Yellow
    Write-Host "  • Grafana:       http://localhost:3000 (admin/admin)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Databases:" -ForegroundColor Cyan
    Write-Host "  • PostgreSQL:    localhost:5432 (user:pass)" -ForegroundColor Yellow
    Write-Host "  • Redis:         localhost:6379" -ForegroundColor Yellow
    Write-Host "  • Elasticsearch: http://localhost:9200" -ForegroundColor Yellow
    Write-Host "  • Milvus:        localhost:19530" -ForegroundColor Yellow
    Write-Host "  • MinIO:         http://localhost:9001" -ForegroundColor Yellow
    Write-Host ""
}

function Show-Logs {
    Write-Header "Recent Logs"
    docker-compose -f "$projectRoot\docker-compose-scalable.yml" logs --tail=20 api
}

function Stop-Services {
    Write-Header "Stopping Services"
    docker-compose -f "$projectRoot\docker-compose-scalable.yml" down -v
    Write-Success "Services stopped"
}

function Show-Status {
    Write-Header "Service Status"
    docker-compose -f "$projectRoot\docker-compose-scalable.yml" ps
}

function Test-API {
    Write-Header "Testing API"
    
    Write-Host "Testing health endpoint..." -ForegroundColor Cyan
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -Method Get
        $data = $response.Content | ConvertFrom-Json
        Write-Success "API is healthy: $($data.status)"
    }
    catch {
        Write-Error "Health check failed: $_"
        exit 1
    }
    
    Write-Host "`nTesting database stats endpoint..." -ForegroundColor Cyan
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/database-stats" -Method Get
        $stats = $response.Content | ConvertFrom-Json
        Write-Success "Database stats retrieved:"
        Write-Host "  • Documents: $($stats.total_documents)"
        Write-Host "  • Embeddings: $($stats.total_embeddings)"
    }
    catch {
        Write-Error "Database stats failed: $_"
    }
}

function Cleanup-All {
    Write-Header "Full Cleanup"
    
    Write-Warning "This will delete all containers, volumes, and data!"
    $confirm = Read-Host "Continue? (yes/no)"
    
    if ($confirm -eq "yes") {
        docker-compose -f "$projectRoot\docker-compose-scalable.yml" down -v --remove-orphans
        Write-Success "Cleanup complete"
    }
    else {
        Write-Host "Cleanup cancelled"
    }
}

# Main execution
Write-Host ""
Write-Host "===============================================================" -ForegroundColor Magenta
Write-Host "|  GraphPlag Scalable System - Quick Start Script            |" -ForegroundColor Magenta
Write-Host "===============================================================" -ForegroundColor Magenta

switch ($Action.ToLower()) {
    "start" {
        Check-Prerequisites
        Start-Services
        Initialize-Databases
        Test-API
        Show-Dashboard
    }
    "stop" {
        Stop-Services
    }
    "status" {
        Show-Status
    }
    "logs" {
        Show-Logs
    }
    "test" {
        Test-API
    }
    "cleanup" {
        Cleanup-All
    }
    "restart" {
        Stop-Services
        Start-Sleep -Seconds 3
        Start-Services
        Initialize-Databases
        Show-Dashboard
    }
    default {
        Write-Host ""
        Write-Host "Usage: .\quickstart.ps1 [action]" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Actions:" -ForegroundColor Cyan
        Write-Host "  start     - Start services and initialize databases (default)" -ForegroundColor White
        Write-Host "  stop      - Stop all services" -ForegroundColor White
        Write-Host "  status    - Show service status" -ForegroundColor White
        Write-Host "  logs      - Show recent logs" -ForegroundColor White
        Write-Host "  test      - Run API tests" -ForegroundColor White
        Write-Host "  restart   - Restart all services" -ForegroundColor White
        Write-Host "  cleanup   - Delete all containers and data" -ForegroundColor White
        Write-Host ""
        Write-Host "Examples:" -ForegroundColor Cyan
        Write-Host "  .\quickstart.ps1 start" -ForegroundColor Gray
        Write-Host "  .\quickstart.ps1 stop" -ForegroundColor Gray
        Write-Host "  .\quickstart.ps1 status" -ForegroundColor Gray
    }
}

Write-Host ""
