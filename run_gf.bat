@echo off
REM GraphPlag Scalable System - Quick Start
REM This script starts all Docker services for the scalable deployment

echo ========================================
echo GraphPlag Scalable System
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running or not installed
    echo Please start Docker Desktop and try again
    pause
    exit /b 1
)
echo [OK] Docker is running

echo.
echo [2/3] Starting services...
echo This will start: Redis, PostgreSQL, API, Celery Workers, and Flower
docker-compose -f docker-compose-fast.yml up -d

if errorlevel 1 (
    echo ERROR: Failed to start services
    pause
    exit /b 1
)

echo.
echo [3/3] Waiting for services to be ready...
timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo Services are READY!
echo ========================================
echo.
echo API Documentation:  http://localhost:8000/docs
echo API Root:           http://localhost:8000
echo Flower Dashboard:   http://localhost:5555
echo.
echo PostgreSQL:         localhost:5432
echo Redis:              localhost:6379
echo.
echo To stop services:   docker-compose -f docker-compose-fast.yml down
echo To view logs:       docker-compose -f docker-compose-fast.yml logs -f
echo.
echo Opening application in browser...
timeout /t 2 /nobreak >nul

REM Open main application
start http://localhost:8000

REM Also open API docs in another tab
timeout /t 1 /nobreak >nul
start http://localhost:8000/docs

echo.
echo Browser windows opened! Check your browser.
echo.
pause
