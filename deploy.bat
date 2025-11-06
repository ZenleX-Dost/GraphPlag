@echo off
REM Deployment script for GraphPlag (Windows)

echo ================================
echo GraphPlag Deployment Script
echo ================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not installed
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker Compose is not installed
    exit /b 1
)

echo [OK] Docker and Docker Compose are installed
echo.

REM Create necessary directories
echo Creating directories...
if not exist "cache\embeddings" mkdir cache\embeddings
if not exist "cache\sentences" mkdir cache\sentences
if not exist "logs" mkdir logs
if not exist "models" mkdir models
echo [OK] Directories created
echo.

REM Copy environment file if not exists
if not exist ".env" (
    echo Creating .env file from template...
    copy .env.example .env
    echo [WARNING] Please update .env with your configuration
    echo.
)

REM Build Docker image
echo Building Docker image...
docker-compose build
if errorlevel 1 (
    echo [ERROR] Docker build failed
    exit /b 1
)
echo [OK] Docker image built
echo.

REM Start services
echo Starting services...
docker-compose up -d
if errorlevel 1 (
    echo [ERROR] Failed to start services
    exit /b 1
)
echo [OK] Services started
echo.

REM Wait for service to be healthy
echo Waiting for service to be healthy...
timeout /t 10 /nobreak >nul

echo.
echo ================================
echo [OK] Deployment successful!
echo ================================
echo.
echo API is running at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo Health Check: http://localhost:8000/health
echo.
echo Useful commands:
echo   - View logs: docker-compose logs -f
echo   - Stop services: docker-compose down
echo   - Restart: docker-compose restart
echo.

pause
