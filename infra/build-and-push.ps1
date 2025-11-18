# build-and-push.ps1
# Script para construir y subir todas las imágenes al registry local

param(
    [string]$Registry = "localhost:5000",
    [string]$Version = "latest"
)

Write-Host "=== Building and Pushing MLOps Pipeline Images ===" -ForegroundColor Cyan

# Verificar que registry está corriendo
Write-Host "Checking local registry..." -ForegroundColor Yellow
$registryRunning = docker ps --filter "name=registry" --filter "status=running" --format "{{.Names}}"
if (-not $registryRunning) {
    Write-Host "Starting local registry on port 5000..." -ForegroundColor Yellow
    docker run -d -p 5000:5000 --restart=always --name registry registry:2
    Start-Sleep -Seconds 3
}

$services = @(
    @{name="mlflow-server"; path="../mlflow-server"},
    @{name="llm-connector"; path="../llm-connector"},
    @{name="sklearn-model"; path="../sklearn-model"},
    @{name="cnn-image"; path="../cnn-image"},
    @{name="gradio-frontend"; path="../gradio-frontend"}
)

foreach ($service in $services) {
    $name = $service.name
    $path = $service.path
    
    Write-Host "`nBuilding $name..." -ForegroundColor Green
    # CORREGIDO: Usar ${} para delimitar variables
    docker build -t "${name}:${Version}" $path
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR building $name" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Tagging $name..." -ForegroundColor Green
    docker tag "${name}:${Version}" "${Registry}/${name}:${Version}"
    
    Write-Host "Pushing $name..." -ForegroundColor Green
    docker push "${Registry}/${name}:${Version}"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR pushing $name" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n=== All images built and pushed successfully! ===" -ForegroundColor Cyan
Write-Host "`nImages available in registry:" -ForegroundColor Yellow
foreach ($service in $services) {
    Write-Host "  - ${Registry}/$($service.name):${Version}" -ForegroundColor White
}
