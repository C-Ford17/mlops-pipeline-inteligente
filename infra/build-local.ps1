# build-local.ps1
# Script para construir imágenes localmente (sin registry)

param(
    [string]$Version = "latest"
)

Write-Host "=== Building MLOps Pipeline Images Locally ===" -ForegroundColor Cyan

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
    
    Write-Host "`nBuilding ${name}:${Version}..." -ForegroundColor Green
    docker build -t "${name}:${Version}" $path
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR building $name" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n=== All images built successfully! ===" -ForegroundColor Cyan
Write-Host "`nAvailable images:" -ForegroundColor Yellow
docker images | Select-String -Pattern "mlflow-server|llm-connector|sklearn-model|cnn-image|gradio-frontend"
