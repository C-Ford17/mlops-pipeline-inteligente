# run-tests.ps1
# Script para ejecutar todos los tests localmente

Write-Host "=== Running MLOps Pipeline Tests ===" -ForegroundColor Cyan

$services = @("llm-connector", "sklearn-model", "cnn-image", "gradio-frontend")
$allPassed = $true

foreach ($service in $services) {
    Write-Host "`n🧪 Testing $service..." -ForegroundColor Yellow
    
    cd $service
    
    # Instalar dependencias de test
    if (Test-Path "requirements-dev.txt") {
        pip install -q -r requirements-dev.txt
    } else {
        pip install -q pytest pytest-cov httpx
    }
    
    # Ejecutar tests
    pytest tests/ -v --cov=app --cov-report=term-missing
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Tests failed for $service" -ForegroundColor Red
        $allPassed = $false
    } else {
        Write-Host "✅ Tests passed for $service" -ForegroundColor Green
    }
    
    cd ..
}

if ($allPassed) {
    Write-Host "`n✅ All tests passed!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n❌ Some tests failed" -ForegroundColor Red
    exit 1
}
