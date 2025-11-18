# deploy-stack.ps1
# Script para deploy del stack en Docker Swarm

param(
    [string]$StackName = "mlops",
    [string]$ComposeFile = "swarm-stack.yml"
)

Write-Host "=== Deploying MLOps Stack to Swarm ===" -ForegroundColor Cyan

# Verificar que Swarm está inicializado
Write-Host "Checking Swarm status..." -ForegroundColor Yellow
$swarmActive = docker info | Select-String "Swarm: active"
if (-not $swarmActive) {
    Write-Host "Initializing Docker Swarm..." -ForegroundColor Yellow
    docker swarm init
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to initialize Swarm" -ForegroundColor Red
        exit 1
    }
}

# Verificar que .env existe
if (-not (Test-Path ".env")) {
    Write-Host "ERROR: .env file not found!" -ForegroundColor Red
    Write-Host "Create .env with GOOGLE_API_KEY and GOOGLE_MODEL" -ForegroundColor Yellow
    exit 1
}

# Cargar variables de entorno desde .env
Write-Host "Loading environment variables from .env..." -ForegroundColor Yellow
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        $name = $matches[1]
        $value = $matches[2]
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
        Write-Host "  Loaded: $name" -ForegroundColor Green
    }
}

# Verificar variables críticas
$googleApiKey = [Environment]::GetEnvironmentVariable("GOOGLE_API_KEY", "Process")
if (-not $googleApiKey) {
    Write-Host "ERROR: GOOGLE_API_KEY not found in .env" -ForegroundColor Red
    exit 1
}

# Deploy stack (las variables se heredan del proceso)
Write-Host "`nDeploying stack '$StackName'..." -ForegroundColor Green
docker stack deploy -c $ComposeFile $StackName

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== Stack deployed successfully! ===" -ForegroundColor Cyan
    Write-Host "`nWaiting for services to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    
    Write-Host "`nService status:" -ForegroundColor Yellow
    docker stack services $StackName
    
    Write-Host "`nAccess points:" -ForegroundColor Cyan
    Write-Host "- Gradio UI:     http://localhost:7860" -ForegroundColor White
    Write-Host "- MLflow UI:     http://localhost:5000" -ForegroundColor White
    Write-Host "- MinIO Console: http://localhost:9001" -ForegroundColor White
    
    Write-Host "`nTo view logs:" -ForegroundColor Yellow
    Write-Host "  docker service logs ${StackName}_llm-connector -f" -ForegroundColor White
} else {
    Write-Host "ERROR: Stack deployment failed" -ForegroundColor Red
    exit 1
}
