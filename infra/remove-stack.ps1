# remove-stack.ps1
# Script para eliminar el stack de Swarm

param(
    [string]$StackName = "mlops",
    [switch]$RemoveVolumes = $false
)

Write-Host "=== Removing MLOps Stack from Swarm ===" -ForegroundColor Cyan

# Remover stack
Write-Host "Removing stack '$StackName'..." -ForegroundColor Yellow
docker stack rm $StackName

Write-Host "Waiting for services to shut down..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

if ($RemoveVolumes) {
    Write-Host "`nRemoving volumes..." -ForegroundColor Yellow
    docker volume rm mlops_minio-data mlops_sklearn-data mlops_sklearn-models mlops_cnn-data mlops_cnn-models 2>$null
}

Write-Host "`n=== Stack removed successfully! ===" -ForegroundColor Cyan
