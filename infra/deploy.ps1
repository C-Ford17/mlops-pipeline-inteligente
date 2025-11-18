# deploy.ps1
# Script maestro para deploy completo

param(
    [ValidateSet('build', 'deploy', 'rebuild', 'remove', 'logs', 'scale')]
    [string]$Action = "deploy",
    [string]$Service = "",
    [int]$Replicas = 1
)

$StackName = "mlops"

switch ($Action) {
    "build" {
        Write-Host "Building and pushing images..." -ForegroundColor Cyan
        .\build-and-push.ps1
    }
    
    "deploy" {
        Write-Host "Deploying stack..." -ForegroundColor Cyan
        .\deploy-stack.ps1 -StackName $StackName
    }
    
    "rebuild" {
        Write-Host "Rebuilding and redeploying..." -ForegroundColor Cyan
        .\build-and-push.ps1
        .\deploy-stack.ps1 -StackName $StackName
    }
    
    "remove" {
        Write-Host "Removing stack..." -ForegroundColor Cyan
        .\remove-stack.ps1 -StackName $StackName
    }
    
    "logs" {
        if ($Service -eq "") {
            Write-Host "Available services:" -ForegroundColor Yellow
            docker stack services $StackName
        } else {
            docker service logs "${StackName}_${Service}" -f
        }
    }
    
    "scale" {
        if ($Service -eq "") {
            Write-Host "ERROR: Service name required" -ForegroundColor Red
            Write-Host "Usage: .\deploy.ps1 scale -Service llm-connector -Replicas 3" -ForegroundColor Yellow
        } else {
            docker service scale "${StackName}_${Service}=$Replicas"
        }
    }
}
