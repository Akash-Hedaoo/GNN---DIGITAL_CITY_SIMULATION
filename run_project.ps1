# ============================================================
# GNN Digital Twin City Simulation - Project Runner
# ============================================================
# Run: .\run_project.ps1
# ============================================================

param(
    [switch]$Server,       # Start web server (default)
    [switch]$Train,        # Train the GNN model
    [switch]$Test,         # Test the trained model
    [switch]$Manual,       # Interactive manual testing
    [switch]$Help          # Show help
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot "twin-city-env"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"

function Show-Banner {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "   GNN Digital Twin City Simulation" -ForegroundColor Yellow
    Write-Host "   AI-Powered Traffic Prediction System" -ForegroundColor White
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Show-Help {
    Show-Banner
    Write-Host "USAGE:" -ForegroundColor Green
    Write-Host "  .\run_project.ps1              - Start web server (default)"
    Write-Host "  .\run_project.ps1 -Server      - Start web server"
    Write-Host "  .\run_project.ps1 -Train       - Train the GNN model"
    Write-Host "  .\run_project.ps1 -Test        - Test the trained model"
    Write-Host "  .\run_project.ps1 -Manual      - Interactive manual testing"
    Write-Host "  .\run_project.ps1 -Help        - Show this help message"
    Write-Host ""
    Write-Host "After starting the server, open: http://localhost:5000" -ForegroundColor Cyan
    Write-Host ""
}

function Test-VenvExists {
    return (Test-Path $PythonPath)
}

Set-Location $ProjectRoot

if ($Help) {
    Show-Help
    exit 0
}

if (-not (Test-VenvExists)) {
    Write-Host "ERROR: Virtual environment not found at: $VenvPath" -ForegroundColor Red
    Write-Host "Please create it first with: python -m venv twin-city-env" -ForegroundColor Yellow
    exit 1
}

if ($Train) {
    Show-Banner
    Write-Host "Starting GNN Model Training..." -ForegroundColor Green
    Write-Host "This may take ~23 minutes on RTX 3050" -ForegroundColor Yellow
    Write-Host ""
    & $PythonPath "train_model.py"
    exit 0
}

if ($Test) {
    Show-Banner
    Write-Host "Testing Trained GNN Model..." -ForegroundColor Green
    Write-Host ""
    & $PythonPath "test_trained_model.py"
    exit 0
}

if ($Manual) {
    Show-Banner
    Write-Host "Starting Interactive Manual Model Testing..." -ForegroundColor Green
    Write-Host ""
    & $PythonPath "manual_model_test.py"
    exit 0
}

# Default: Start web server
Show-Banner
Write-Host "Starting Web Server..." -ForegroundColor Green
Write-Host ""
Write-Host "Frontend: http://localhost:5000" -ForegroundColor Cyan
Write-Host "API:      http://localhost:5000/api/status" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press CTRL+C to stop the server" -ForegroundColor Gray
Write-Host ""
& $PythonPath "backend\app.py"
