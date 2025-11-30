# Digital Twin City Simulation - Streamlit GUI Launcher
# Run this script to launch the application

Write-Host "🚦 Digital Twin City Simulation - Streamlit GUI" -ForegroundColor Cyan
Write-Host "=" -NoNewline; Write-Host ("=" * 50) -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
$venvPath = ".\twin-city-env\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "✅ Virtual environment found" -ForegroundColor Green
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & $venvPath
} else {
    Write-Host "⚠️  Virtual environment not found at $venvPath" -ForegroundColor Yellow
    Write-Host "Continuing without activation..." -ForegroundColor Yellow
}

Write-Host ""

# Check required files
Write-Host "Checking required files..." -ForegroundColor Yellow
$requiredFiles = @(
    "streamlit_gui.py",
    "gnn_model.py",
    "trained_gnn.pt",
    "city_graph.graphml"
)

$allFilesPresent = $true
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file (missing)" -ForegroundColor Red
        $allFilesPresent = $false
    }
}

# Check optional files
$optionalFiles = @("gnn_training_data.pkl")
foreach ($file in $optionalFiles) {
    if (Test-Path $file) {
        Write-Host "  ✅ $file (optional)" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  $file (optional, missing - some features limited)" -ForegroundColor Yellow
    }
}

Write-Host ""

if (-not $allFilesPresent) {
    Write-Host "❌ Some required files are missing!" -ForegroundColor Red
    Write-Host "Please ensure all required files are present before running." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "🚀 Launching Streamlit application..." -ForegroundColor Cyan
Write-Host ""
Write-Host "The application will open in your default browser." -ForegroundColor Yellow
Write-Host "If it doesn't open automatically, navigate to: http://localhost:8501" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

# Launch streamlit
streamlit run streamlit_gui.py
