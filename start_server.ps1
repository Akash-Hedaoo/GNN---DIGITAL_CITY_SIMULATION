# PowerShell script to start the Flask server
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting GNN Traffic Simulation Server" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Activate venv and run
& ".\.venv\Scripts\python.exe" app.py



