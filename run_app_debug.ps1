# Debug script to run the Flask app with visible output
Write-Host "Starting Flask Application..." -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Change to script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Run the app
& ".\venv\Scripts\python.exe" run_app.py

# Keep window open if there's an error
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nApplication exited with error code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

