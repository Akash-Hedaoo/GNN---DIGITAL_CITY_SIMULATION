# PowerShell script to activate the virtual environment
# Usage: . .\activate_venv.ps1

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Activate the virtual environment
& ".\.venv\Scripts\Activate.ps1"

Write-Host "✅ Virtual environment activated!" -ForegroundColor Green
Write-Host "Python location: $(Get-Command python).Source" -ForegroundColor Cyan
Write-Host "You can now run Python scripts directly." -ForegroundColor Yellow

