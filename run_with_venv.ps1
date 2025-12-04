# PowerShell script to run Python scripts with the virtual environment
# Usage: .\run_with_venv.ps1 step1_acquire_larger_network.py

param(
    [Parameter(Mandatory=$true)]
    [string]$ScriptName
)

# Change to script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Activate venv and run script
& ".\.venv\Scripts\python.exe" $ScriptName

