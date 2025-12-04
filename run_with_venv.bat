@echo off
REM Batch script to run Python scripts with the virtual environment
REM Usage: run_with_venv.bat step1_acquire_larger_network.py

cd /d "%~dp0"
.venv\Scripts\python.exe %*

