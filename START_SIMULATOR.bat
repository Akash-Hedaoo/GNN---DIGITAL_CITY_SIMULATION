@echo off
REM Pune GNN City Simulator - Start Script

echo.
echo ============================================
echo   Pune GNN City Simulator - Starting...
echo ============================================
echo.

REM Get the directory where this script is located
cd /d "%~dp0"

echo [1/3] Starting Backend Server...
echo        Flask running on http://localhost:5000
echo.

REM Start backend in a new window
start "Backend Server" cmd /k "python -m flask --app backend.app run --port 5000"

REM Wait for backend to start
timeout /t 3 /nobreak

echo [2/3] Starting Frontend Server...
echo        HTTP Server running on http://localhost:8000
echo.

REM Start frontend in a new window
start "Frontend Server" cmd /k "cd frontend && python -m http.server 8000"

REM Wait for frontend to start
timeout /t 2 /nobreak

echo [3/3] Opening Application...
echo.

REM Open browser (works on most Windows systems)
start http://localhost:8000/index.html

echo.
echo ============================================
echo   Application started!
echo ============================================
echo.
echo Backend:  http://localhost:5000
echo Frontend: http://localhost:8000
echo.
echo Servers running in background windows.
echo Press Ctrl+C in each window to stop.
echo.
pause
