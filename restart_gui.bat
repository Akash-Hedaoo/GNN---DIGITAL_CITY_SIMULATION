@echo off
cls
echo.
echo ===============================================
echo   Restarting Digital Twin City Simulation
echo ===============================================
echo.
echo [*] Stopping any running instances...
taskkill /F /IM streamlit.exe 2>nul

timeout /t 2 /nobreak >nul

echo [*] Activating virtual environment...
if exist "twin-city-env\Scripts\activate.bat" (
    call twin-city-env\Scripts\activate.bat
)

echo [*] Starting Streamlit application...
echo.
echo The application will open at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run streamlit_gui.py

pause
