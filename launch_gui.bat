@echo off
REM Digital Twin City Simulation - Streamlit GUI Launcher
REM Simple batch file to launch the application

echo.
echo ========================================================
echo   Digital Twin City Simulation - Streamlit GUI
echo ========================================================
echo.

REM Activate virtual environment
if exist "twin-city-env\Scripts\activate.bat" (
    echo [*] Activating virtual environment...
    call twin-city-env\Scripts\activate.bat
) else (
    echo [!] Virtual environment not found, continuing...
)

echo.
echo [*] Launching Streamlit application...
echo.
echo The application will open in your browser at:
echo     http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run streamlit_gui.py

pause
