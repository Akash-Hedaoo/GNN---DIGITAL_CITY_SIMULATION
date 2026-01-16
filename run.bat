@echo off
echo Starting Neuro-Physical Digital Twin...

:: 1. Launch SUMO Backend (Persistent)
:: We use python -c to instantiate SumoManager and start it. 
:: Added keep-alive loop so the process doesn't exit (closing TraCI).
start "SUMO Backend" cmd /k python -c "from sumo_connector import SumoManager; import time; mgr=SumoManager(); mgr.start(); print('Backend Running - Do Not Close'); time.sleep(999999)"

timeout /t 5

:: 2. Launch Streamlit Frontend
echo Launching Dashboard...
streamlit run main_app.py
