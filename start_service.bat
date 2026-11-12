@echo off
REM Start AIG Companion Speech Synthesis Microservice
cd /d "%~dp0"
set PORT=8880
set HOST=0.0.0.0
set MODEL_DIR=%~dp0models\tts\supertonic3
echo Starting AIG TTS Service on http://127.0.0.1:%PORT%...
if exist "C:\Users\Andrew\AppData\Local\Python\bin\python.exe" (
    "C:\Users\Andrew\AppData\Local\Python\bin\python.exe" tts_service.py
) else (
    python tts_service.py
)
pause
