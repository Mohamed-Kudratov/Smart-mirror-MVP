@echo off
title Lookzi Virtual Try-On
echo ============================================================
echo   Lookzi Virtual Try-On
echo   Opening browser at http://127.0.0.1:7860
echo   Close this window to stop the app.
echo ============================================================
echo.

cd /d "%~dp0"
set PYTHONPATH=%~dp0
..\ComfyUI_windows_portable\python_embeded\python.exe app.py

echo.
pause
