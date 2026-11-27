@echo off
setlocal
cd /d "%~dp0"
title CineSense AI

rem A .venv copied from another folder or PC is broken - detect it and rebuild.
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
  if errorlevel 1 (
    echo Found a broken virtual environment. Rebuilding it...
    rmdir /s /q ".venv"
  )
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3.12 -m venv .venv >nul 2>&1
  if errorlevel 1 python -m venv .venv
  if errorlevel 1 (
    echo.
    echo Could not create the environment. Install Python 3.12 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during setup, then run this file again.
    pause
    exit /b 1
  )
)

echo Installing requirements (first run takes a few minutes)...
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
if errorlevel 1 (
  echo.
  echo Installing requirements failed. Check your internet connection and try again.
  pause
  exit /b 1
)

echo Starting CineSense AI - your browser will open shortly. Press Ctrl+C here to stop.
".venv\Scripts\python.exe" -m streamlit run app.py
pause
