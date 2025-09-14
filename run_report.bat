@echo off
setlocal

REM Run from this folder
cd /d "%~dp0"

REM Ensure venv exists
if not exist ".venv\Scripts\python.exe" (
  echo [setup] Creating virtual environment...
  py -3 -m venv .venv || (
    echo [error] Python launcher not found. Install Python from python.org and retry.
    exit /b 1
  )
)

REM Upgrade pip and install requirements
echo [setup] Installing dependencies (pandas, numpy, plotly)...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1
".venv\Scripts\python.exe" -m pip install -r requirements.txt >nul 2>&1 || (
  echo [error] Failed to install dependencies.
  exit /b 1
)

REM Run the report (no args -> opens file picker)
echo [run] Starting report generator...
".venv\Scripts\python.exe" -X utf8 -u lap_analysis.py --pdf %*

REM Try opening the generated HTML automatically
if exist "timing_report\index.pdf" (
  echo [open] Opening timing_report\index.pdf
  start "" "timing_report\index.pdf"
) else if exist "timing_report\index.html" (
  echo [open] Opening timing_report\index.html
  start "" "timing_report\index.html"
)

endlocal
