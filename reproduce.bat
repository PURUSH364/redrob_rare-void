@echo off
REM reproduce.bat - End-to-end pipeline for India Data Challenge

setlocal

echo Starting RARE_VOID pipeline...

REM Set UTF-8 encoding to prevent Windows cp1252 crashes
set PYTHONIOENCODING=utf-8
chcp 65001 >nul

REM 0. Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat

REM 0b. Install dependencies
echo Installing dependencies...
pip install -q -r requirements.txt
pip install -q pytest

REM 1. Run unit tests
python -m pytest tests/test_pipeline.py -v
if %errorlevel% neq 0 exit /b %errorlevel%

REM 2. Run the pipeline end-to-end
python -m src.main --candidates data/candidates.jsonl --out output/team_rare_void.csv
if %errorlevel% neq 0 exit /b %errorlevel%

REM 3. Validate the final submission format
python validate_submission.py output/team_rare_void.csv
if %errorlevel% neq 0 exit /b %errorlevel%

echo Pipeline completed successfully!
endlocal
