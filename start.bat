@echo off
echo ========================================
echo     Crypto Filter Pro Launcher (Windows)
echo ========================================

if not exist .venv (
    echo Error: .venv not found. Creating it...
    python -m venv .venv
    call .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    echo Activating virtual environment...
    call .venv\Scripts\activate
)

echo Select Frontend Version:
echo 1) Shiny Python (Legacy)
echo 2) Next.js (New)
set choice=1
set /p choice="Enter choice [1 or 2] (default 1): "

if "%choice%"=="1" (
    echo Starting Crypto Filter Shiny App on Port 8000...
    python -m shiny run app.py --reload --port 8000 --launch-browser
) else (
    echo Starting Backend (FastAPI) on Port 8000...
    start /B python main.py
    
    echo Starting Frontend (Next.js) on Port 3000...
    cd frontend
    if not exist node_modules (
        echo Installing frontend dependencies...
        call npm install
    )
    call npm run dev
)
pause
