#!/bin/bash

echo "========================================"
echo "    Crypto Filter Pro Launcher"
echo "========================================"

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Error: .venv not found. Please create it first."
    exit 1
fi

function cleanup() {
    echo "Cleaning up any processes currently using port 8000 or 3000..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    lsof -ti:3000 | xargs kill -9 2>/dev/null
    sleep 1
}

cleanup

echo "Select Frontend Version:"
echo "1) Shiny Python (Legacy)"
echo "2) Next.js (New)"
read -p "Enter choice [1 or 2] (default 1): " choice

if [ -z "$choice" ]; then
    choice="1"
fi

if [ "$choice" == "1" ]; then
    echo "Starting Crypto Filter Shiny App (Legacy) on Port 8000..."
    python -m shiny run app.py --reload --port 8000 --launch-browser
else
    echo "Starting Backend (FastAPI) on Port 8000..."
    python main.py &
    BACKEND_PID=$!
    
    echo "Starting Frontend (Next.js) on Port 3000..."
    cd frontend
    if [ ! -d "node_modules" ]; then
        echo "Installing frontend dependencies..."
        npm install
    fi
    npm run dev &
    FRONTEND_PID=$!
    
    # Handle termination
    trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM
    
    # Wait for both processes
    wait $BACKEND_PID
    wait $FRONTEND_PID
fi
