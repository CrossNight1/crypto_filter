#!/bin/bash

echo "========================================"
echo "    Crypto Filter Pro Launcher"
echo "========================================"
echo "1) Legacy Shiny App (Port 8000)"
echo "2) New Next.js App + FastAPI Backend"
echo "3) Just FastAPI Backend (Port 8000)"
echo "4) Just Next.js Frontend (Port 3000)"
echo "5) Stop All Running Services"
echo "========================================"
read -p "Enter your choice (1-5): " choice

function cleanup() {
    echo "Cleaning up any processes currently using ports 8000 and 3000..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    lsof -ti:3000 | xargs kill -9 2>/dev/null
    sleep 1
}

case $choice in
    1)
        cleanup
        echo "Starting Legacy Shiny App on Port 8000..."
        python -m shiny run app.py --reload --port 8000
        ;;
    2)
        cleanup
        echo "Starting FastAPI Backend on Port 8000..."
        uvicorn main:app --port 8000 &
        
        echo "Starting Next.js Frontend on Port 3000..."
        cd frontend && npm run dev
        ;;
    3)
        cleanup
        echo "Starting FastAPI Backend on Port 8000..."
        uvicorn main:app --reload --port 8000
        ;;
    4)
        cleanup
        echo "Starting Next.js Frontend on Port 3000..."
        cd frontend && npm run dev
        ;;
    5)
        cleanup
        echo "All services stopped."
        ;;
    *)
        echo "Invalid choice. Exiting."
        ;;
esac
