#!/bin/zsh
# Crypto Market Radar - Quick Start
# This script closes itself after launching the app

# Move to the project directory
cd "$(dirname "$0")"

echo "📡 Launching Crypto Market Radar..."

# Check if venv exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "⚠️ Virtual environment not found. Please run setup_venv.sh first."
    read -k 1 -s "Press any key to exit..."
    exit 1
fi

# Run the app in the background
# This allows the terminal window to be closed if desired, 
# though usually it stays open to see logs.
python3 run.py
