#!/bin/zsh
# Crypto Market Radar - Environment Setup

# Move to the project directory
cd "$(dirname "$0")"

echo "🛠️ Starting Environment Setup..."

# 1. Create venv if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists."
fi

# 2. Install requirements
echo "📥 Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete!"
echo "👉 You can now launch the app using 'start_app.command'"
