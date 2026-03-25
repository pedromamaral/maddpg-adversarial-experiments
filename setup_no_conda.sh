#!/bin/bash
# Alternative setup for GPU servers without conda
# Use this if the main setup_environment.sh fails

set -e

echo "🚀 GPU Server Setup (No Conda Required)"
echo "======================================="

# Detect Python
if which python3.9 >/dev/null 2>&1; then
    PYTHON_CMD="python3.9"
elif which python3.8 >/dev/null 2>&1; then
    PYTHON_CMD="python3.8"
elif which python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
else
    echo "❌ No Python 3.8+ found."
    echo "🛠️  Install with: sudo apt install python3-pip python3-venv"
    exit 1
fi

echo "✅ Using: $PYTHON_CMD ($($PYTHON_CMD --version))"

# Create virtual environment
echo "📦 Creating virtual environment..."
$PYTHON_CMD -m venv maddpg_env
source maddpg_env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements from requirements.txt
echo "📚 Installing packages from requirements.txt..."
pip install -r requirements.txt

# Create activation script
cat > activate_env.sh << 'EOF'
#!/bin/bash
if [ -d "maddpg_env" ]; then
    source maddpg_env/bin/activate
    echo "✅ MADDPG environment activated"
else
    echo "❌ Run setup_no_conda.sh first"
fi
EOF
chmod +x activate_env.sh

echo "✅ Setup complete!"
echo "🎯 Next: source activate_env.sh && python test_standalone.sh"