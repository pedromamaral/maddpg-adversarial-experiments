#!/bin/bash
# Fixed setup script for GPU servers without conda
# Works with system Python, pip, and virtualenv

echo "🔧 DIAGNOSING GPU SERVER ENVIRONMENT"
echo "===================================="

# Check what's available
echo "📋 Available Python versions:"
which python3 2>/dev/null && python3 --version || echo "❌ python3 not found"
which python 2>/dev/null && python --version || echo "❌ python not found" 
which python3.8 2>/dev/null && python3.8 --version || echo "❌ python3.8 not found"
which python3.9 2>/dev/null && python3.9 --version || echo "❌ python3.9 not found"
which python3.10 2>/dev/null && python3.10 --version || echo "❌ python3.10 not found"

echo ""
echo "📋 Available package managers:"
which pip3 2>/dev/null && echo "✅ pip3 found" || echo "❌ pip3 not found"
which pip 2>/dev/null && echo "✅ pip found" || echo "❌ pip not found"
which apt 2>/dev/null && echo "✅ apt found (Debian/Ubuntu)" || echo "❌ apt not found"
which yum 2>/dev/null && echo "✅ yum found (RHEL/CentOS)" || echo "❌ yum not found"
which dnf 2>/dev/null && echo "✅ dnf found (Fedora)" || echo "❌ dnf not found"

echo ""
echo "📋 CUDA Environment:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "❌ nvidia-smi not found"
which nvcc 2>/dev/null && nvcc --version | grep "release" || echo "❌ CUDA compiler not found"

echo ""
echo "📋 Virtual Environment Tools:"
which virtualenv 2>/dev/null && echo "✅ virtualenv found" || echo "❌ virtualenv not found"
which conda 2>/dev/null && echo "✅ conda found" || echo "❌ conda not found"

echo ""
echo "🔍 RECOMMENDATIONS:"
echo "=================="

# Determine best Python to use
if which python3.9 >/dev/null 2>&1; then
    PYTHON_CMD="python3.9"
    echo "✅ Will use: python3.9"
elif which python3.8 >/dev/null 2>&1; then
    PYTHON_CMD="python3.8"
    echo "✅ Will use: python3.8"
elif which python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
    echo "✅ Will use: python3"
else
    echo "❌ No suitable Python found. Install Python 3.8+ first."
    exit 1
fi

# Check if we can create virtual environments
if ! $PYTHON_CMD -m venv --help >/dev/null 2>&1; then
    echo "❌ venv module not available. Need to install python3-venv"
    echo ""
    echo "🛠️  INSTALLATION COMMANDS:"
    echo "========================"
    echo "# For Ubuntu/Debian:"
    echo "sudo apt update"
    echo "sudo apt install python3-pip python3-venv python3-dev"
    echo ""
    echo "# For RHEL/CentOS:"
    echo "sudo yum install python3-pip python3-venv python3-devel"
    echo ""
    echo "# For Fedora:"
    echo "sudo dnf install python3-pip python3-venv python3-devel"
    exit 1
fi

echo "✅ Virtual environment support available"
echo "📝 Creating fixed setup script..."

# Create the fixed setup script
cat > setup_fixed.sh << 'EOF'
#!/bin/bash
# GPU Server Setup Script (Fixed for no conda)

set -e  # Exit on error

echo "🚀 Setting up MADDPG Environment (GPU Server - No Conda)"
echo "========================================================"

# Detect Python
if which python3.9 >/dev/null 2>&1; then
    PYTHON_CMD="python3.9"
elif which python3.8 >/dev/null 2>&1; then
    PYTHON_CMD="python3.8" 
elif which python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
else
    echo "❌ No Python 3.8+ found. Please install first."
    exit 1
fi

echo "✅ Using: $PYTHON_CMD ($($PYTHON_CMD --version))"

# Create virtual environment
echo "📦 Creating virtual environment..."
$PYTHON_CMD -m venv maddpg_env
source maddpg_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch with CUDA
echo "🔥 Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install torch-geometric (requires torch first)
echo "📊 Installing torch-geometric..."  
pip install torch-geometric

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install numpy scipy pandas matplotlib seaborn

# Install ML/experiment dependencies
echo "🧪 Installing ML dependencies..."
pip install scikit-learn tensorboard wandb tqdm

# Install network analysis
echo "🌐 Installing network analysis..."
pip install networkx

# Install development tools
echo "🛠️  Installing development tools..."
pip install jupyter ipython

# Install system monitoring
echo "📈 Installing monitoring tools..."
pip install psutil GPUtil

# Install additional utilities
echo "🔧 Installing utilities..."
pip install pyyaml configargparse h5py rich colorama click

# Test CUDA availability
echo "🧪 Testing CUDA availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  CUDA not available - will use CPU')
"

# Create activation script
echo "📝 Creating activation script..."
cat > activate_env.sh << 'ACTIVATE_EOF'
#!/bin/bash
# Activate the MADDPG environment

if [ -d "maddpg_env" ]; then
    source maddpg_env/bin/activate
    echo "✅ MADDPG environment activated"
    echo "🐍 Python: $(which python)"
    echo "📦 Packages installed: $(pip list | wc -l) packages"
else
    echo "❌ Environment not found. Run setup_fixed.sh first."
fi
ACTIVATE_EOF

chmod +x activate_env.sh

# Create test script
echo "📝 Creating test script..."
cat > test_setup.py << 'TEST_EOF'
#!/usr/bin/env python3
"""
Test the setup and verify all components work
"""

print("🧪 Testing MADDPG Framework Setup")
print("=" * 40)

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch error: {e}")

try:
    import torch_geometric
    print(f"✅ PyTorch Geometric: {torch_geometric.__version__}")
except Exception as e:
    print(f"❌ PyTorch Geometric error: {e}")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except Exception as e:
    print(f"❌ NumPy error: {e}")

try:
    import matplotlib
    print(f"✅ Matplotlib: {matplotlib.__version__}")
except Exception as e:
    print(f"❌ Matplotlib error: {e}")

try:
    import networkx as nx
    print(f"✅ NetworkX: {nx.__version__}")
except Exception as e:
    print(f"❌ NetworkX error: {e}")

print("\n🎯 Testing basic functionality...")

try:
    # Test tensor creation
    x = torch.randn(5, 5)
    if torch.cuda.is_available():
        x_gpu = x.cuda()
        print("✅ GPU tensor operations work")
    else:
        print("⚠️  Using CPU (CUDA not available)")
    
    # Test network creation
    G = nx.erdos_renyi_graph(10, 0.3)
    print(f"✅ Network creation works ({G.number_of_nodes()} nodes)")
    
    print("\n🎉 All tests passed! Setup successful.")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")

TEST_EOF

echo ""
echo "✅ Setup Complete!"
echo "=================="
echo "📁 Files created:"
echo "   • setup_fixed.sh     - This setup script"
echo "   • activate_env.sh    - Environment activation"  
echo "   • test_setup.py      - Installation test"
echo "   • maddpg_env/        - Virtual environment"
echo ""
echo "🎯 Next steps:"
echo "   1. source activate_env.sh"
echo "   2. python test_setup.py"
echo "   3. Continue with MADDPG experiments"

EOF

chmod +x setup_fixed.sh

echo ""
echo "✅ Diagnostic complete! Fixed setup script created."
echo "🎯 Run: ./setup_fixed.sh"