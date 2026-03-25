# MADDPG Adversarial Robustness - Docker Setup
# Use CUDA-enabled PyTorch base image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    nvidia-utils-525 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages for experiments
RUN pip install --no-cache-dir \
    torch-geometric \
    jupyter \
    ipython \
    tensorboard \
    wandb \
    GPUtil \
    psutil

# Copy framework code
COPY src/ ./src/
COPY standalone_experiment_runner.py .
COPY experiment_config.json .
COPY test_standalone.sh .
COPY analyze_topology.py .
COPY analyze_traffic_matrix.py .

# Copy documentation  
COPY README.md .
COPY THESIS_GUIDANCE.md .

# Create results directory with proper permissions
RUN mkdir -p /workspace/data/results /workspace/data/models /workspace/logs \
    && chmod -R 777 /workspace/data /workspace/logs

# Set environment variables
ENV PYTHONPATH=/workspace/src/maddpg_clean:/workspace/src/attack_framework
ENV CUDA_VISIBLE_DEVICES=0

# Create entrypoint script
RUN echo '#!/bin/bash\n\
echo "🚀 MADDPG Adversarial Robustness Container"\n\
echo "========================================"\n\
echo "🐍 Python: $(python --version)"\n\
echo "🔥 PyTorch: $(python -c \"import torch; print(torch.__version__)\")"\n\
echo "🎯 CUDA available: $(python -c \"import torch; print(torch.cuda.is_available())\")"\n\
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then\n\
    echo "✅ GPU: $(python -c \"import torch; print(torch.cuda.get_device_name(0))\")"\n\
    echo "💾 GPU Memory: $(python -c \"import torch; print(f\"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB\")\")"\n\
else\n\
    echo "⚠️  CUDA not available - using CPU"\n\
fi\n\
echo "📁 Workspace: /workspace"\n\
echo "📊 Results: /workspace/data/results"\n\
echo ""\n\
echo "🎯 Available commands:"\n\
echo "  python test_standalone.sh           # Test framework"\n\
echo "  python standalone_experiment_runner.py --quick  # Quick test"\n\
echo "  python standalone_experiment_runner.py --gpu 0  # Full experiment"\n\
echo "  jupyter lab --ip=0.0.0.0 --allow-root         # Start Jupyter"\n\
echo ""\n\
exec "$@"\n\
' > /workspace/entrypoint.sh && chmod +x /workspace/entrypoint.sh

# Default command
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["bash"]