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

# Copy source code
COPY src/ ./src/
COPY standalone_experiment_runner.py .
COPY experiment_config.json .
COPY pyproject.toml .

# Install the package in editable mode
RUN pip install -e .

# Create directories for results and models
RUN mkdir -p /workspace/results /workspace/models /workspace/logs

# Set Python path
ENV PYTHONPATH=/workspace/src:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Default command runs the full experiment
CMD ["python", "standalone_experiment_runner.py", "--config", "experiment_config.json"]
