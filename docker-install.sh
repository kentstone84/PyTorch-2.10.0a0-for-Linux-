#!/bin/bash
# ========================================================================
# Docker Installation Script for PyTorch RTX 5080 Optimization Stack
# ========================================================================

set -e

echo "========================================================================"
echo "PyTorch 2.10.0a0 + Triton 3.5.0 Docker Installation"
echo "RTX 50-series (SM 12.0 Blackwell) Optimization Stack"
echo "========================================================================"
echo ""

# -------------------------------------------------------------------------
# Step 1: Check Prerequisites
# -------------------------------------------------------------------------
echo "[1/5] Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ ERROR: Docker is not installed"
    echo "   Install from: https://docs.docker.com/engine/install/"
    exit 1
fi
echo "✓ Docker found: $(docker --version)"

# Check NVIDIA Docker Runtime
if ! docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi &> /dev/null; then
    echo "❌ ERROR: NVIDIA Docker runtime not available"
    echo "   Install nvidia-container-toolkit:"
    echo "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi
echo "✓ NVIDIA Docker runtime available"

# Check GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "✓ GPU detected: $GPU_NAME (Compute Capability: $COMPUTE_CAP)"

if [[ ! "$COMPUTE_CAP" =~ ^120 ]]; then
    echo "⚠️  WARNING: This build is optimized for SM 12.0 (RTX 50-series)"
    echo "   Your GPU has compute capability: $COMPUTE_CAP"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# -------------------------------------------------------------------------
# Step 2: Check for Wheel Files
# -------------------------------------------------------------------------
echo ""
echo "[2/5] Checking for wheel files..."

if [ ! -f "triton_sm120.whl" ]; then
    echo "❌ ERROR: triton_sm120.whl not found"
    echo "   Download from: https://github.com/kentstone84/PyTorch-2.10.0a0-for-Linux-/releases/latest"
    exit 1
fi
echo "✓ Found: triton_sm120.whl ($(du -h triton_sm120.whl | cut -f1))"

if [ ! -f "torch_sm120.whl" ]; then
    echo "❌ ERROR: torch_sm120.whl not found"
    echo "   Download from: https://github.com/kentstone84/PyTorch-2.10.0a0-for-Linux-/releases/latest"
    exit 1
fi
echo "✓ Found: torch_sm120.whl ($(du -h torch_sm120.whl | cut -f1))"

# -------------------------------------------------------------------------
# Step 3: Build Docker Image
# -------------------------------------------------------------------------
echo ""
echo "[3/5] Building Docker image..."
docker build -f Dockerfile.rtx5080-runtime -t pytorch-rtx5080:latest .

# -------------------------------------------------------------------------
# Step 4: Test Installation
# -------------------------------------------------------------------------
echo ""
echo "[4/5] Testing installation..."
docker run --rm --gpus all pytorch-rtx5080:latest python3 -c "
import torch
import triton

print('=' * 70)
print('Installation Test Results')
print('=' * 70)
print(f'PyTorch Version: {torch.__version__}')
print(f'Triton Version: {triton.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'Compute Capability: {\".\" .join(map(str, torch.cuda.get_device_capability(0)))}')

    # Test tensor operation
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f'GPU Tensor Test: ✓ PASSED')

print('=' * 70)
"

# -------------------------------------------------------------------------
# Step 5: Run Test Suite (Optional)
# -------------------------------------------------------------------------
echo ""
echo "[5/5] Running comprehensive test suite..."

if [ -f "test-triton-sm120.py" ]; then
    docker run --rm --gpus all -v $(pwd):/workspace pytorch-rtx5080:latest \
        python3 /workspace/test-triton-sm120.py
else
    echo "⚠️  test-triton-sm120.py not found, skipping comprehensive tests"
fi

# -------------------------------------------------------------------------
# Complete
# -------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "✓ Installation Complete!"
echo "========================================================================"
echo ""
echo "Docker image: pytorch-rtx5080:latest"
echo ""
echo "Quick Start:"
echo ""
echo "  # Run interactive Python session"
echo "  docker run --rm -it --gpus all pytorch-rtx5080:latest"
echo ""
echo "  # Run your training script"
echo "  docker run --rm --gpus all -v \$(pwd):/workspace pytorch-rtx5080:latest \\"
echo "    python3 /workspace/train.py"
echo ""
echo "  # Use Docker Compose"
echo "  docker-compose up -d"
echo "  docker-compose exec pytorch-rtx5080 python3"
echo ""
echo "========================================================================"
