#!/bin/bash
# PyTorch 2.10.0a0 SM120 Installation Script for RTX 50-series GPUs
# 
# This script installs PyTorch with native SM 12.0 (Blackwell) support
# for NVIDIA RTX 5090, 5080, 5070 Ti, and 5070 GPUs

set -e

echo "=========================================="
echo "PyTorch 2.10.0a0 SM120 Installer"
echo "RTX 50-series GPU Support"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Detected Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.12" ]]; then
    echo "WARNING: This wheel was built for Python 3.12"
    echo "Your Python version: $PYTHON_VERSION"
    echo "Installation may fail or produce compatibility issues."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. GPU functionality may not be available."
    echo ""
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install PyTorch wheel
echo ""
echo "Installing PyTorch 2.10.0a0 with SM120 support..."
pip install torch_sm120.whl --force-reinstall

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Verifying installation..."
python3 << 'PYEOF'
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {'.'.join(map(str, torch.cuda.get_device_capability(0)))}")
    print("")
    print("Testing GPU tensor operations...")
    x = torch.rand(5, 3).cuda()
    print(f"Test tensor on GPU: {x.device}")
    print("SUCCESS! PyTorch is working with your GPU.")
else:
    print("WARNING: CUDA is not available. GPU acceleration disabled.")
PYEOF

echo ""
echo "Installation verified successfully!"
echo ""
