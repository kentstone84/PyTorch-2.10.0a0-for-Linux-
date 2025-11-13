#!/bin/bash
# ========================================================================
# Native Triton Build Script for SM 12.0 (Blackwell) + CUDA 13.0
# Run this directly on your system (no Docker required)
# ========================================================================

set -e

echo "======================================================================="
echo "Building Triton for SM 12.0 (Blackwell) with CUDA 13.0"
echo "System: 14900KS (24 cores), 128GB RAM, RTX 5080"
echo "======================================================================="

# Check prerequisites
echo ""
echo "Checking prerequisites..."

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA not found. Please install CUDA 13.0+"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
echo "✓ CUDA version: $CUDA_VERSION"

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✓ Python version: $PYTHON_VERSION"

# Check LLVM
if ! command -v llvm-config &> /dev/null && ! command -v llvm-config-18 &> /dev/null; then
    echo "WARNING: LLVM 18 not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y llvm-18 llvm-18-dev libz3-dev zlib1g-dev
fi

echo "✓ LLVM available"

# Create build directory
BUILD_DIR="$HOME/triton-build-sm120"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "======================================================================="
echo "Cloning Triton repository..."
echo "======================================================================="

if [ -d "triton" ]; then
    echo "Triton directory exists, pulling latest..."
    cd triton
    git pull
    cd ..
else
    git clone https://github.com/triton-lang/triton.git
fi

cd triton

# Get current commit
COMMIT_HASH=$(git rev-parse --short HEAD)
echo "Building from commit: $COMMIT_HASH"

# Create virtual environment
echo ""
echo "======================================================================="
echo "Setting up Python environment..."
echo "======================================================================="

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install numpy setuptools cmake wheel lit pybind11

echo ""
echo "======================================================================="
echo "Configuring build for SM 12.0..."
echo "======================================================================="

# Set environment variables
export TRITON_BUILD_WITH_CCACHE=1
export TRITON_BUILD_WITH_O1=1
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDACXX=/usr/local/cuda/bin/nvcc
export MAX_JOBS=24

# Verify PTXAS version
echo "PTXAS location: $TRITON_PTXAS_PATH"
$TRITON_PTXAS_PATH --version

echo ""
echo "Build configuration:"
echo "  TRITON_PTXAS_PATH: $TRITON_PTXAS_PATH"
echo "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "  CUDACXX: $CUDACXX"
echo "  MAX_JOBS: $MAX_JOBS"

echo ""
echo "======================================================================="
echo "Building Triton (this will take ~20-30 minutes with 24 cores)..."
echo "======================================================================="

cd python

# Clean previous builds
rm -rf build dist *.egg-info

# Build wheel
python setup.py bdist_wheel 2>&1 | tee ../build.log

echo ""
echo "======================================================================="
echo "Build complete!"
echo "======================================================================="

# Find the wheel
WHEEL_PATH=$(ls dist/triton-*.whl | head -1)

if [ -f "$WHEEL_PATH" ]; then
    echo "✓ Wheel created: $WHEEL_PATH"
    echo ""

    # Copy to easier location
    cp "$WHEEL_PATH" "$HOME/triton_sm120.whl"
    echo "✓ Copied to: $HOME/triton_sm120.whl"

    echo ""
    echo "======================================================================="
    echo "Installing and testing..."
    echo "======================================================================="

    pip install "$HOME/triton_sm120.whl" --force-reinstall

    # Test import
    python -c "import triton; print(f'Triton version: {triton.__version__}')"

    echo ""
    echo "======================================================================="
    echo "SUCCESS! Triton built successfully"
    echo "======================================================================="
    echo ""
    echo "Wheel location: $HOME/triton_sm120.whl"
    echo "Build log: $BUILD_DIR/triton/build.log"
    echo ""
    echo "To install on other systems:"
    echo "  pip install $HOME/triton_sm120.whl"
    echo ""
else
    echo "ERROR: Wheel not found in dist/"
    echo "Check build log: $BUILD_DIR/triton/build.log"
    exit 1
fi
