#!/bin/bash
# ========================================================================
# Native Triton Build Script for SM 12.0 (Blackwell) - Python 3.13
# ========================================================================
# This script builds Triton natively with CUDA 13.0 PTXAS for SM 12.0 support
# Produces: triton-*-cp313-cp313-linux_x86_64.whl
# ========================================================================

set -e

echo "========================================================================"
echo "Triton Native Build Script for SM 12.0 (Blackwell) - Python 3.13"
echo "========================================================================"
echo ""

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
BUILD_DIR="$HOME/triton-build-sm120-py313"
REPO_URL="https://github.com/triton-lang/triton.git"
REPO_BRANCH="main"

# Check Python 3.13
if ! command -v python3.13 &> /dev/null; then
    echo "ERROR: Python 3.13 not found"
    echo "Install with: sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.13 python3.13-dev python3.13-venv"
    exit 1
fi

PYTHON_VERSION=$(python3.13 --version | awk '{print $2}')
echo "Using Python: $PYTHON_VERSION"
echo "Build directory: $BUILD_DIR"
echo ""

# -------------------------------------------------------------------------
# Check CUDA 13.0 PTXAS
# -------------------------------------------------------------------------
if [ ! -f "/usr/local/cuda/bin/ptxas" ]; then
    echo "ERROR: /usr/local/cuda/bin/ptxas not found"
    echo "Ensure CUDA 13.0+ is installed"
    exit 1
fi

PTXAS_VERSION=$(/usr/local/cuda/bin/ptxas --version | grep release | awk '{print $6}' | cut -d, -f1)
echo "Using PTXAS version: $PTXAS_VERSION"

if [[ ! "$PTXAS_VERSION" =~ ^(13\.|14\.) ]]; then
    echo "WARNING: PTXAS version $PTXAS_VERSION may not support SM 12.0"
    echo "CUDA 13.0+ recommended for Blackwell support"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

# -------------------------------------------------------------------------
# Setup Build Directory
# -------------------------------------------------------------------------
echo "[1/6] Setting up build directory..."

if [ -d "$BUILD_DIR" ]; then
    echo "Build directory exists. Removing..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# -------------------------------------------------------------------------
# Clone Triton Repository
# -------------------------------------------------------------------------
echo ""
echo "[2/6] Cloning Triton repository..."

git clone --branch "$REPO_BRANCH" --depth 1 "$REPO_URL" triton

cd triton

echo "Repository cloned successfully"
echo "Latest commit: $(git log -1 --oneline)"

# -------------------------------------------------------------------------
# Check Repository Structure
# -------------------------------------------------------------------------
echo ""
echo "[3/6] Detecting repository structure..."

if [ -f "setup.py" ]; then
    echo "✓ Found setup.py at root (new Triton structure)"
    SETUP_DIR="."
elif [ -f "python/setup.py" ]; then
    echo "✓ Found setup.py in python/ (old Triton structure)"
    SETUP_DIR="python"
else
    echo "ERROR: setup.py not found!"
    ls -la
    exit 1
fi

# -------------------------------------------------------------------------
# Configure Build Environment
# -------------------------------------------------------------------------
echo ""
echo "[4/6] Configuring build environment..."

export TRITON_BUILD_WITH_CCACHE=1
export TRITON_BUILD_WITH_O1=1
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDACXX=/usr/local/cuda/bin/nvcc
export MAX_JOBS=${MAX_JOBS:-24}

echo "✓ TRITON_PTXAS_PATH=$TRITON_PTXAS_PATH"
echo "✓ TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "✓ MAX_JOBS=$MAX_JOBS"

# -------------------------------------------------------------------------
# Build Triton
# -------------------------------------------------------------------------
echo ""
echo "[5/6] Building Triton with Python 3.13..."
echo "This will take 10-30 minutes..."
echo ""

cd "$SETUP_DIR"

# Create virtual environment with Python 3.13
python3.13 -m venv "$BUILD_DIR/venv"
source "$BUILD_DIR/venv/bin/activate"

# Install build dependencies
pip install --upgrade pip setuptools wheel
pip install numpy pyyaml cmake ninja lit

# Build with verbose output
python setup.py bdist_wheel 2>&1 | tee "$BUILD_DIR/triton-build.log"

# -------------------------------------------------------------------------
# Copy Wheel File
# -------------------------------------------------------------------------
echo ""
echo "[6/6] Copying wheel file..."

WHEEL_FILE=$(ls dist/triton-*cp313*.whl | head -1)

if [ ! -f "$WHEEL_FILE" ]; then
    echo "ERROR: Wheel file not found!"
    echo "Expected: dist/triton-*cp313*.whl"
    ls -la dist/
    exit 1
fi

WHEEL_NAME="triton_sm120_cp313.whl"
cp "$WHEEL_FILE" "$BUILD_DIR/$WHEEL_NAME"

echo "✓ Wheel copied to: $BUILD_DIR/$WHEEL_NAME"

# -------------------------------------------------------------------------
# Verify Build
# -------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "Build Complete!"
echo "========================================================================"
echo ""
echo "Wheel file: $BUILD_DIR/$WHEEL_NAME"
echo "Size: $(du -h "$BUILD_DIR/$WHEEL_NAME" | cut -f1)"
echo ""
echo "Build log: $BUILD_DIR/triton-build.log"
echo ""
echo "To install:"
echo "  pip install $BUILD_DIR/$WHEEL_NAME"
echo ""
echo "To verify:"
echo "  python3.13 -c 'import triton; print(triton.__version__)'"
echo ""
echo "========================================================================"
