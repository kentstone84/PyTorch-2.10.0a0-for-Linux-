#!/bin/bash
# ========================================================================
# FIXED: Triton Build Script for SM 12.0 (Blackwell) + CUDA 13.0
# Handles interrupted clones and uses pip install method
# ========================================================================

set -e

echo "======================================================================="
echo "FIXED Triton Build for SM 12.0 (Blackwell)"
echo "System: 14900KS (24 cores), 128GB RAM, RTX 5080"
echo "======================================================================="

# Build directory
BUILD_DIR="$HOME/triton-build-sm120"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "======================================================================="
echo "Step 1: Ensuring Triton repository is complete..."
echo "======================================================================="

# Remove incomplete clone if it exists
if [ -d "triton" ]; then
    echo "Removing existing triton directory..."
    rm -rf triton
fi

# Fresh clone with progress
echo "Cloning Triton (this may take a few minutes)..."
git clone --depth 1 https://github.com/triton-lang/triton.git

# Verify clone succeeded
if [ ! -d "triton" ]; then
    echo "ERROR: Clone failed. triton directory not found."
    exit 1
fi

cd triton

# Get commit info
COMMIT_HASH=$(git rev-parse --short HEAD)
echo "✓ Clone complete! Commit: $COMMIT_HASH"

# Check repository structure (Triton changed structure recently)
if [ -f "setup.py" ]; then
    echo "✓ Found setup.py at root (new Triton structure)"
    SETUP_DIR="."
elif [ -f "python/setup.py" ]; then
    echo "✓ Found setup.py in python/ (old Triton structure)"
    SETUP_DIR="python"
else
    echo "ERROR: setup.py not found!"
    echo "Repository root contents:"
    ls -la
    echo ""
    echo "Checking python/ directory:"
    ls -la python/ 2>/dev/null || echo "python/ directory does not exist"
    exit 1
fi

echo ""
echo "======================================================================="
echo "Step 2: Setting up Python environment..."
echo "======================================================================="

# Create fresh venv
if [ -d "venv" ]; then
    rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing build dependencies..."
pip install numpy cmake lit pybind11

echo ""
echo "======================================================================="
echo "Step 3: Configuring build for SM 12.0..."
echo "======================================================================="

# Set environment variables
export TRITON_BUILD_WITH_CCACHE=1
export TRITON_BUILD_WITH_O1=1
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDACXX=/usr/local/cuda/bin/nvcc
export MAX_JOBS=24

# Verify CUDA tools
echo "CUDA Configuration:"
echo "  nvcc: $(which nvcc)"
nvcc --version | grep "release"
echo "  ptxas: $TRITON_PTXAS_PATH"
$TRITON_PTXAS_PATH --version | grep -E "(release|Build)"

echo ""
echo "Build Environment:"
echo "  TRITON_PTXAS_PATH: $TRITON_PTXAS_PATH"
echo "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "  CUDACXX: $CUDACXX"
echo "  MAX_JOBS: $MAX_JOBS"

echo ""
echo "======================================================================="
echo "Step 4: Building Triton (20-30 mins with 24 cores)..."
echo "======================================================================="
echo "Your 14900KS is about to GO BRRRR! 🔥"
echo ""

# Navigate to setup.py location
cd "$SETUP_DIR"

echo "Working directory: $(pwd)"

# Clean any previous builds
rm -rf build dist *.egg-info

# Build with verbose output
echo "Starting build at $(date)..."
python setup.py bdist_wheel 2>&1 | tee "$BUILD_DIR/triton-build.log"

echo ""
echo "Build finished at $(date)"

echo ""
echo "======================================================================="
echo "Step 5: Verifying wheel..."
echo "======================================================================="

# Check if wheel was created
if ls dist/triton-*.whl 1> /dev/null 2>&1; then
    WHEEL_PATH=$(ls dist/triton-*.whl | head -1)
    echo "✓ Wheel created: $WHEEL_PATH"

    # Copy to home directory
    cp "$WHEEL_PATH" "$HOME/triton_sm120.whl"
    echo "✓ Copied to: $HOME/triton_sm120.whl"

    # Get wheel size
    WHEEL_SIZE=$(du -h "$HOME/triton_sm120.whl" | cut -f1)
    echo "✓ Wheel size: $WHEEL_SIZE"

    echo ""
    echo "======================================================================="
    echo "Step 6: Installing and testing..."
    echo "======================================================================="

    # Install the wheel
    pip install "$HOME/triton_sm120.whl" --force-reinstall

    # Test import
    echo "Testing Triton import..."
    python -c "import triton; print(f'✓ Triton version: {triton.__version__}')"

    # Test CUDA
    if python -c "import torch; print('PyTorch available')" 2>/dev/null; then
        echo "✓ PyTorch is available - ready for full testing!"
    else
        echo "⚠️  PyTorch not installed - install torch_sm120.whl to run full tests"
    fi

    echo ""
    echo "======================================================================="
    echo "🎉 SUCCESS! TRITON BUILT SUCCESSFULLY! 🎉"
    echo "======================================================================="
    echo ""
    echo "Wheel location: $HOME/triton_sm120.whl"
    echo "Build log: $BUILD_DIR/triton-build.log"
    echo ""
    echo "Next steps:"
    echo "  1. Install PyTorch: pip install torch_sm120.whl"
    echo "  2. Test Triton: python test-triton-sm120.py"
    echo "  3. Enjoy full SM 12.0 performance! 🚀"
    echo ""
    echo "After a week of fighting, YOU DID IT! 💪"
    echo "======================================================================="

else
    echo "❌ ERROR: No wheel found in dist/"
    echo ""
    echo "Checking dist/ directory:"
    ls -la dist/
    echo ""
    echo "Last 50 lines of build log:"
    tail -n 50 "$BUILD_DIR/triton-build.log"
    echo ""
    echo "Full build log: $BUILD_DIR/triton-build.log"
    exit 1
fi
