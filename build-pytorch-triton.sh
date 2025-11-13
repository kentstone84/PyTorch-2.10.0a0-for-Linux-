#!/bin/bash
# ========================================================================
# Build PyTorch + Triton with SM 12.0 (Blackwell) Support
# ========================================================================

set -e

echo "======================================================================="
echo "Building PyTorch 2.10.0a0 + Triton 3.3+ for SM 12.0 (Blackwell)"
echo "This will build both PyTorch and Triton with CUDA 13.0 support"
echo "======================================================================="

# Build the Docker image
echo "Building Docker image..."
docker build -f Dockerfile.pytorch-triton-builder -t pytorch-triton-sm120-builder .

# Run the container
echo "Starting build container..."
CONTAINER_ID=$(docker run -d pytorch-triton-sm120-builder)

# Wait for build to complete (this can take 1-3 hours)
echo "Building... This will take 1-3 hours depending on your hardware."
echo "You can monitor progress with: docker logs -f $CONTAINER_ID"
docker wait $CONTAINER_ID

# Show final logs
echo "======================================================================="
echo "Build logs:"
echo "======================================================================="
docker logs --tail 50 $CONTAINER_ID

# Extract the wheels
echo ""
echo "======================================================================="
echo "Extracting wheels..."
echo "======================================================================="

docker cp $CONTAINER_ID:/build/torch_sm120.whl ./torch_sm120.whl
docker cp $CONTAINER_ID:/build/triton_sm120.whl ./triton_sm120.whl

# Cleanup
docker stop $CONTAINER_ID 2>/dev/null || true
docker rm $CONTAINER_ID

echo ""
echo "======================================================================="
echo "Build complete!"
echo "======================================================================="
echo "Created wheels:"
ls -lh torch_sm120.whl triton_sm120.whl
echo ""
echo "To install:"
echo "  pip install triton_sm120.whl"
echo "  pip install torch_sm120.whl --force-reinstall"
echo "======================================================================="
