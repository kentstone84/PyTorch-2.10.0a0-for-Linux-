#!/bin/bash
# ========================================================================
# Build Triton with SM 12.0 (Blackwell) Support
# ========================================================================

set -e

echo "======================================================================="
echo "Building Triton for SM 12.0 (Blackwell) with CUDA 13.0"
echo "======================================================================="

# Build the Docker image
docker build -f Dockerfile.triton-builder -t triton-sm120-builder .

# Run the container
CONTAINER_ID=$(docker run -d triton-sm120-builder)

# Wait for build to complete
echo "Waiting for Triton build to complete..."
docker logs -f $CONTAINER_ID

# Copy the built wheel (if using bdist_wheel)
# docker cp $CONTAINER_ID:/build/triton/python/dist/triton-*.whl ./triton_sm120.whl

# Or create a wheel from the editable install
echo "Creating Triton wheel..."
docker exec $CONTAINER_ID bash -c "cd /build/triton/python && python setup.py bdist_wheel"
docker cp $CONTAINER_ID:/build/triton/python/dist/triton-3.3.0-cp312-cp312-linux_x86_64.whl ./triton_sm120.whl || \
docker cp $CONTAINER_ID:/build/triton/python/dist/triton-*.whl ./triton_sm120.whl

# Cleanup
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID

echo "======================================================================="
echo "Triton wheel created: triton_sm120.whl"
echo "======================================================================="
