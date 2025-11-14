# Docker Installation Guide - PyTorch RTX 5080 Optimization Stack

Complete guide for running PyTorch 2.10.0a0 + Triton 3.5.0 with Blackwell (SM 12.0) support in Docker.

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Download wheel files to this directory
# (triton_sm120.whl + torch_sm120.whl)

# 2. Run automated installer
./docker-install.sh

# 3. Start container
docker run --rm -it --gpus all pytorch-rtx5080:latest
```

---

## 📋 Prerequisites

### Required Software

1. **Docker** (20.10+)
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   ```

2. **NVIDIA Container Toolkit**
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. **NVIDIA Driver** (570.00+)
   ```bash
   # Check current driver
   nvidia-smi
   ```

4. **Wheel Files** (Download from GitHub Releases)
   - `triton_sm120.whl` (~106 MB)
   - `torch_sm120.whl` (~180 MB)

### Required Hardware

- **GPU**: NVIDIA RTX 5090/5080/5070 Ti/5070 (SM 12.0)
- **VRAM**: 8GB minimum, 16GB+ recommended
- **RAM**: 16GB minimum, 32GB+ recommended
- **Disk**: 10GB free space

---

## 🔧 Installation Methods

### Method 1: Automated Script (Recommended)

```bash
# 1. Place wheel files in this directory
ls -lh *.whl
# triton_sm120.whl
# torch_sm120.whl

# 2. Run installer (checks prerequisites, builds, tests)
./docker-install.sh
```

The script will:
- ✅ Check Docker and NVIDIA runtime
- ✅ Verify GPU compatibility
- ✅ Build Docker image
- ✅ Run installation tests
- ✅ Run comprehensive test suite

### Method 2: Manual Docker Build

```bash
# 1. Build the image
docker build -f Dockerfile.rtx5080-runtime -t pytorch-rtx5080:latest .

# 2. Verify installation
docker run --rm --gpus all pytorch-rtx5080:latest \
  python3 -c "import torch; import triton; \
  print(f'PyTorch: {torch.__version__}'); \
  print(f'Triton: {triton.__version__}'); \
  print(f'CUDA: {torch.cuda.is_available()}')"

# 3. Run test suite
docker run --rm --gpus all -v $(pwd):/workspace \
  pytorch-rtx5080:latest python3 /workspace/test-triton-sm120.py
```

### Method 3: Docker Compose

```bash
# 1. Start services
docker-compose up -d

# 2. Access container
docker-compose exec pytorch-rtx5080 python3

# 3. Run scripts
docker-compose exec pytorch-rtx5080 python3 /workspace/your_script.py

# 4. Stop services
docker-compose down
```

---

## 🎯 Usage Examples

### Interactive Python Session

```bash
docker run --rm -it --gpus all pytorch-rtx5080:latest

# Inside container:
>>> import torch
>>> torch.cuda.get_device_name(0)
'NVIDIA GeForce RTX 5080'
>>> torch.cuda.get_device_capability(0)
(12, 0)
```

### Run Training Script

```bash
# Mount your code directory
docker run --rm --gpus all \
  -v $(pwd)/my_project:/workspace \
  pytorch-rtx5080:latest \
  python3 /workspace/train.py
```

### Jupyter Notebook

```bash
# Run with Jupyter
docker run --rm -it --gpus all \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  pytorch-rtx5080:latest \
  bash -c "pip install jupyter && jupyter notebook --ip=0.0.0.0 --allow-root"
```

### Multi-GPU Training

```bash
# Use all GPUs
docker run --rm --gpus all \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -v $(pwd):/workspace \
  pytorch-rtx5080:latest \
  torchrun --nproc_per_node=2 /workspace/train.py
```

---

## 📊 Verification Tests

### Quick Test

```bash
docker run --rm --gpus all pytorch-rtx5080:latest python3 << 'EOF'
import torch
import triton

# Version check
print(f"PyTorch: {torch.__version__}")
print(f"Triton: {triton.__version__}")
print(f"CUDA: {torch.version.cuda}")

# GPU check
assert torch.cuda.is_available(), "CUDA not available!"
assert torch.cuda.get_device_capability(0) == (12, 0), "Not SM 12.0!"

# Performance test
x = torch.rand(5000, 5000).cuda()
y = torch.rand(5000, 5000).cuda()
z = torch.matmul(x, y)

print("✓ All tests passed!")
EOF
```

### Comprehensive Test Suite

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  pytorch-rtx5080:latest \
  python3 /workspace/test-triton-sm120.py
```

Expected output:
```
======================================================================
Triton SM 12.0 (Blackwell) Test Suite
======================================================================

[1/6] Checking versions...
  ✓ PyTorch version: 2.10.0a0
  ✓ Triton version: 3.5.0
  ✓ CUDA version: 13.0

[2/6] Checking GPU...
  ✓ GPU: NVIDIA GeForce RTX 5080
  ✓ Compute Capability: 12.0

[3/6] Testing simple Triton kernel compilation...
  ✓ Simple kernel compiled successfully

[4/6] Testing matrix multiplication kernel...
  ✓ MatMul kernel working correctly

[5/6] Testing fused operations...
  ✓ Fused operations working correctly

[6/6] Running performance benchmark...
  ✓ Performance benchmark complete

======================================================================
✓ ALL TESTS PASSED!
======================================================================
```

---

## 🔧 Customization

### Modify Dockerfile

```dockerfile
# Add custom packages
RUN pip install --no-cache-dir \
    transformers \
    diffusers \
    accelerate \
    wandb
```

### Adjust Memory Limits

```yaml
# docker-compose.yml
services:
  pytorch-rtx5080:
    shm_size: 32gb  # Increase shared memory
    mem_limit: 64g  # Set memory limit
```

### Use Specific GPU

```bash
# Use GPU 1 only
docker run --rm --gpus '"device=1"' \
  -e CUDA_VISIBLE_DEVICES=0 \
  pytorch-rtx5080:latest \
  python3 train.py
```

---

## 🐛 Troubleshooting

### Issue: "docker: Error response from daemon: could not select device driver"

**Solution**: Install NVIDIA Container Toolkit
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Issue: "RuntimeError: CUDA error: no kernel image is available"

**Solution**: Verify you're using SM 12.0 GPU
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Should show: 12.0
```

### Issue: Import errors or version mismatches

**Solution**: Rebuild image without cache
```bash
docker build --no-cache -f Dockerfile.rtx5080-runtime -t pytorch-rtx5080:latest .
```

### Issue: Out of memory errors

**Solution**: Increase shared memory
```bash
docker run --rm --gpus all --shm-size=16g pytorch-rtx5080:latest python3 train.py
```

### Issue: Slow performance

**Solution**: Enable TF32 and optimizations
```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
```

---

## 📦 Image Details

### Base Image
- `nvidia/cuda:13.0.1-runtime-ubuntu24.04`
- CUDA 13.0.1 runtime
- Ubuntu 24.04 LTS

### Installed Packages
- Python 3.12
- PyTorch 2.10.0a0 (SM 12.0 native)
- Triton 3.5.0 (SM 12.0 native)
- NumPy, packaging, filelock, etc.

### Environment Variables
- `TORCH_CUDA_ARCH_LIST=12.0`
- `CUDA_VISIBLE_DEVICES=0`
- `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`
- `CUDA_MODULE_LOADING=LAZY`

### Image Size
- Approximately 8-10 GB

---

## 🚀 Performance Tips

1. **Enable TF32 for faster matmuls**
   ```python
   torch.backends.cuda.matmul.allow_tf32 = True
   ```

2. **Use BFloat16 for training**
   ```python
   torch.set_default_dtype(torch.bfloat16)
   ```

3. **Enable CUDA Graphs**
   ```python
   # Capture compute graph for repeated operations
   g = torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):
       output = model(input)
   ```

4. **Use torch.compile with Triton**
   ```python
   model = torch.compile(model, mode="max-autotune")
   ```

5. **Increase shared memory**
   ```bash
   docker run --shm-size=16g ...
   ```

---

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs)
- [Triton Documentation](https://triton-lang.org)
- [NVIDIA Docker Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit)
- [Build from Source Guide](BUILD_INSTRUCTIONS.md)
- [Feature List](FEATURES.md)

---

## 🤝 Support

- **Issues**: https://github.com/kentstone84/PyTorch-2.10.0a0-for-Linux-/issues
- **Discussions**: https://github.com/kentstone84/PyTorch-2.10.0a0-for-Linux-/discussions
- **Releases**: https://github.com/kentstone84/PyTorch-2.10.0a0-for-Linux-/releases

---

**Built with ❤️ for the RTX 50-series community**

🐳 **Docker-ready. Production-tested. Blackwell-optimized.** 🚀
