# PyTorch 2.10.0a0 with SM 12.0 Support for RTX 50-series GPUs

🚀 **Complete Optimization Stack for NVIDIA Blackwell Architecture** 🚀

**Available 12 months before PyTorch 2.10 official release.**

Native Blackwell architecture support for NVIDIA GeForce RTX 5090, 5080, 5070 Ti, and 5070 GPUs.

📖 **[Quick Start Guide](RTX5080_README.md)** | 🔧 **[Triton Build Guide](TRITON_BUILD_GUIDE.md)** | 📋 **[Complete Feature List](FEATURES.md)** | 🐍 **[Python Versions](PYTHON_VERSIONS.md)** | 🐳 **[Docker Install](DOCKER_INSTALL.md)**

## Overview

This is a custom-built PyTorch 2.10.0a0 wheel compiled with **native SM 12.0 (Blackwell) support**. Unlike PyTorch nightlies which only provide PTX backward compatibility (~70-80% performance), this build includes optimized CUDA kernels specifically compiled for RTX 50-series GPUs.

### Why This Build?

Official PyTorch releases and nightlies currently only support up to SM 8.9 (Ada Lovelace/RTX 40-series). When running on RTX 50-series GPUs, they fall back to PTX compatibility mode which:
- Reduces performance by 20-30%
- Increases JIT compilation overhead
- Lacks Blackwell-specific optimizations

This build solves that problem by compiling PyTorch from source with `TORCH_CUDA_ARCH_LIST=12.0`, enabling full native performance.

## Specifications

- **PyTorch Version:** 2.10.0a0+gitc5d91d9
- **CUDA Version:** 13.0.1 (compatible with all CUDA 12.x and 13.x)
- **Python Versions:** 3.12 (stable), 3.13 (available), 3.14 (coming soon) - [See guide](PYTHON_VERSIONS.md)
- **Platform:** Linux x86_64
- **Architecture:** SM 12.0 (compute_120, code_sm_120)
- **Build Date:** November 12, 2025
- **Wheel Size:** 180 MB

## Features Included

This build includes all PyTorch 2.7-2.10 features and Blackwell-specific optimizations:

### Core Features
✅ **Native SM 12.0 Support** - Full Blackwell architecture support
✅ **CUDA 12.x and 13.x Compatible** - Works with CUDA 12.0 through 13.0+
✅ **cuDNN 9.7.0+** - Latest cuDNN with Blackwell optimizations
✅ **NCCL 2.25.1** - Multi-GPU communication library
✅ **CUTLASS 3.8.0** - CUDA template library for linear algebra
✅ **128-bit Vectorization** - Enhanced memory bandwidth utilization
✅ **5th Gen Tensor Cores** - Native Blackwell Tensor Core support

### Compiler Features
✅ **torch.compile** - PyTorch 2.x compilation (Triton required for full optimization)
✅ **Torch Function Modes** - Custom operation overriding
✅ **Mega Cache** - Improved compilation caching

### Known Limitations
⚠️ **Triton Not Included** - Triton 3.3+ with SM 12.0 support requires separate compilation due to CUDA 13.0 PTXAS dependencies. See [TRITON_BUILD_GUIDE.md](TRITON_BUILD_GUIDE.md) for build instructions.

**Impact of Missing Triton:**
- torch.compile works but with reduced optimization (~30-40% slower on attention-heavy workloads)
- FlexAttention limited functionality
- No custom Triton kernels

**Want full torch.compile performance?** Build Triton separately using our guide!

## Supported GPUs

- NVIDIA GeForce RTX 5090
- NVIDIA GeForce RTX 5080
- NVIDIA GeForce RTX 5070 Ti
- NVIDIA GeForce RTX 5070

All GPUs with Blackwell architecture (SM 12.0 / Compute Capability 12.0)

## Requirements

### System Requirements
- Linux x86_64 (Ubuntu 22.04+ recommended)
- Python 3.12
- NVIDIA Driver 570.00 or newer
- CUDA 13.0+ compatible driver

### Python Dependencies
- numpy >= 2.3.0
- packaging >= 25.0
- PyYAML >= 6.0
- typing-extensions >= 4.15.0

All dependencies are listed in `requirements.txt` and will be installed automatically.

## Installation

### Recommended: Install via PyPI

The easiest way to install PyTorch with RTX 50-series support:

```bash
# Install the stone-linux package
pip install stone-linux

# Run the installer (downloads and installs the appropriate PyTorch wheel)
stone-install

# Verify installation
stone-verify
```

The installer will automatically:
- Detect your Python version
- Download the correct PyTorch wheel from GitHub releases
- Install all dependencies
- Verify GPU compatibility

### Alternative: Direct Wheel Installation

Download and install the wheel directly for your Python version:

```bash
# Python 3.10
pip install https://github.com/kentstone84/PyTorch-2.10.0a0-for-Linux-/releases/download/v2.10.0a0/torch-2.10.0a0-cp310-cp310-linux_x86_64.whl

# Python 3.11
pip install https://github.com/kentstone84/PyTorch-2.10.0a0-for-Linux-/releases/download/v2.10.0a0/torch-2.10.0a0-cp311-cp311-linux_x86_64.whl

# Python 3.12
pip install https://github.com/kentstone84/PyTorch-2.10.0a0-for-Linux-/releases/download/v2.10.0a0/torch-2.10.0a0-cp312-cp312-linux_x86_64.whl

# Python 3.13
pip install https://github.com/kentstone84/PyTorch-2.10.0a0-for-Linux-/releases/download/v2.10.0a0/torch-2.10.0a0-cp313-cp313-linux_x86_64.whl

# Python 3.14
pip install https://github.com/kentstone84/PyTorch-2.10.0a0-for-Linux-/releases/download/v2.10.0a0/torch-2.10.0a0-cp314-cp314-linux_x86_64.whl
```

### Alternative: Clone and Install Script

```bash
# Clone this repository
git clone https://github.com/kentstone84/PyTorch-2.10.0a0-for-Linux-.git
cd PyTorch-2.10.0a0-for-Linux-

# Run the installation script
chmod +x install.sh
./install.sh
```

### Windows (WSL2 Required)

This is a Linux wheel. Windows users need WSL2 with Ubuntu:

```bash
# In WSL2 Ubuntu terminal
pip install stone-linux
stone-install
```

## Verification

### Quick Verification

If you installed via `stone-linux`:

```bash
stone-verify
```

### Python Verification

Verify PyTorch is working correctly:

```python
import torch
import stone_linux

# Quick verification
stone_linux.verify_installation()

# Or manually check
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

# Test GPU operation
x = torch.rand(5, 3).cuda()
print(f"Tensor device: {x.device}")
```

Expected output:
```
PyTorch Version: 2.10.0a0+gitc5d91d9
CUDA Available: True
CUDA Version: 13.0
GPU Name: NVIDIA GeForce RTX 5080
Compute Capability: (12, 0)
Tensor device: cuda:0
```

## Performance

Compared to PyTorch nightlies on RTX 50-series:
- **20-30% faster** training and inference
- **No JIT overhead** from PTX compilation
- **Native Blackwell optimizations**
- **Full Tensor Core utilization**

See the [performance benchmarks notebook](notebooks/02_performance_benchmarks.ipynb) for detailed metrics.

## Examples and Tutorials

### Jupyter Notebooks

- **[Getting Started](notebooks/01_getting_started.ipynb)** - Basic PyTorch operations, neural networks, and mixed precision training
- **[Performance Benchmarks](notebooks/02_performance_benchmarks.ipynb)** - Comprehensive performance analysis and optimization tips

### Integration Examples

- **[vLLM Integration](stone_linux/examples/vllm_example.py)** - High-performance LLM inference with vLLM
- **[LangChain Integration](stone_linux/examples/langchain_example.py)** - LLM applications with LangChain
- **[Benchmarking Script](stone_linux/examples/benchmark.py)** - Performance benchmarking utilities

Run examples:
```bash
# Install with examples
pip install 'stone-linux[examples,vllm,langchain]'

# Run vLLM example
python -m stone_linux.examples.vllm_example

# Run LangChain example
python -m stone_linux.examples.langchain_example

# Run benchmarks
python -m stone_linux.examples.benchmark --output results.json
```

## Troubleshooting

### "CUDA not available" after installation

1. Verify NVIDIA driver version:
   ```bash
   nvidia-smi
   ```
   Should show driver >= 570.00

2. Check GPU compute capability:
   ```bash
   nvidia-smi --query-gpu=compute_cap --format=csv,noheader
   ```
   Should show `12.0`

### Python version mismatch

This wheel requires Python 3.12. Create a virtual environment:

```bash
python3.12 -m venv pytorch-env
source pytorch-env/bin/activate
pip install torch_sm120.whl
```

## Building From Source

### PyTorch Only

See the included `Dockerfile.pytorch-builder` for PyTorch-only build instructions.

```bash
docker build -f Dockerfile.pytorch-builder -t pytorch-sm120-builder .
```

### PyTorch + Triton (Recommended for Full Performance)

For the complete build with Triton 3.3+ support:

```bash
chmod +x build-pytorch-triton.sh
./build-pytorch-triton.sh
```

This will create both `torch_sm120.whl` and `triton_sm120.whl`.

**See [TRITON_BUILD_GUIDE.md](TRITON_BUILD_GUIDE.md) for detailed Triton build instructions and troubleshooting.**

### Build Time
- PyTorch only: 1-2 hours
- PyTorch + Triton: 1.5-3 hours

## License

PyTorch is released under the BSD-3-Clause license. This wheel is compiled from the official PyTorch source code with no modifications except for the architecture target.

## Changelog

### v2.10.0a0+gitc5d91d9 (November 12, 2025)
- Initial release
- Built from PyTorch main branch (commit c5d91d9)
- Native SM 12.0 support for RTX 50-series
- CUDA 13.0.1 compatibility
- Python 3.12 support

---

Built with care for the RTX 50-series community.

## Download

The PyTorch wheel file (180 MB) is too large for direct GitHub hosting.

**Download from GitHub Releases:**
https://github.com/kentstone84/PyTorch-2.10.0a0-for-Linux-/releases/latest

Or use this direct link:
```bash
wget https://github.com/kentstone84/PyTorch-2.10.0a0-for-Linux-/releases/download/v2.10.0a0/torch_sm120.whl
```

Then follow the installation instructions above.
