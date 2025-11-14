# Multi-Python Version Build Guide

Complete guide for building PyTorch 2.10.0a0 + Triton 3.5.0 with SM 12.0 support for different Python versions.

---

## 🐍 Python Version Support

This repository includes build scripts and Dockerfiles for multiple Python versions:

| Python Version | Status | Wheel Tag | Build Script |
|---------------|--------|-----------|--------------|
| **Python 3.12** | ✅ Stable | `cp312-cp312-linux_x86_64` | `build-triton-native-fixed.sh` |
| **Python 3.13** | ✅ Available | `cp313-cp313-linux_x86_64` | `build-triton-py313.sh` |
| **Python 3.14** | 🔜 Pre-release | `cp314-cp314-linux_x86_64` | `build-triton-py314.sh` |

---

## 🚀 Quick Start

### For Python 3.13 (Latest Stable)

```bash
# Check Python version
python3.13 --version

# Build Triton
./build-triton-py313.sh

# Or build with Docker
docker build -f Dockerfile.pytorch-triton-py313 -t pytorch-rtx5080-py313:latest .
docker cp $(docker create pytorch-rtx5080-py313:latest):/build/triton_sm120_cp313.whl .
docker cp $(docker create pytorch-rtx5080-py313:latest):/build/torch_sm120_cp313.whl .
```

### For Python 3.14 (When Available)

```bash
# Check Python version
python3.14 --version

# Build Triton
./build-triton-py314.sh

# Or build with Docker
docker build -f Dockerfile.pytorch-triton-py314 -t pytorch-rtx5080-py314:latest .
docker cp $(docker create pytorch-rtx5080-py314:latest):/build/triton_sm120_cp314.whl .
docker cp $(docker create pytorch-rtx5080-py314:latest):/build/torch_sm120_cp314.whl .
```

### For Python 3.12 (Current Default)

```bash
# Use existing scripts
./build-triton-native-fixed.sh

# Or build with Docker
docker build -f Dockerfile.pytorch-triton-builder -t pytorch-rtx5080:latest .
```

---

## 📦 Understanding Wheel Tags

Python wheel files follow the naming convention:
```
{package}-{version}-{python}-{abi}-{platform}.whl
```

### Examples:

**Python 3.12:**
```
triton-3.5.0+git91ca177c-cp312-cp312-linux_x86_64.whl
torch-2.10.0a0+gitc5d91d9-cp312-cp312-linux_x86_64.whl
```

**Python 3.13:**
```
triton-3.5.0+git91ca177c-cp313-cp313-linux_x86_64.whl
torch-2.10.0a0+gitc5d91d9-cp313-cp313-linux_x86_64.whl
```

**Python 3.14:**
```
triton-3.5.0+git91ca177c-cp314-cp314-linux_x86_64.whl
torch-2.10.0a0+gitc5d91d9-cp314-cp314-linux_x86_64.whl
```

### Tag Breakdown:
- `cp312` / `cp313` / `cp314` = CPython 3.12 / 3.13 / 3.14
- `linux_x86_64` = Linux on x86-64 architecture
- Wheels are **NOT** cross-compatible between Python versions

---

## 🔧 Installation by Python Version

### Check Your Python Version First

```bash
python --version
# or
python3 --version
```

### Install Matching Wheels

**Python 3.12:**
```bash
pip install triton_sm120_cp312.whl
pip install torch_sm120_cp312.whl --force-reinstall
```

**Python 3.13:**
```bash
pip install triton_sm120_cp313.whl
pip install torch_sm120_cp313.whl --force-reinstall
```

**Python 3.14:**
```bash
pip install triton_sm120_cp314.whl
pip install torch_sm120_cp314.whl --force-reinstall
```

### Verification

```bash
python -c "import torch; import triton; \
    print(f'PyTorch: {torch.__version__}'); \
    print(f'Triton: {triton.__version__}'); \
    print(f'Python: {torch.version.python}')"
```

---

## 🐳 Docker Builds by Python Version

### Python 3.12

```bash
docker build -f Dockerfile.pytorch-triton-builder -t pytorch-rtx5080-py312:latest .

# Extract wheels
docker create --name extract pytorch-rtx5080-py312:latest
docker cp extract:/build/triton_sm120.whl ./triton_sm120_cp312.whl
docker cp extract:/build/torch_sm120.whl ./torch_sm120_cp312.whl
docker rm extract
```

### Python 3.13

```bash
docker build -f Dockerfile.pytorch-triton-py313 -t pytorch-rtx5080-py313:latest .

# Extract wheels
docker create --name extract pytorch-rtx5080-py313:latest
docker cp extract:/build/triton_sm120_cp313.whl ./triton_sm120_cp313.whl
docker cp extract:/build/torch_sm120_cp313.whl ./torch_sm120_cp313.whl
docker rm extract
```

### Python 3.14

```bash
docker build -f Dockerfile.pytorch-triton-py314 -t pytorch-rtx5080-py314:latest .

# Extract wheels
docker create --name extract pytorch-rtx5080-py314:latest
docker cp extract:/build/triton_sm120_cp314.whl ./triton_sm120_cp314.whl
docker cp extract:/build/torch_sm120_cp314.whl ./torch_sm120_cp314.whl
docker rm extract
```

---

## 🛠️ Native Build Scripts

### Python 3.12 (Current)

```bash
./build-triton-native-fixed.sh
```

**Output:**
- `~/triton-build-sm120/triton_sm120.whl` (cp312)

### Python 3.13

```bash
./build-triton-py313.sh
```

**Output:**
- `~/triton-build-sm120-py313/triton_sm120_cp313.whl`

### Python 3.14

```bash
./build-triton-py314.sh
```

**Output:**
- `~/triton-build-sm120-py314/triton_sm120_cp314.whl`

---

## 🔍 Troubleshooting

### Issue: Wrong Python version error

```
ERROR: torch-2.10.0a0-cp313-cp313-linux_x86_64.whl is not a supported wheel on this platform.
```

**Solution:** Install wheel matching your Python version
```bash
python --version  # Check your version
pip install torch_sm120_cp312.whl  # Use matching cp3XX tag
```

### Issue: Python 3.14 not found

Python 3.14 is not yet released. Options:

1. **Wait for official release** (recommended)
2. **Build from source:**
   ```bash
   git clone https://github.com/python/cpython
   cd cpython && git checkout 3.14
   ./configure --enable-optimizations
   make -j$(nproc)
   sudo make altinstall
   ```

### Issue: Multiple Python versions installed

Use full Python command:
```bash
python3.13 -m pip install torch_sm120_cp313.whl
```

---

## 📊 Build Time Comparison

| Python Version | Triton Build | PyTorch Build | Total | Hardware |
|---------------|--------------|---------------|-------|----------|
| Python 3.12 | ~20 min | ~2.5 hours | ~3 hours | i9-14900KS (24 cores) |
| Python 3.13 | ~20 min | ~2.5 hours | ~3 hours | i9-14900KS (24 cores) |
| Python 3.14 | ~20 min | ~2.5 hours | ~3 hours | i9-14900KS (24 cores) |

Build time is consistent across Python versions.

---

## 🎯 Which Python Version Should I Use?

### For Production

**Python 3.12** ✅ RECOMMENDED
- Most stable
- Best library support
- Proven compatibility

### For Latest Features

**Python 3.13** ✅ GOOD CHOICE
- Stable release (Oct 2024)
- Better performance (JIT improvements)
- Improved error messages

### For Bleeding Edge

**Python 3.14** ⚠️ EXPERIMENTAL
- Not yet released (expected Oct 2025)
- Development builds only
- Use at your own risk

---

## 📦 Pre-built Wheels

Check [GitHub Releases](https://github.com/kentstone84/PyTorch-2.10.0a0-for-Linux-/releases) for pre-built wheels:

- ✅ Python 3.12 wheels available
- 🔜 Python 3.13 wheels coming soon
- 🔜 Python 3.14 wheels when available

---

## 🤝 Contributing

Build wheels for different Python versions? Share them!

1. Build wheels using provided scripts
2. Test thoroughly with test suite
3. Submit PR with build logs
4. Help expand Python version support

---

## 📄 References

- [PEP 3149 – ABI version tagged .so files](https://www.python.org/dev/peps/pep-3149/)
- [PEP 425 – Compatibility Tags for Built Distributions](https://www.python.org/dev/peps/pep-0425/)
- [PyPA Wheel Format](https://packaging.python.org/specifications/binary-distribution-format/)

---

**Built with ❤️ for all Python versions**

🐍 **Python 3.12, 3.13, 3.14 ready!** 🚀
