# Building Triton with SM 12.0 (Blackwell) Support

## The Problem

PyTorch 2.7+ includes Triton 3.3 which adds Blackwell (SM 12.0) support. However, when building with **CUDA 13.0**, Triton compilation fails because:

1. **PTXAS Version Mismatch**: Triton 3.3 ships with CUDA 12.8's PTXAS, but SM 12.0 requires CUDA 13.0's PTXAS
2. **Architecture Recognition**: Older PTXAS doesn't recognize `sm_120` architecture
3. **Build Complexity**: Triton requires LLVM 18, MLIR support, and proper CUDA configuration

## Common Errors

```
ptxas fatal: Value 'sm_120' is not defined for option 'gpu-name'
```

or

```
LLVM ERROR: Failed to translate TritonGPU to LLVM IR
```

## Solutions

### Option 1: Build Both PyTorch + Triton (Recommended)

Use the combined Dockerfile that builds both together:

```bash
chmod +x build-pytorch-triton.sh
./build-pytorch-triton.sh
```

This will create two wheels:
- `torch_sm120.whl` - PyTorch with SM 12.0 support
- `triton_sm120.whl` - Triton with SM 12.0 support

**Installation:**
```bash
pip install triton_sm120.whl
pip install torch_sm120.whl --force-reinstall
```

### Option 2: Build Only Triton

If you already have PyTorch but need Triton:

```bash
chmod +x build-triton.sh
./build-triton.sh
```

Then install:
```bash
pip install triton_sm120.whl
```

### Option 3: Manual Build

For advanced users who want to customize the build:

```bash
# Install dependencies
apt-get install -y llvm-18 llvm-18-dev libz3-dev zlib1g-dev

# Clone Triton
git clone https://github.com/triton-lang/triton.git
cd triton/python

# Set environment variables
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export CUDACXX=/usr/local/cuda/bin/nvcc
export TORCH_CUDA_ARCH_LIST="12.0"

# Build wheel
python setup.py bdist_wheel
```

## Key Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `TRITON_PTXAS_PATH` | `/usr/local/cuda/bin/ptxas` | Use CUDA 13.0's PTXAS |
| `CUDACXX` | `/usr/local/cuda/bin/nvcc` | CUDA compiler path |
| `TORCH_CUDA_ARCH_LIST` | `"12.0"` | Target SM 12.0 architecture |
| `TRITON_BUILD_WITH_CCACHE` | `1` | Speed up compilation |

## Verification

After installation, verify Triton works with your RTX 50-series GPU:

```python
import torch
import triton
import triton.language as tl

print(f"PyTorch: {torch.__version__}")
print(f"Triton: {triton.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

# Test Triton kernel compilation
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

# Test kernel
size = 1024
x = torch.randn(size, device='cuda')
y = torch.randn(size, device='cuda')
output = torch.empty_like(x)

grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
add_kernel[grid](x, y, output, size, BLOCK_SIZE=128)

print(f"Triton kernel test: {'PASSED' if torch.allclose(x + y, output) else 'FAILED'}")
```

Expected output:
```
PyTorch: 2.10.0a0+gitc5d91d9
Triton: 3.3.0
CUDA Available: True
GPU: NVIDIA GeForce RTX 5080
Compute Capability: (12, 0)
Triton kernel test: PASSED
```

## Build Time Expectations

- **Triton only**: 30-60 minutes
- **PyTorch + Triton**: 1.5-3 hours

Build time depends on:
- CPU cores (set `MAX_JOBS` to match your CPU)
- RAM (16GB+ recommended)
- SSD vs HDD
- ccache availability

## Troubleshooting

### "LLVM ERROR: Failed to translate TritonGPU to LLVM IR"

**Solution**: Make sure you have LLVM 18 installed:
```bash
apt-get install llvm-18 llvm-18-dev
```

### "ptxas not found"

**Solution**: Ensure CUDA 13.0 is installed and set the path:
```bash
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
```

### Build fails with "out of memory"

**Solution**: Reduce parallel jobs:
```bash
export MAX_JOBS=4
```

### "sm_120 is not defined"

**Solution**: Make sure you're using CUDA 13.0's PTXAS, not an older version:
```bash
/usr/local/cuda/bin/ptxas --version
```

Should show version 13.0 or higher.

## Features Enabled with Triton

With Triton successfully built and integrated, you get:

✅ **torch.compile with Blackwell optimization**
✅ **FlexAttention for long-context models**
✅ **5th generation Tensor Core support**
✅ **Tensor Memory modeling and support**
✅ **Native mxfp4 and mxfp8 formats** (microscaling)
✅ **Improved software pipeliner**
✅ **Custom kernel compilation for RTX 50-series**

## Performance Impact

Without Triton:
- ❌ torch.compile falls back to eager mode or TorchScript
- ❌ No custom kernel optimizations
- ❌ Limited FlexAttention support
- 📉 **~30-40% slower** on attention-heavy workloads

With Triton:
- ✅ Full torch.compile optimization
- ✅ Custom Blackwell-optimized kernels
- ✅ Complete FlexAttention support
- 📈 **Maximum performance** on RTX 50-series

## Alternative: Use PyTorch Nightly

If you don't want to build from source, you can use PyTorch nightly builds with CUDA 12.8 (which includes pre-built Triton). However, this won't work with CUDA 13.0.

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Note**: This only supports CUDA 12.8, not 13.0, and may have performance differences.

## Contributing

If you successfully build Triton with CUDA 13.0 and have improvements to the build process, please share your findings!

## References

- [Triton GitHub](https://github.com/triton-lang/triton)
- [PyTorch Blackwell Tracking Issue](https://github.com/pytorch/pytorch/issues/145949)
- [Triton 3.3 Blackwell Support PR](https://github.com/triton-lang/triton/pull/5724)
- [PyTorch SM 12.0 Support Issue](https://github.com/pytorch/pytorch/issues/159207)
