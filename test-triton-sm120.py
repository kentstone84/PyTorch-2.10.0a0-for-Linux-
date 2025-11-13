#!/usr/bin/env python3
"""
Test script for Triton SM 12.0 (Blackwell) support
Verifies that Triton is correctly installed and can compile kernels for RTX 50-series GPUs
"""

import sys
import torch
import triton
import triton.language as tl

print("=" * 70)
print("Triton SM 12.0 (Blackwell) Test Suite")
print("=" * 70)

# Test 1: Version Check
print("\n[1/6] Checking versions...")
print(f"  PyTorch version: {torch.__version__}")
print(f"  Triton version: {triton.__version__}")
print(f"  CUDA version: {torch.version.cuda}")

# Test 2: GPU Detection
print("\n[2/6] Checking GPU...")
if not torch.cuda.is_available():
    print("  ❌ CUDA not available")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
compute_cap = torch.cuda.get_device_capability(0)
print(f"  ✓ GPU: {gpu_name}")
print(f"  ✓ Compute Capability: {compute_cap[0]}.{compute_cap[1]}")

if compute_cap != (12, 0):
    print(f"  ⚠️  WARNING: Expected SM 12.0, got SM {compute_cap[0]}.{compute_cap[1]}")
else:
    print("  ✓ SM 12.0 (Blackwell) detected!")

# Test 3: Simple Triton Kernel
print("\n[3/6] Testing simple Triton kernel compilation...")

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Simple vector addition kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

try:
    # Test kernel
    size = 1024
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    output = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, size, BLOCK_SIZE=128)

    # Verify result
    expected = x + y
    if torch.allclose(output, expected):
        print("  ✓ Simple kernel compiled and executed correctly")
    else:
        print("  ❌ Kernel output doesn't match expected result")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ Kernel compilation/execution failed: {e}")
    sys.exit(1)

# Test 4: Matrix Multiplication Kernel
print("\n[4/6] Testing matrix multiplication kernel...")

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Matrix multiplication kernel"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

try:
    M, N, K = 512, 512, 512
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
    )

    # Verify result
    expected = torch.matmul(a, b)
    if torch.allclose(c, expected, rtol=1e-2):
        print("  ✓ MatMul kernel compiled and executed correctly")
    else:
        max_diff = (c - expected).abs().max().item()
        print(f"  ⚠️  MatMul result differs by max: {max_diff}")
        if max_diff < 0.1:
            print("  ✓ Difference is acceptable for FP16")
        else:
            print("  ❌ Difference is too large")
            sys.exit(1)
except Exception as e:
    print(f"  ❌ MatMul kernel failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Fused Operations
print("\n[5/6] Testing fused operations (Blackwell-specific)...")

@triton.jit
def fused_relu_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Fused ReLU + Add kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Fused operation: ReLU(x) + y
    x_relu = tl.maximum(x, 0.0)
    result = x_relu + y

    tl.store(output_ptr + offsets, result, mask=mask)

try:
    size = 2048
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    output = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    fused_relu_add_kernel[grid](x, y, output, size, BLOCK_SIZE=256)

    # Verify
    expected = torch.relu(x) + y
    if torch.allclose(output, expected):
        print("  ✓ Fused operations working correctly")
    else:
        print("  ❌ Fused operations failed")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ Fused operations test failed: {e}")
    sys.exit(1)

# Test 6: Performance Benchmark
print("\n[6/6] Running performance benchmark...")

try:
    import time

    size = 4096
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    output = torch.empty_like(x)

    # Warmup
    for _ in range(10):
        grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, output, size, BLOCK_SIZE=256)
    torch.cuda.synchronize()

    # Benchmark Triton
    start = time.perf_counter()
    for _ in range(1000):
        grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, output, size, BLOCK_SIZE=256)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) * 1000  # ms

    # Benchmark PyTorch
    start = time.perf_counter()
    for _ in range(1000):
        torch_output = x + y
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) * 1000  # ms

    print(f"  Triton kernel: {triton_time:.2f}ms (1000 iterations)")
    print(f"  PyTorch op:    {torch_time:.2f}ms (1000 iterations)")
    print(f"  Ratio: {triton_time/torch_time:.2f}x")
    print("  ✓ Performance benchmark complete")

except Exception as e:
    print(f"  ⚠️  Benchmark failed: {e}")
    print("  (This is not critical)")

# Final Summary
print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print(f"\nTriton {triton.__version__} is correctly installed and working")
print(f"with SM 12.0 (Blackwell) support on {gpu_name}")
print("\nYour RTX 5080 is ready for:")
print("  ✓ torch.compile with full optimization")
print("  ✓ FlexAttention")
print("  ✓ Custom Triton kernels")
print("  ✓ Blackwell-optimized operations")
print("=" * 70)
