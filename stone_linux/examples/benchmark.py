"""
Benchmark script for PyTorch RTX 50-series performance

This script runs comprehensive benchmarks to measure PyTorch performance
on RTX 50-series GPUs with SM 12.0 support.

Usage:
    python benchmark.py --output results.json
    python benchmark.py --model resnet50 --batch-size 64
"""

import torch
import torch.nn as nn
import time
import json
import argparse
from typing import Dict, List, Any
from pathlib import Path


def verify_rtx_setup() -> Dict[str, Any]:
    """Verify RTX 50-series setup and collect system info."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    info = {
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "cudnn_version": torch.backends.cudnn.version(),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
    }

    return info


def benchmark_matmul(
    size: int = 8192,
    iterations: int = 100,
    dtype: torch.dtype = torch.float16
) -> Dict[str, float]:
    """Benchmark matrix multiplication performance."""
    device = torch.device("cuda")

    # Create random matrices
    a = torch.randn(size, size, dtype=dtype, device=device)
    b = torch.randn(size, size, dtype=dtype, device=device)

    # Warmup
    for _ in range(10):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / iterations
    tflops = (2 * size ** 3) / (avg_time * 1e12)  # TFLOPS

    return {
        "total_time_s": total_time,
        "avg_time_ms": avg_time * 1000,
        "tflops": tflops,
        "iterations": iterations,
        "matrix_size": size,
        "dtype": str(dtype),
    }


def benchmark_conv2d(
    batch_size: int = 32,
    iterations: int = 100,
    dtype: torch.dtype = torch.float16
) -> Dict[str, float]:
    """Benchmark 2D convolution performance."""
    device = torch.device("cuda")

    # Create conv layer and input
    conv = nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device).to(dtype)
    x = torch.randn(batch_size, 3, 224, 224, dtype=dtype, device=device)

    # Warmup
    for _ in range(10):
        _ = conv(x)
    torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        y = conv(x)
    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / iterations
    throughput = batch_size / avg_time  # images/second

    return {
        "total_time_s": total_time,
        "avg_time_ms": avg_time * 1000,
        "throughput_imgs_per_sec": throughput,
        "iterations": iterations,
        "batch_size": batch_size,
        "dtype": str(dtype),
    }


def benchmark_transformer_block(
    batch_size: int = 32,
    seq_len: int = 512,
    hidden_dim: int = 768,
    iterations: int = 50,
    dtype: torch.dtype = torch.float16
) -> Dict[str, float]:
    """Benchmark transformer block performance."""
    device = torch.device("cuda")

    # Create transformer layer
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=12,
        dim_feedforward=3072,
        batch_first=True
    ).to(device).to(dtype)

    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype, device=device)

    # Warmup
    for _ in range(10):
        _ = encoder_layer(x)
    torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        y = encoder_layer(x)
    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / iterations
    throughput = (batch_size * seq_len) / avg_time  # tokens/second

    return {
        "total_time_s": total_time,
        "avg_time_ms": avg_time * 1000,
        "throughput_tokens_per_sec": throughput,
        "iterations": iterations,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "dtype": str(dtype),
    }


def benchmark_memory_bandwidth() -> Dict[str, float]:
    """Benchmark GPU memory bandwidth."""
    device = torch.device("cuda")
    size = 1024 * 1024 * 256  # 256M elements
    iterations = 50

    # Create large tensor
    x = torch.randn(size, dtype=torch.float32, device=device)
    y = torch.randn(size, dtype=torch.float32, device=device)

    # Warmup
    for _ in range(10):
        _ = x + y
    torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        z = x + y
    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    bytes_transferred = size * 4 * 3 * iterations  # 4 bytes per float32, 3 arrays (read x, read y, write z)
    bandwidth_gb_s = bytes_transferred / (total_time * 1e9)

    return {
        "bandwidth_gb_per_sec": bandwidth_gb_s,
        "total_time_s": total_time,
        "iterations": iterations,
        "tensor_size_mb": (size * 4) / (1024 * 1024),
    }


def benchmark_mixed_precision() -> Dict[str, Any]:
    """Benchmark mixed precision training performance."""
    device = torch.device("cuda")
    batch_size = 32
    iterations = 50

    # Simple model
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    results = {}

    # FP32 benchmark
    model.float()
    x = torch.randn(batch_size, 1024, dtype=torch.float32, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)

    # Warmup
    for _ in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(iterations):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    end_time = time.time()

    results["fp32"] = {
        "time_per_step_ms": ((end_time - start_time) / iterations) * 1000,
        "throughput_samples_per_sec": batch_size * iterations / (end_time - start_time),
    }

    # FP16 with automatic mixed precision
    model.half()
    x = x.half()
    scaler = torch.cuda.amp.GradScaler()

    # Warmup
    for _ in range(10):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(x)
            loss = criterion(output, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(iterations):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(x)
            loss = criterion(output, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    end_time = time.time()

    results["fp16_amp"] = {
        "time_per_step_ms": ((end_time - start_time) / iterations) * 1000,
        "throughput_samples_per_sec": batch_size * iterations / (end_time - start_time),
    }

    results["speedup"] = results["fp32"]["time_per_step_ms"] / results["fp16_amp"]["time_per_step_ms"]

    return results


def run_all_benchmarks() -> Dict[str, Any]:
    """Run all benchmarks and return results."""
    print("=" * 60)
    print("PyTorch RTX 50-Series Performance Benchmark")
    print("=" * 60)

    # System info
    print("\nCollecting system information...")
    system_info = verify_rtx_setup()
    print(f"GPU: {system_info['gpu_name']}")
    print(f"Compute Capability: {system_info['compute_capability']}")
    print(f"PyTorch: {system_info['pytorch_version']}")
    print(f"CUDA: {system_info['cuda_version']}")

    results = {
        "system_info": system_info,
        "benchmarks": {}
    }

    # Matrix multiplication
    print("\n" + "=" * 60)
    print("Benchmarking Matrix Multiplication (FP16)...")
    results["benchmarks"]["matmul_fp16"] = benchmark_matmul(dtype=torch.float16)
    print(f"  TFLOPS: {results['benchmarks']['matmul_fp16']['tflops']:.2f}")

    print("\nBenchmarking Matrix Multiplication (FP32)...")
    results["benchmarks"]["matmul_fp32"] = benchmark_matmul(dtype=torch.float32)
    print(f"  TFLOPS: {results['benchmarks']['matmul_fp32']['tflops']:.2f}")

    # Conv2D
    print("\n" + "=" * 60)
    print("Benchmarking 2D Convolution...")
    results["benchmarks"]["conv2d"] = benchmark_conv2d()
    print(f"  Throughput: {results['benchmarks']['conv2d']['throughput_imgs_per_sec']:.2f} imgs/s")

    # Transformer
    print("\n" + "=" * 60)
    print("Benchmarking Transformer Block...")
    results["benchmarks"]["transformer"] = benchmark_transformer_block()
    print(f"  Throughput: {results['benchmarks']['transformer']['throughput_tokens_per_sec']:.2f} tokens/s")

    # Memory bandwidth
    print("\n" + "=" * 60)
    print("Benchmarking Memory Bandwidth...")
    results["benchmarks"]["memory_bandwidth"] = benchmark_memory_bandwidth()
    print(f"  Bandwidth: {results['benchmarks']['memory_bandwidth']['bandwidth_gb_per_sec']:.2f} GB/s")

    # Mixed precision
    print("\n" + "=" * 60)
    print("Benchmarking Mixed Precision Training...")
    results["benchmarks"]["mixed_precision"] = benchmark_mixed_precision()
    print(f"  FP32: {results['benchmarks']['mixed_precision']['fp32']['time_per_step_ms']:.2f} ms/step")
    print(f"  FP16 AMP: {results['benchmarks']['mixed_precision']['fp16_amp']['time_per_step_ms']:.2f} ms/step")
    print(f"  Speedup: {results['benchmarks']['mixed_precision']['speedup']:.2f}x")

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch performance on RTX 50-series GPUs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark-results.json",
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["matmul", "conv2d", "transformer", "memory", "mixed-precision", "all"],
        default="all",
        help="Which benchmark to run"
    )

    args = parser.parse_args()

    if args.benchmark == "all":
        results = run_all_benchmarks()
    else:
        system_info = verify_rtx_setup()
        results = {"system_info": system_info, "benchmarks": {}}

        if args.benchmark == "matmul":
            results["benchmarks"]["matmul_fp16"] = benchmark_matmul(dtype=torch.float16)
            results["benchmarks"]["matmul_fp32"] = benchmark_matmul(dtype=torch.float32)
        elif args.benchmark == "conv2d":
            results["benchmarks"]["conv2d"] = benchmark_conv2d()
        elif args.benchmark == "transformer":
            results["benchmarks"]["transformer"] = benchmark_transformer_block()
        elif args.benchmark == "memory":
            results["benchmarks"]["memory_bandwidth"] = benchmark_memory_bandwidth()
        elif args.benchmark == "mixed-precision":
            results["benchmarks"]["mixed_precision"] = benchmark_mixed_precision()

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
