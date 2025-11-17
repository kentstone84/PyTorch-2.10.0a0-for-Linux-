"""
Example: Using vLLM with PyTorch RTX 50-series support

This example demonstrates how to use vLLM with the stone-linux optimized PyTorch build
for maximum performance on RTX 50-series GPUs.

Requirements:
    pip install stone-linux vllm

GPU Requirements:
    - NVIDIA RTX 5090, 5080, 5070 Ti, or 5070
    - NVIDIA Driver >= 570.00
    - CUDA 13.0+
"""

import torch
from typing import List, Optional
import time


def verify_rtx_setup():
    """Verify that we're running on RTX 50-series with proper setup."""
    print("=" * 60)
    print("RTX 50-Series Setup Verification")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    cuda_version = torch.version.cuda

    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
    print(f"CUDA Version: {cuda_version}")

    if compute_cap != (12, 0):
        print("⚠️  Warning: Not running on SM 12.0 (Blackwell) GPU")
        print("   Performance may be suboptimal")
    else:
        print("✓ SM 12.0 (Blackwell) support detected!")

    print("=" * 60)


def run_vllm_inference(
    model_name: str = "facebook/opt-125m",
    prompts: Optional[List[str]] = None,
    max_tokens: int = 100,
    temperature: float = 0.7,
) -> None:
    """
    Run inference using vLLM.

    Args:
        model_name: HuggingFace model name
        prompts: List of prompts to generate from
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError(
            "vLLM is not installed. Install with: pip install vllm\n"
            "Or install with stone-linux: pip install 'stone-linux[vllm]'"
        )

    # Verify RTX setup
    verify_rtx_setup()

    # Default prompts if none provided
    if prompts is None:
        prompts = [
            "The future of AI is",
            "RTX 50-series GPUs are designed for",
            "Deep learning with PyTorch allows us to",
        ]

    print(f"\nLoading model: {model_name}")
    print("This may take a minute on first run...\n")

    # Initialize vLLM with optimal settings for RTX 50-series
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.9,  # Use 90% of GPU memory
        max_model_len=2048,
        dtype="float16",  # Use FP16 for faster inference
        trust_remote_code=True,
    )

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    # Generate
    print("=" * 60)
    print("vLLM Inference on RTX 50-Series")
    print("=" * 60)

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("-" * 60)

    # Performance stats
    total_time = end_time - start_time
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    throughput = total_tokens / total_time

    print("\n" + "=" * 60)
    print("Performance Statistics")
    print("=" * 60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Throughput: {throughput:.2f} tokens/second")
    print("=" * 60)


def run_batch_inference_benchmark(
    model_name: str = "facebook/opt-125m",
    batch_sizes: List[int] = [1, 4, 8, 16, 32],
    max_tokens: int = 100,
) -> None:
    """
    Benchmark vLLM with different batch sizes.

    Args:
        model_name: HuggingFace model name
        batch_sizes: List of batch sizes to test
        max_tokens: Maximum tokens to generate
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError("vLLM is not installed. Install with: pip install vllm")

    verify_rtx_setup()

    print(f"\nLoading model: {model_name}\n")

    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        dtype="float16",
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
    )

    results = []

    print("=" * 60)
    print("Batch Size Benchmark")
    print("=" * 60)

    for batch_size in batch_sizes:
        prompts = [f"Test prompt number {i}" for i in range(batch_size)]

        # Warmup
        llm.generate(prompts[:1], sampling_params)

        # Benchmark
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.time()

        total_time = end_time - start_time
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        throughput = total_tokens / total_time

        results.append({
            "batch_size": batch_size,
            "time": total_time,
            "tokens": total_tokens,
            "throughput": throughput,
        })

        print(f"Batch Size {batch_size:3d}: {throughput:8.2f} tokens/s ({total_time:.2f}s)")

    print("=" * 60)

    # Find optimal batch size
    optimal = max(results, key=lambda x: x["throughput"])
    print(f"\nOptimal batch size: {optimal['batch_size']}")
    print(f"Peak throughput: {optimal['throughput']:.2f} tokens/s")
    print("=" * 60)


def main():
    """Main function to run examples."""
    import argparse

    parser = argparse.ArgumentParser(
        description="vLLM examples for RTX 50-series GPUs"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model name from HuggingFace",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run batch size benchmark",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        action="append",
        help="Custom prompt (can be specified multiple times)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )

    args = parser.parse_args()

    if args.benchmark:
        run_batch_inference_benchmark(
            model_name=args.model,
            max_tokens=args.max_tokens,
        )
    else:
        run_vllm_inference(
            model_name=args.model,
            prompts=args.prompt,
            max_tokens=args.max_tokens,
        )


if __name__ == "__main__":
    main()
