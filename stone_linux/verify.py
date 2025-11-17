"""Verification utilities for PyTorch RTX 50-series installation."""

import sys
from typing import Dict, Optional, Tuple
from stone_linux.config import (
    EXPECTED_COMPUTE_CAPABILITY,
    MIN_DRIVER_VERSION,
    SUPPORTED_GPUS,
    get_python_version,
    is_python_version_supported,
)


def check_python_version() -> Tuple[bool, str]:
    """
    Check if the current Python version is supported.

    Returns:
        Tuple of (is_supported, message)
    """
    if is_python_version_supported():
        return True, f"Python {get_python_version()} is supported"
    return False, f"Python {get_python_version()} is not supported"


def check_cuda_available() -> Tuple[bool, str]:
    """
    Check if CUDA is available in PyTorch.

    Returns:
        Tuple of (is_available, message)
    """
    try:
        import torch

        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            return True, f"CUDA {cuda_version} is available"
        else:
            return False, "CUDA is not available"
    except ImportError:
        return False, "PyTorch is not installed"
    except Exception as e:
        return False, f"Error checking CUDA: {str(e)}"


def get_gpu_info() -> Dict[str, any]:
    """
    Get information about the GPU.

    Returns:
        Dictionary containing GPU information
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        info = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": torch.cuda.get_device_capability(0),
            "total_memory": torch.cuda.get_device_properties(0).total_memory / (1024**3),  # GB
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
        }

        return info
    except ImportError:
        return {"error": "PyTorch is not installed"}
    except Exception as e:
        return {"error": str(e)}


def check_compute_capability() -> Tuple[bool, str]:
    """
    Check if the GPU has the expected compute capability (12.0).

    Returns:
        Tuple of (is_correct, message)
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA is not available"

        capability = torch.cuda.get_device_capability(0)

        if capability == EXPECTED_COMPUTE_CAPABILITY:
            return True, f"Compute capability is {capability[0]}.{capability[1]} (SM 12.0)"
        else:
            return False, f"Compute capability is {capability[0]}.{capability[1]}, expected {EXPECTED_COMPUTE_CAPABILITY[0]}.{EXPECTED_COMPUTE_CAPABILITY[1]}"
    except ImportError:
        return False, "PyTorch is not installed"
    except Exception as e:
        return False, f"Error checking compute capability: {str(e)}"


def check_gpu_supported() -> Tuple[bool, str]:
    """
    Check if the GPU is in the list of supported GPUs.

    Returns:
        Tuple of (is_supported, message)
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA is not available"

        gpu_name = torch.cuda.get_device_name(0)

        if gpu_name in SUPPORTED_GPUS:
            return True, f"{gpu_name} is supported"
        else:
            # Check if it's still a Blackwell GPU with SM 12.0
            capability = torch.cuda.get_device_capability(0)
            if capability == EXPECTED_COMPUTE_CAPABILITY:
                return True, f"{gpu_name} has SM 12.0 support (may be a new RTX 50-series model)"
            return False, f"{gpu_name} is not officially supported"
    except ImportError:
        return False, "PyTorch is not installed"
    except Exception as e:
        return False, f"Error checking GPU: {str(e)}"


def verify_installation(verbose: bool = True) -> bool:
    """
    Verify the complete PyTorch RTX 50-series installation.

    Args:
        verbose: If True, print detailed information

    Returns:
        True if installation is valid, False otherwise
    """
    checks = [
        ("Python Version", check_python_version),
        ("CUDA Available", check_cuda_available),
        ("Compute Capability", check_compute_capability),
        ("GPU Support", check_gpu_supported),
    ]

    all_passed = True

    if verbose:
        print("=" * 60)
        print("PyTorch RTX 50-series Installation Verification")
        print("=" * 60)

    for check_name, check_func in checks:
        passed, message = check_func()

        if verbose:
            status = "✓" if passed else "✗"
            print(f"{status} {check_name}: {message}")

        if not passed:
            all_passed = False

    if verbose:
        print("=" * 60)

        if all_passed:
            print("✓ All checks passed!")

            # Print GPU info
            gpu_info = get_gpu_info()
            if "error" not in gpu_info:
                print("\nGPU Information:")
                print(f"  Name: {gpu_info['name']}")
                print(f"  Compute Capability: {gpu_info['compute_capability'][0]}.{gpu_info['compute_capability'][1]}")
                print(f"  Total Memory: {gpu_info['total_memory']:.2f} GB")
                print(f"  CUDA Version: {gpu_info['cuda_version']}")
                print(f"  PyTorch Version: {gpu_info['pytorch_version']}")
        else:
            print("✗ Some checks failed. Please review the errors above.")

        print("=" * 60)

    return all_passed


def print_system_info():
    """Print detailed system information for debugging."""
    print("System Information")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")

    try:
        import torch
        print(f"\nPyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}:")
                print(f"  Name: {torch.cuda.get_device_name(i)}")
                print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
    except ImportError:
        print("\nPyTorch is not installed")
    except Exception as e:
        print(f"\nError getting PyTorch info: {e}")

    print("=" * 60)
