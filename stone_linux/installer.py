"""Installer utilities for PyTorch RTX 50-series wheel."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

from stone_linux.config import (
    WHEEL_URLS,
    get_python_version,
    get_wheel_url_for_current_python,
    is_python_version_supported,
)


def get_wheel_url(python_version: Optional[str] = None) -> str:
    """
    Get the wheel URL for a specific Python version.

    Args:
        python_version: Python version (e.g., '3.12'). If None, uses current version.

    Returns:
        URL to the wheel file

    Raises:
        ValueError: If the Python version is not supported
    """
    if python_version is None:
        return get_wheel_url_for_current_python()

    if python_version not in WHEEL_URLS:
        raise ValueError(
            f"Python {python_version} is not supported. "
            f"Supported versions: {', '.join(WHEEL_URLS.keys())}"
        )

    return WHEEL_URLS[python_version]


def download_wheel(
    url: str,
    destination: Optional[Path] = None,
    progress: bool = True
) -> Path:
    """
    Download the PyTorch wheel file.

    Args:
        url: URL to download from
        destination: Where to save the file. If None, uses temp directory
        progress: Show download progress

    Returns:
        Path to the downloaded file
    """
    if destination is None:
        temp_dir = tempfile.gettempdir()
        destination = Path(temp_dir) / "torch_sm120.whl"

    print(f"Downloading PyTorch wheel from:")
    print(f"  {url}")
    print(f"Saving to:")
    print(f"  {destination}")

    if progress:
        try:
            from tqdm import tqdm

            class TqdmUpTo(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)

            with TqdmUpTo(
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc="Downloading"
            ) as t:
                urlretrieve(url, destination, reporthook=t.update_to)

        except ImportError:
            # Fallback if tqdm is not available
            print("Downloading... (this may take a while, ~180 MB)")
            urlretrieve(url, destination)
    else:
        urlretrieve(url, destination)

    print(f"✓ Download complete: {destination}")
    return destination


def install_wheel(wheel_path: Path, force: bool = False) -> bool:
    """
    Install the PyTorch wheel using pip.

    Args:
        wheel_path: Path to the wheel file
        force: Force reinstall if already installed

    Returns:
        True if installation succeeded, False otherwise
    """
    if not wheel_path.exists():
        print(f"✗ Error: Wheel file not found: {wheel_path}")
        return False

    print(f"\nInstalling PyTorch from {wheel_path}...")

    cmd = [sys.executable, "-m", "pip", "install"]

    if force:
        cmd.append("--force-reinstall")

    cmd.append(str(wheel_path))

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print("✓ PyTorch installation complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Installation failed:")
        print(e.stderr)
        return False


def install_pytorch_wheel(
    python_version: Optional[str] = None,
    download_only: bool = False,
    force: bool = False,
    wheel_path: Optional[Path] = None,
) -> bool:
    """
    Download and install the PyTorch wheel for RTX 50-series.

    Args:
        python_version: Python version to install for. If None, uses current version.
        download_only: Only download, don't install
        force: Force reinstall if already installed
        wheel_path: Use existing wheel file instead of downloading

    Returns:
        True if successful, False otherwise
    """
    # Check Python version
    if not is_python_version_supported():
        print(f"✗ Error: Python {get_python_version()} is not supported")
        print(f"  Supported versions: {', '.join(WHEEL_URLS.keys())}")
        return False

    # Use existing wheel or download
    if wheel_path is None:
        try:
            url = get_wheel_url(python_version)
            wheel_path = download_wheel(url)
        except Exception as e:
            print(f"✗ Error downloading wheel: {e}")
            return False
    else:
        if not wheel_path.exists():
            print(f"✗ Error: Wheel file not found: {wheel_path}")
            return False

    # Install if requested
    if not download_only:
        return install_wheel(wheel_path, force=force)
    else:
        print(f"\n✓ Wheel downloaded to: {wheel_path}")
        print("To install, run:")
        print(f"  pip install {wheel_path}")
        return True


def check_wheel_exists_locally(wheel_name: str = "torch_sm120.whl") -> Optional[Path]:
    """
    Check if the wheel file exists in common locations.

    Args:
        wheel_name: Name of the wheel file to look for

    Returns:
        Path to wheel if found, None otherwise
    """
    search_paths = [
        Path.cwd() / wheel_name,
        Path.home() / "Downloads" / wheel_name,
        Path(tempfile.gettempdir()) / wheel_name,
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None
