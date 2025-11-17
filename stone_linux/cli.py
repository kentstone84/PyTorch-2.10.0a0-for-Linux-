"""Command-line interface for stone-linux."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from stone_linux import __version__
from stone_linux.installer import install_pytorch_wheel, check_wheel_exists_locally
from stone_linux.verify import verify_installation, print_system_info


def verify_installation_cli():
    """CLI entry point for stone-verify command."""
    parser = argparse.ArgumentParser(
        description="Verify PyTorch RTX 50-series installation"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Minimal output"
    )
    parser.add_argument(
        "--system-info",
        action="store_true",
        help="Print detailed system information"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"stone-linux {__version__}"
    )

    args = parser.parse_args()

    if args.system_info:
        print_system_info()
        sys.exit(0)

    result = verify_installation(verbose=not args.quiet)

    if result:
        sys.exit(0)
    else:
        sys.exit(1)


def install_pytorch():
    """CLI entry point for stone-install command."""
    parser = argparse.ArgumentParser(
        description="Install PyTorch with RTX 50-series (SM 12.0) support"
    )
    parser.add_argument(
        "--python-version",
        type=str,
        default=None,
        help="Python version to install for (e.g., 3.12). Defaults to current version."
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download wheel but don't install"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reinstall if already installed"
    )
    parser.add_argument(
        "--wheel",
        type=str,
        default=None,
        help="Path to existing wheel file to install"
    )
    parser.add_argument(
        "--check-local",
        action="store_true",
        help="Check for wheel in common locations (Downloads, current directory)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"stone-linux {__version__}"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PyTorch RTX 50-series Installer")
    print(f"stone-linux v{__version__}")
    print("=" * 60)

    wheel_path: Optional[Path] = None

    # Check for local wheel if requested
    if args.check_local:
        print("\nChecking for wheel file in common locations...")
        wheel_path = check_wheel_exists_locally()
        if wheel_path:
            print(f"✓ Found wheel at: {wheel_path}")
            use_local = input("Use this wheel? [Y/n]: ").strip().lower()
            if use_local in ['n', 'no']:
                wheel_path = None
        else:
            print("No local wheel found. Will download from GitHub.")

    # Use specified wheel path if provided
    if args.wheel:
        wheel_path = Path(args.wheel)

    # Install
    success = install_pytorch_wheel(
        python_version=args.python_version,
        download_only=args.download_only,
        force=args.force,
        wheel_path=wheel_path,
    )

    print("=" * 60)

    if success:
        if not args.download_only:
            print("\n✓ Installation complete!")
            print("\nTo verify your installation, run:")
            print("  stone-verify")
            print("\nOr in Python:")
            print("  import stone_linux")
            print("  stone_linux.verify_installation()")
        sys.exit(0)
    else:
        print("\n✗ Installation failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    # For testing
    print("This module provides CLI commands.")
    print("Use 'stone-verify' or 'stone-install' after installing the package.")
