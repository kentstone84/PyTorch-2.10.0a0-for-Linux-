# Changelog

All notable changes to stone-linux will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.10.0a0] - 2025-11-17

### Added

- 🎉 Initial release of stone-linux PyPI package
- PyPI package distribution for easy installation (`pip install stone-linux`)
- Command-line tools: `stone-install` and `stone-verify`
- Automated PyTorch wheel download and installation
- Multi-Python version support (3.10, 3.11, 3.12, 3.13, 3.14)
- Comprehensive verification utilities
- GitHub Actions CI/CD workflows:
  - Automated testing and linting
  - PyPI publishing workflow
  - Multi-version wheel building
  - Docker image building
  - Performance benchmarking
- GitHub issue templates (bug report, feature request, question)
- Pull request template
- Integration examples:
  - vLLM integration example
  - LangChain integration example
- Jupyter notebooks:
  - Getting started tutorial
  - Performance benchmarking notebook
- Benchmarking utilities and scripts
- Complete API documentation

### Features

- Native SM 12.0 (Blackwell) support for RTX 50-series GPUs
- CUDA 13.0+ compatibility
- Automatic Python version detection
- GPU compatibility verification
- Performance benchmarking tools
- Mixed precision training support
- Memory management utilities

### Documentation

- Updated README with pip installation instructions
- Examples and tutorials section
- Performance metrics and benchmarks
- Troubleshooting guide
- API reference documentation

### Infrastructure

- PyPI publishing workflow
- Automated testing on multiple Python versions
- Docker build automation
- GitHub Actions for CI/CD
- Issue and PR templates for community contributions

## Future Releases

### Planned for v2.11.0

- Triton 3.3+ wheel integration
- Additional model zoo examples
- Multi-GPU distributed training examples
- Performance dashboard
- Video tutorials
- RTX 5070 extended support

### Planned for v3.0.0

- Breaking: Simplified API
- Enhanced CLI with interactive mode
- Web-based performance dashboard
- Automated model optimization tools
- Integration with more ML frameworks

---

## Release Notes

### v2.10.0a0 - Initial Release

This is the first public release of stone-linux, making PyTorch with native RTX 50-series (Blackwell) support easily accessible through PyPI.

**Installation:**
```bash
pip install stone-linux
stone-install
```

**Key Features:**
- 20-30% performance improvement over standard PyTorch builds
- Native SM 12.0 support
- Easy installation and verification
- Comprehensive examples and documentation

**Compatibility:**
- Python 3.10, 3.11, 3.12, 3.13, 3.14
- NVIDIA Driver >= 570.00
- RTX 5090, 5080, 5070 Ti, 5070

For detailed information, see the [README](README.md).
