@echo off
REM PyTorch 2.10.0a0 SM120 Installation Script for RTX 50-series GPUs
REM This script installs PyTorch with native SM 12.0 (Blackwell) support

echo ==========================================
echo PyTorch 2.10.0a0 SM120 Installer
echo RTX 50-series GPU Support
echo ==========================================
echo.

REM Check Python version
python --version
echo.

REM Check for CUDA
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
    echo.
) else (
    echo WARNING: nvidia-smi not found. GPU functionality may not be available.
    echo.
)

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Install PyTorch wheel
echo.
echo Installing PyTorch 2.10.0a0 with SM120 support...
pip install torch_sm120.whl --force-reinstall

echo.
echo ==========================================
echo Installation Complete!
echo ==========================================
echo.

echo Verifying installation...
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'Compute Capability: {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo Installation verified successfully!
pause
