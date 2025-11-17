#!/usr/bin/env python3
"""
stone-linux: PyTorch 2.10 with native SM 12.0 support for RTX 50-series GPUs

This package provides a convenient installer and utilities for PyTorch
compiled with native Blackwell architecture (SM 12.0) support.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

long_description = read_file('README.md')

setup(
    name='stone-linux',
    use_scm_version=False,
    version='2.10.0a0',
    description='PyTorch 2.10 with native SM 12.0 (Blackwell) support for NVIDIA RTX 50-series GPUs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='PyTorch RTX Community',
    author_email='kentstone84@users.noreply.github.com',
    url='https://github.com/kentstone84/PyTorch-2.10.0a0-for-Linux-',
    packages=find_packages(exclude=['tests', 'tests.*', 'examples.*']),
    include_package_data=True,
    python_requires='>=3.10',
    install_requires=[
        'numpy>=2.0.0',
        'packaging>=24.0',
        'PyYAML>=6.0',
        'typing-extensions>=4.8.0',
        'requests>=2.31.0',
        'tqdm>=4.66.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'black>=23.0.0',
            'ruff>=0.1.0',
            'mypy>=1.7.0',
        ],
        'examples': [
            'jupyter>=1.0.0',
            'notebook>=7.0.0',
            'matplotlib>=3.8.0',
            'pandas>=2.1.0',
        ],
        'vllm': ['vllm>=0.6.0'],
        'langchain': ['langchain>=0.1.0', 'langchain-community>=0.0.1'],
    },
    entry_points={
        'console_scripts': [
            'stone-verify=stone_linux.cli:verify_installation',
            'stone-install=stone_linux.cli:install_pytorch',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='pytorch machine-learning deep-learning gpu cuda rtx-5090 rtx-5080 blackwell sm120 nvidia',
    project_urls={
        'Documentation': 'https://github.com/kentstone84/PyTorch-2.10.0a0-for-Linux-/blob/main/README.md',
        'Source': 'https://github.com/kentstone84/PyTorch-2.10.0a0-for-Linux-',
        'Issues': 'https://github.com/kentstone84/PyTorch-2.10.0a0-for-Linux-/issues',
        'Changelog': 'https://github.com/kentstone84/PyTorch-2.10.0a0-for-Linux-/releases',
    },
)
