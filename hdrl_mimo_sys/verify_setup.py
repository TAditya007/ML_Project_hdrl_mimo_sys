#!/usr/bin/env python3
"""
verify_setup.py - Verify all installed packages and system status
Run: python verify_setup.py
"""

import sys
import platform
import os

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_status(name, status, emoji="✅"):
    print(f"  {emoji} {name}: {status}")

def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, "not installed"

def main():
    torch = None  # Initialize so it's always bound

    print_header("SYSTEM INFORMATION")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Python: {sys.version}")
    print(f"  Architecture: {platform.machine()}")

    print_header("PACKAGE VERSIONS")

    packages = {
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "gymnasium": "gymnasium",
        "stable_baselines3": "stable_baselines3",
        "torch": "torch",
        "jupyterlab": "jupyterlab",
        "tensorboard": "tensorboard",
        "pyyaml": "yaml",
        "tqdm": "tqdm",
        "sklearn": "sklearn",
        "h5py": "h5py",
    }

    all_ok = True
    for pkg_name, import_name in packages.items():
        ok, version = check_package(pkg_name, import_name)
        status = version if ok else "MISSING"
        emoji = "✅" if ok else "❌"
        if not ok:
            all_ok = False
        print_status(pkg_name, status, emoji)

    print_header("PYTORCH GPU STATUS")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_available}")

        if cuda_available:
            print(f"  GPU device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print("\n  🚀 GPU training is ENABLED - Your RTX 4060 is ready!")
        else:
            print("  ❌ CUDA not available - Check NVIDIA drivers")
    except ImportError:
        print("  ❌ PyTorch not installed")
        all_ok = False

    print_header("PROJECT STRUCTURE")
    current_dir = os.getcwd()
    print(f"  Project root: {current_dir}")

    dirs_to_check = ['src', 'notebooks', 'data', 'configs', 'results', 'logs', 'tests']
    for d in dirs_to_check:
        exists = os.path.exists(d)
        emoji = "✅" if exists else "❌"
        print(f"  {emoji} {d}/")

    print_header("VERIFICATION COMPLETE")

    if all_ok and torch is not None and torch.cuda.is_available():
        print("  🎉 ALL SYSTEMS READY! Your RTX 4060 is ready for training!")
    elif all_ok:
        print("  ⚠️  All packages installed but GPU not available.")
    else:
        print("  ⚠️  Some packages are missing. Run: pip install -r requirements.txt")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
    