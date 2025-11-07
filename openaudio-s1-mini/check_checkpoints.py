#!/usr/bin/env python3
"""
Quick script to check checkpoint directory structure
"""
import os
import sys

checkpoint_paths = [
    "checkpoints/openaudio-s1-mini",
    os.path.expanduser("~/openaudio_checkpoints/openaudio-s1-mini"),
]

print("Checking for checkpoint directories...\n")

for path in checkpoint_paths:
    print(f"Checking: {path}")
    if os.path.exists(path):
        print(f"  ✓ Directory exists")
        files = os.listdir(path)
        print(f"  Files found: {len(files)}")
        for f in sorted(files)[:20]:  # Show first 20 files
            filepath = os.path.join(path, f)
            size = os.path.getsize(filepath) if os.path.isfile(filepath) else 0
            print(f"    - {f} ({size:,} bytes)" if size > 0 else f"    - {f}/")
        
        # Check for required files
        required = ["config.json", "codec.pth"]
        print(f"\n  Required files:")
        for req in required:
            req_path = os.path.join(path, req)
            exists = os.path.exists(req_path)
            print(f"    {'✓' if exists else '✗'} {req}")
        
        if len(files) > 20:
            print(f"    ... and {len(files) - 20} more files")
    else:
        print(f"  ✗ Directory does not exist")
    print()

