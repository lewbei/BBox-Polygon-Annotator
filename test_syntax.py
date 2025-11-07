#!/usr/bin/env python3
"""
Basic syntax validation test that doesn't require dependencies.
"""

import sys
import os
import ast

def test_syntax(filepath):
    """Test that a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("SYNTAX VALIDATION")
    print("=" * 60)

    files_to_test = [
        "image_labelling/editor.py",
        "image_labelling/exporter.py",
        "image_labelling/helpers.py",
        "image_labelling/startup_optimizer.py",
        "image_labelling/main.py",
    ]

    all_passed = True

    for filepath in files_to_test:
        if not os.path.exists(filepath):
            print(f"⚠ SKIP: {filepath} (not found)")
            continue

        success, error = test_syntax(filepath)
        if success:
            print(f"✓ PASS: {filepath}")
        else:
            print(f"✗ FAIL: {filepath}")
            print(f"  Error: {error}")
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("All files have valid Python syntax!")
        return 0
    else:
        print("Some files have syntax errors!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
