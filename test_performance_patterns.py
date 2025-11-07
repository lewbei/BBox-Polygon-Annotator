#!/usr/bin/env python3
"""
Test that performance improvements are correctly implemented.
Validates code patterns without requiring dependencies.
"""

import os
import re

def test_lazy_loading_in_editor():
    """Verify that editor.py uses lazy loading instead of eager imports."""
    print("\nTesting lazy loading in editor.py...")

    with open("image_labelling/editor.py", "r") as f:
        content = f.read()

    # Check that PIL and numpy are NOT eagerly imported at top level
    eager_pil_import = re.search(r'^from PIL import Image, ImageTk', content, re.MULTILINE)
    eager_numpy_import = re.search(r'^import numpy as np', content, re.MULTILINE)

    if eager_pil_import:
        print("  ✗ FAIL: Found eager PIL import (should be commented/removed)")
        return False

    if eager_numpy_import:
        print("  ✗ FAIL: Found eager numpy import (should be commented/removed)")
        return False

    # Check that _get_pil() helper function exists
    if '_get_pil()' not in content:
        print("  ✗ FAIL: _get_pil() helper function not found")
        return False

    # Check that lazy_importer is used
    if 'lazy_importer.get_pil()' not in content:
        print("  ✗ FAIL: lazy_importer.get_pil() not used")
        return False

    # Check for usage of Image and ImageTk through _get_pil()
    if 'Image, ImageTk = _get_pil()' not in content and 'Image, _ = _get_pil()' not in content:
        print("  ✗ FAIL: _get_pil() not used to get PIL components")
        return False

    print("  ✓ PASS: Lazy loading correctly implemented")
    return True

def test_cv2_import_optimization():
    """Verify that cv2 is imported outside the loop in exporter.py."""
    print("\nTesting cv2 import optimization in exporter.py...")

    with open("image_labelling/exporter.py", "r") as f:
        lines = f.readlines()

    # Find the convert_to_coco_format function
    in_function = False
    found_loop = False
    cv2_import_line = None
    loop_line = None

    for i, line in enumerate(lines):
        if 'def convert_to_coco_format' in line:
            in_function = True
            continue

        if in_function:
            if 'import cv2' in line and not line.strip().startswith('#'):
                cv2_import_line = i

            if 'for img_idx, image_path in enumerate(image_files)' in line:
                loop_line = i
                found_loop = True
                break

    if not found_loop:
        print("  ✗ FAIL: Could not find the image loop")
        return False

    if cv2_import_line is None:
        print("  ✗ FAIL: cv2 import not found in convert_to_coco_format")
        return False

    if cv2_import_line > loop_line:
        print(f"  ✗ FAIL: cv2 imported inside loop (line {cv2_import_line+1}) instead of before (line {loop_line+1})")
        return False

    print(f"  ✓ PASS: cv2 imported at line {cv2_import_line+1}, before loop at line {loop_line+1}")
    return True

def test_canvas_throttling():
    """Verify that canvas throttling mechanism is implemented."""
    print("\nTesting canvas redraw throttling in editor.py...")

    with open("image_labelling/editor.py", "r") as f:
        content = f.read()

    # Check for throttling variables
    if '_pending_redraw' not in content:
        print("  ✗ FAIL: _pending_redraw variable not found")
        return False

    if '_redraw_throttle_ms' not in content:
        print("  ✗ FAIL: _redraw_throttle_ms variable not found")
        return False

    # Check for throttling methods
    if 'def _schedule_display_image(self):' not in content:
        print("  ✗ FAIL: _schedule_display_image() method not found")
        return False

    if 'def _execute_display_image(self):' not in content:
        print("  ✗ FAIL: _execute_display_image() method not found")
        return False

    # Check that throttling is used in pan_drag
    if '_schedule_display_image()' not in content:
        print("  ✗ FAIL: _schedule_display_image() not called anywhere")
        return False

    # Verify it's used in the right places
    pan_drag_match = re.search(r'def on_pan_drag\(.*?\):(.*?)(?=\n    def |\Z)', content, re.DOTALL)
    if pan_drag_match and '_schedule_display_image' in pan_drag_match.group(1):
        print("  ✓ PASS: Throttling used in on_pan_drag")
    else:
        print("  ⚠ WARNING: Throttling might not be used in on_pan_drag")

    canvas_resize_match = re.search(r'def on_canvas_resize\(.*?\):(.*?)(?=\n    def |\Z)', content, re.DOTALL)
    if canvas_resize_match and '_schedule_display_image' in canvas_resize_match.group(1):
        print("  ✓ PASS: Throttling used in on_canvas_resize")
    else:
        print("  ⚠ WARNING: Throttling might not be used in on_canvas_resize")

    print("  ✓ PASS: Canvas throttling mechanism implemented")
    return True

def test_file_cache_infrastructure():
    """Verify that file cache infrastructure is in place."""
    print("\nTesting file cache infrastructure in editor.py...")

    with open("image_labelling/editor.py", "r") as f:
        content = f.read()

    if '_file_exists_cache' not in content:
        print("  ✗ FAIL: _file_exists_cache not found")
        return False

    print("  ✓ PASS: File cache infrastructure in place")
    return True

def test_no_regressions():
    """Test that critical functions still exist."""
    print("\nTesting for regressions...")

    with open("image_labelling/editor.py", "r") as f:
        editor_content = f.read()

    with open("image_labelling/exporter.py", "r") as f:
        exporter_content = f.read()

    # Critical functions should still exist
    critical_functions = {
        "editor.py": [
            "def display_image(",
            "def load_image(",
            "def on_pan_drag(",
            "def on_zoom(",
            "class BoundingBoxEditor"
        ],
        "exporter.py": [
            "def convert_to_coco_format(",
            "def convert_to_pascal_voc_format(",
            "def convert_to_csv_format("
        ]
    }

    all_found = True

    for func in critical_functions["editor.py"]:
        if func not in editor_content:
            print(f"  ✗ FAIL: Missing {func} in editor.py")
            all_found = False

    for func in critical_functions["exporter.py"]:
        if func not in exporter_content:
            print(f"  ✗ FAIL: Missing {func} in exporter.py")
            all_found = False

    if all_found:
        print("  ✓ PASS: All critical functions present")

    return all_found

def main():
    """Run all pattern tests."""
    print("=" * 60)
    print("PERFORMANCE PATTERNS VALIDATION")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(("Lazy Loading", test_lazy_loading_in_editor()))
    results.append(("CV2 Import Optimization", test_cv2_import_optimization()))
    results.append(("Canvas Throttling", test_canvas_throttling()))
    results.append(("File Cache Infrastructure", test_file_cache_infrastructure()))
    results.append(("No Regressions", test_no_regressions()))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if passed == total else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
