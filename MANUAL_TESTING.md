# Manual Testing Guide for Performance Improvements

Since this is a GUI application that requires dependencies (tkinter, OpenCV, PIL) not available in the test environment, here's a guide for manual testing with actual images.

## Prerequisites

Ensure you have all dependencies installed:
```bash
pip install opencv-python Pillow PyYAML numpy ultralytics
```

## Test 1: Startup Performance

**What to test**: Application should start faster

**Steps**:
1. Close the application if running
2. Time the startup:
   ```bash
   time python -m image_labelling.main
   ```
3. Compare with previous startup time (should be 200-500ms faster)

**Expected**: Application window appears noticeably faster

---

## Test 2: Image Loading

**What to test**: First image load triggers lazy imports

**Steps**:
1. Launch the application
2. Create a new project or open an existing one
3. Load an image (first image load will trigger PIL/cv2 imports)
4. Verify image displays correctly
5. Navigate to another image - should be instant

**Expected**:
- First image takes slightly longer (lazy loading)
- Subsequent images load instantly
- No errors in console

---

## Test 3: Pan Performance

**What to test**: Smooth panning on large images with throttling

**Steps**:
1. Open a large image (preferably 4K+ resolution)
2. Zoom in using Ctrl+MouseWheel
3. Pan the image using middle-mouse button and drag
4. Observe smoothness and responsiveness

**Expected**:
- Smooth panning at ~60 FPS
- No stuttering or lag
- Canvas updates feel responsive

**Before/After Comparison**:
- Before: Noticeable lag, choppy panning
- After: Smooth, fluid panning

---

## Test 4: Zoom Performance

**What to test**: Smooth zooming with throttled redraws

**Steps**:
1. Open any image
2. Use Ctrl+MouseWheel to zoom in and out repeatedly
3. Test both zoom in and zoom out
4. Try rapid zoom changes

**Expected**:
- Smooth zoom transitions
- No excessive CPU usage
- Responsive controls

---

## Test 5: Export to COCO Format

**What to test**: Faster export with optimized cv2 import

**Steps**:
1. Open a project with many images (100+)
2. Add annotations to several images
3. Click export button and select COCO format
4. Time the export process
5. Verify exported files are correct

**Expected**:
- Export completes 5-15% faster (more noticeable with 1000+ images)
- All annotations exported correctly
- Generated JSON is valid

---

## Test 6: Annotation Workflow

**What to test**: Normal annotation functions still work

**Steps**:
1. Create bounding boxes on images
2. Create polygon annotations
3. Edit existing annotations (move, resize, delete)
4. Switch between images
5. Save annotations (Ctrl+S)
6. Close and reopen project - verify annotations persist

**Expected**:
- All annotation tools work normally
- No errors or crashes
- Annotations save and load correctly

---

## Test 7: Auto-annotation (if YOLO model available)

**What to test**: YOLO auto-annotation still works

**Steps**:
1. Load a YOLO model (.pt file)
2. Run auto-annotation on dataset
3. Verify predictions appear as annotations
4. Check console for any import errors

**Expected**:
- Auto-annotation completes successfully
- Predictions displayed correctly
- No lazy-loading related errors

---

## Quick Smoke Test

If you're short on time, run this minimal test:

```bash
# 1. Start application
python -m image_labelling.main

# 2. Create a test project pointing to any folder with images

# 3. Open an image - should display without errors

# 4. Draw a bounding box - should work normally

# 5. Zoom and pan - should feel smooth

# 6. Export to any format - should complete successfully
```

---

## Automated Tests

Run the automated validation tests:

```bash
# Syntax validation
python test_syntax.py

# Performance patterns validation
python test_performance_patterns.py
```

Both should show all tests passing.

---

## Performance Metrics to Watch

Monitor these during testing:

1. **Startup Time**: Should be 20-40% faster
2. **Memory Usage**: Should be similar or slightly lower initially
3. **CPU During Pan**: Should be lower and more consistent
4. **Export Time**: Noticeable on 1000+ images

---

## Troubleshooting

### Issue: "No module named tkinter"
- **Cause**: tkinter not installed (Linux)
- **Fix**: `sudo apt-get install python3-tk`

### Issue: "No module named cv2"
- **Cause**: OpenCV not installed
- **Fix**: `pip install opencv-python`

### Issue: Images not loading
- **Cause**: PIL import issue
- **Fix**: `pip install Pillow --upgrade`

### Issue: Lag during panning
- **Check**: CPU usage - should be moderate
- **Adjust**: Modify `_redraw_throttle_ms` in editor.py (line ~160) to higher value

---

## What to Look For

**Good Signs** ✓:
- Application starts quickly
- Pan/zoom feels smooth
- No console errors
- All features work as before

**Bad Signs** ✗:
- Import errors on startup
- Images fail to load
- Lag during pan/zoom
- Annotation tools broken
- Export fails

---

## Reporting Issues

If you find any issues:

1. Note the exact steps to reproduce
2. Check console output for errors
3. Include Python version: `python --version`
4. Include OS: Windows/Linux/macOS
5. Note which test failed

Then report in the issue tracker or revert changes if critical.
