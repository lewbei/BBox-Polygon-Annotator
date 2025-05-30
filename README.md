# BBox & Polygon Annotator

MIT License Python 3.8+ Platform: Windows/Linux/macOS

A comprehensive, Tkinter-based image annotation tool supporting both bounding box and polygon annotations for efficient computer vision dataset creation. Features intelligent auto-annotation using YOLO, flexible export formats, robust project management, and an intuitive GUI.

Tip: Add a demo screenshot or GIF here to highlight your UI!

## Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Export Formats](#export-formats)
- [Development Status & Roadmap](#development-status--roadmap)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Dual Annotation:** Bounding box & polygon support
- **AI Integration:** One-click YOLO auto-annotation with confidence filtering
- **Multi-Format Export:** COCO JSON, Pascal VOC XML, CSV, YAML (YOLO), and Generic JSON
- **Project Management:** Status tracking, undo/redo, copy/paste
- **Real-time Visual Feedback:** Class-specific coloring, annotation preview
- **Keyboard Shortcuts:** Efficient navigation and annotation
- **Fully Open Source:** Python, Tkinter, OpenCV, PIL

## Quick Start
### Requirements
- Python 3.8+
- Tkinter (usually included with Python)
- OpenCV
- Pillow (PIL)
- Ultralytics YOLO (optional, for auto-annotation)

### Installation
```bash
git clone https://github.com/lewbei/BBox-Polygon-Annotator.git
cd BBox-Polygon-Annotator
pip install -r requirements.txt
```

### Launch
```bash
python -m image_labelling.main
```

## Usage
- **Load Images:** Select a folder or project to start annotating.
- **Annotate:** Use the toolbar to switch between bounding box and polygon modes. Draw directly on the image canvas.
- **AI Auto-Annotation:** Use "Auto Annotate" for YOLO-powered suggestions (ensure model is loaded).
- **Manage Classes:** Add, remove, or update object classes.
- **Project Tools:** Track progress, undo/redo actions, copy/paste annotations.
- **Export:** Choose from five export formats via the toolbar.

### Keyboard Shortcuts:
- Arrow keys: Navigate images
- 1-9: Quick class selection
- Ctrl+S: Save project
- Ctrl+Z/Y: Undo/Redo
- Delete: Remove selected annotation

## Export Formats
| Format      | Use Case                       | Output Files          |
|-------------|--------------------------------|-----------------------|
| YAML (YOLO) | Direct YOLO model training     | `dataset.yaml` + folders |
| COCO JSON   | Industry-standard object detection | `annotations.json`    |
| Pascal VOC  | Classic CV workflows           | `.xml` per image      |
| CSV         | Manual review, spreadsheet analysis | `annotations.csv`     |
| Generic JSON| Custom integrations            | `annotations.json`    |

All formats support both bounding box and polygon annotations.

## Development Status & Roadmap
- **Current Version:** v9 (Polygon support complete)
- **Recently Added:** Advanced polygon editing, multi-format export, project management fixes
- **In Progress:** Performance optimization, error handling, UI/UX polish, responsive layout, background operations
- **Planned:** Batch operations, analytics, plugin support, PyQt migration, cloud integration

## Contributing
Contributions are welcome! Please open issues or pull requests. For major changes, consider discussing them first.

## License
This project is licensed under the MIT License.
