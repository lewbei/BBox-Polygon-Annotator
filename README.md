# BBox & Polygon Annotator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)](https://github.com/lewbei/BBox-Polygon-Annotator)

A **production-ready**, comprehensive image annotation tool designed for computer vision dataset creation. Built with Python and Tkinter, this professional-grade application supports both bounding box and polygon annotations with advanced features including AI-powered auto-annotation, comprehensive export formats, robust project management, and an intuitive user interface.



## 🚀 Key Highlights

- **🎯 Dual Annotation Support**: Complete bounding box and polygon annotation capabilities with advanced editing
- **🤖 AI-Powered Workflow**: YOLO integration for intelligent auto-annotation with confidence filtering
- **📊 Professional Project Management**: Integrated project system with status tracking and progress monitoring
- **🔄 Comprehensive Export Suite**: 5 industry-standard formats (YOLO, COCO, Pascal VOC, CSV, JSON)
- **⚡ Advanced User Experience**: Modern UI with icons, zoom/pan, keyboard shortcuts, auto-save, undo/redo
- **🏗️ Production Architecture**: Modular codebase with proper error handling, logging, and background processing

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Management](#-project-management)
- [Annotation Workflow](#-annotation-workflow)
- [Export Formats](#-export-formats)
- [Advanced Features](#-advanced-features)
- [Keyboard Shortcuts](#-keyboard-shortcuts)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Development & Contributing](#-development--contributing)
- [Roadmap](#-roadmap)
- [License](#-license)

## ✨ Features

### Core Annotation Capabilities
- **Bounding Box Annotation**: Precise rectangular annotations with drag-to-resize and move functionality
- **Polygon Annotation**: Advanced polygon support with vertex editing, deletion, and movement
- **Multi-Class Support**: Unlimited object classes with color-coded visualization
- **Real-time Visual Feedback**: Live annotation preview with class-specific colors and labels
- **Annotation Management**: Copy, paste, delete, and modify annotations with full undo/redo support

### AI-Powered Workflow
- **YOLO Auto-Annotation**: One-click automatic annotation using pre-trained or custom YOLO models
- **Confidence Filtering**: Configurable confidence thresholds for quality control
- **Progress Tracking**: Real-time progress indicators for batch auto-annotation operations
- **Model Loading**: Support for custom YOLO model integration

### Professional Project Management
- **Project-Based Workflow**: Organize datasets into manageable projects with metadata tracking
- **Status Tracking**: Automatic image status classification (Not Viewed, Viewed, Labeled, Review Needed)
- **Progress Monitoring**: Real-time statistics and completion tracking
- **Last Modified Tracking**: Automatic timestamp tracking for project files
- **Project Templates**: Quick project creation with standardized folder structures

### Advanced User Interface
- **Modern UI Design**: Clean, professional interface with Unicode icons
- **Responsive Canvas**: Dynamic sizing with zoom (mouse wheel + toolbar) and pan (middle-mouse drag) functionality
- **Resizable Layout**: Flexible panel layout with draggable dividers
- **Status Bar**: Real-time display of annotation statistics and project progress
- **Error Handling**: Comprehensive error management with user-friendly feedback

### Export & Integration
- **5 Export Formats**: YOLO YAML, COCO JSON, Pascal VOC XML, CSV, Generic JSON
- **Flexible Export Options**: Include/exclude images, handle unannotated files, custom export locations
- **Data Splitting**: Configurable train/validation/test splits for machine learning workflows
- **Batch Processing**: Efficient handling of large datasets with background operations

## 🔧 Installation

### Prerequisites
- **Python 3.8 or higher**
- **Operating System**: Windows, Linux, or macOS

### Required Dependencies
```bash
# Core dependencies
tkinter          # GUI framework (usually included with Python)
opencv-python    # Computer vision library
Pillow          # Image processing
PyYAML          # YAML file handling

# Optional dependencies
ultralytics     # For YOLO auto-annotation (install if needed)
```

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/lewbei/BBox-Polygon-Annotator.git
   cd BBox-Polygon-Annotator
   ```

2. **Install Dependencies**
   ```bash
   pip install opencv-python Pillow PyYAML
   
   # Optional: For YOLO auto-annotation
   pip install ultralytics
   ```

3. **Verify Installation**
   ```bash
   python -m image_labelling.main
   ```

## 🚀 Quick Start

### 1. Launch the Application
```bash
python -m image_labelling.main
```

### 2. Create Your First Project
- Click **"New Project"** in the Project Manager
- Enter a project name and select your dataset folder
- The tool will automatically create the necessary folder structure

### 3. Start Annotating
- **Select Images**: Navigate through your dataset using the image list
- **Choose Annotation Mode**: Use toolbar buttons to switch between bounding boxes (⬜) and polygons (🔷)
- **Draw Annotations**: Click and drag to create bounding boxes, or click multiple points for polygons
- **Assign Classes**: Select object classes from the class list or use number keys (1-9)
- **Save Progress**: Use Ctrl+S or enable auto-save for automatic progress saving

## 🗂️ Project Management

### Project Structure
```
YourProject/
├── dataset.yaml          # Project configuration
├── images/              # Source images
├── labels/              # YOLO format annotations
└── status.json          # Image status tracking
```

### Project Manager Features
- **Create Projects**: Set up new annotation projects with custom names and dataset paths
- **Open Projects**: Resume work on existing projects with preserved state
- **Delete Projects**: Remove project metadata (datasets remain intact)
- **Status Overview**: View project statistics and last modified dates

### Image Status System
- **Not Viewed** (🔴): Images not yet opened
- **Viewed** (🟡): Images opened but not annotated
- **Labeled** (🟢): Images with annotations
- **Review Needed** (🟠): Auto-annotated images requiring manual review

## 🎨 Annotation Workflow

### Bounding Box Annotation
1. **Select Mode**: Click the bounding box icon (⬜) in the toolbar
2. **Draw**: Click and drag to create rectangular annotations
3. **Edit**: 
   - **Resize**: Drag corner/edge handles
   - **Move**: Click and drag the annotation
   - **Delete**: Select and press Delete key

### Polygon Annotation
1. **Select Mode**: Click the polygon icon (🔷) in the toolbar
2. **Draw**: Click to place vertices, right-click or press Enter to complete
3. **Edit**: 
   - **Move Vertices**: Drag individual points
   - **Add Vertices**: Click on polygon edges
   - **Delete Vertices**: Select vertex and press Delete
   - **Move Polygon**: Drag from the interior

### Class Management
- **Add Classes**: Type new class names in the class entry field
- **Edit Classes**: Select existing classes to modify
- **Color Coding**: Each class gets a unique color for visual distinction
- **Quick Selection**: Use number keys 1-9 for rapid class switching

## 📤 Export Formats

| Format | Use Case | Output Files | Features |
|--------|----------|--------------|----------|
| **YAML (YOLO)** | Direct YOLO training | `dataset.yaml` + train/val/test folders | Data splitting, validation options |
| **COCO JSON** | Industry standard | `annotations.json` | Full COCO compliance, segmentation support |
| **Pascal VOC** | Classic CV workflows | Individual `.xml` files | Per-image annotations, legacy support |
| **CSV** | Analysis & review | `annotations.csv` | Spreadsheet compatibility, easy review |
| **Generic JSON** | Custom integrations | `annotations.json` | Flexible format for custom pipelines |

### Export Options
- **Data Splitting**: Configure train/validation/test ratios (e.g., 60/20/20)
- **Image Inclusion**: Option to copy images to export directory
- **Unannotated Images**: Include or exclude images without annotations
- **Custom Locations**: Export to any directory location
- **Validation Control**: Include/exclude validation sets in YAML exports

## 🔥 Advanced Features

### Auto-Annotation with YOLO
- **Model Loading**: Load pre-trained or custom YOLO models
- **Batch Processing**: Annotate entire datasets automatically
- **Confidence Thresholds**: Filter predictions by confidence scores
- **Progress Tracking**: Real-time progress with cancel capability
- **Review System**: Flag low-confidence predictions for manual review

### Canvas Controls
- **Zoom**: Mouse wheel or toolbar buttons (🔍+ / 🔍-)
- **Pan**: Middle-mouse drag for navigation
- **Responsive Scaling**: Annotations scale correctly with zoom levels
- **Full-Screen Canvas**: Maximize annotation workspace

### Data Management
- **Auto-Save**: Configurable automatic saving intervals
- **Undo/Redo**: Full action history with Ctrl+Z/Ctrl+Y
- **Copy/Paste**: Duplicate annotations across images
- **Batch Operations**: Planned feature for multi-image operations

## ⌨️ Keyboard Shortcuts

### Navigation
- **Arrow Keys**: Navigate between images
- **Home/End**: Jump to first/last image
- **Page Up/Down**: Skip through images quickly

### Annotation
- **1-9**: Quick class selection
- **Delete**: Remove selected annotation
- **Enter**: Complete polygon drawing
- **Escape**: Cancel current operation

### File Operations
- **Ctrl+S**: Save project
- **Ctrl+Z**: Undo last action
- **Ctrl+Y**: Redo last undone action
- **Ctrl+C/V**: Copy/paste annotations

### View Controls
- **Ctrl+Mouse Wheel**: Zoom in/out
- **Middle Mouse**: Pan canvas
- **F1**: Show keyboard shortcuts help

## ⚙️ Configuration

### Auto-Save Settings
Configure automatic saving in your `dataset.yaml`:
```yaml
auto_save_interval: 300  # Save every 5 minutes (in seconds)
```

### YOLO Model Configuration
```yaml
model_path: "path/to/your/model.pt"  # Custom YOLO model
confidence_threshold: 0.5             # Minimum confidence for auto-annotation
```

### UI Preferences
- **Theme Support**: Automatic theme detection (Clam, Vista, Default)
- **Window Sizing**: Responsive layout with configurable panel weights
- **Icon System**: Unicode-based icons for cross-platform compatibility

## 🛠️ Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
# Ensure you're in the correct directory
cd path/to/BBox-Polygon-Annotator

# Run with proper module syntax
python -m image_labelling.main
```

#### YOLO auto-annotation not working
```bash
# Install ultralytics if not already installed
pip install ultralytics

# Verify model loading in the application
# Check the console/logs for model loading errors
```

#### Images not loading
- **Supported formats**: JPG, JPEG, PNG, BMP, TIFF
- **Path issues**: Ensure no special characters in folder paths
- **Permissions**: Verify read access to image directories

#### Performance issues with large datasets
- **Enable auto-save**: Set reasonable intervals (300-600 seconds)
- **Close other applications**: Free up system memory
- **Use image caching**: Feature planned for future releases

### Error Reporting
If you encounter bugs:
1. Check the `error.log` file in your project directory
2. Note the Python version and operating system
3. Provide steps to reproduce the issue
4. Include relevant log messages

## 👥 Development & Contributing

### Project Architecture
```
image_labelling/
├── __init__.py           # Package initialization
├── main.py              # Application entry point
├── project_manager.py   # Project management UI and logic
├── editor.py           # Main annotation editor
├── exporter.py         # Export format converters
├── helpers.py          # Utility functions
├── constants.py        # Application constants
├── settings.py         # Configuration management
└── startup_optimizer.py # Performance optimizations
```

### Contributing Guidelines
We welcome contributions! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow PEP 8** style guidelines
3. **Add tests** for new functionality
4. **Update documentation** for API changes
5. **Submit a pull request** with a clear description

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/BBox-Polygon-Annotator.git
cd BBox-Polygon-Annotator

# Create a development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run in development mode
python -m image_labelling.main
```

### Code Quality
- **Logging**: Use Python's logging module instead of print statements
- **Error Handling**: Implement comprehensive try-catch blocks
- **Modularity**: Keep functions focused and classes cohesive
- **Documentation**: Add docstrings to all public methods

## 🗺️ Roadmap

### Current Version: v10+ (Production Ready)
**Status**: Mature, feature-complete annotation tool

### Short-Term Goals (Q3-Q4 2025)
- **🎨 UI/UX Enhancements**
  - ✅ Button icons implementation
  - ✅ Responsive canvas with zoom/pan
  - ✅ Keyboard shortcuts dialog
  - ✅ Auto-save mechanism

- **🔧 Performance Optimization**
  - ⏳ Image caching system
  - ⏳ Lazy loading for large datasets
  - ⏳ Background processing improvements

### Mid-Term Goals (2025-2026)
- **📊 Advanced Features**
  - Batch operations for multiple images
  - Annotation analytics and insights
  - Advanced polygon editing tools
  - Image list thumbnails and search

- **🏗️ Architecture Improvements**
  - Resizable panel layouts
  - Plugin architecture
  - Improved state management
  - Comprehensive test coverage

### Long-Term Vision (2026+)
- **🚀 Platform Enhancement**
  - PyQt6/PySide6 migration evaluation
  - Dark mode and advanced theming
  - Cross-platform optimization
  - Cloud integration capabilities

- **🤖 AI Integration**
  - Advanced model integration
  - Active learning workflows
  - Automated quality validation
  - Custom model training pipelines

## 📊 Technical Specifications

### Performance Metrics
- **Startup Time**: < 3 seconds (with splash screen)
- **Memory Usage**: ~50-100MB (typical dataset)
- **Image Loading**: < 1 second (standard resolution)
- **Export Speed**: ~1000 annotations/second

### Supported Formats
**Input Images**: JPG, JPEG, PNG, BMP, TIFF, GIF
**Export Formats**: YOLO YAML, COCO JSON, Pascal VOC XML, CSV, Generic JSON
**Model Support**: YOLO v5/v8/v11 (.pt files)

### System Requirements
- **Minimum**: Python 3.8, 4GB RAM, 1GB storage
- **Recommended**: Python 3.9+, 8GB RAM, SSD storage
- **Display**: 1920x1080 resolution recommended for optimal UI experience

## 🙏 Acknowledgments

### Technologies Used
- **[Tkinter](https://docs.python.org/3/library/tkinter.html)**: GUI framework
- **[OpenCV](https://opencv.org/)**: Computer vision library
- **[Pillow](https://pillow.readthedocs.io/)**: Image processing
- **[Ultralytics](https://ultralytics.com/)**: YOLO model integration
- **[PyYAML](https://pyyaml.org/)**: YAML file handling

### Inspiration
This tool was developed to address the need for a comprehensive, open-source annotation solution that combines ease of use with professional-grade features for computer vision research and development.

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 BBox & Polygon Annotator Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🔗 Links

- **GitHub Repository**: [https://github.com/lewbei/BBox-Polygon-Annotator](https://github.com/lewbei/BBox-Polygon-Annotator)
- **Issues & Bug Reports**: [GitHub Issues](https://github.com/lewbei/BBox-Polygon-Annotator/issues)
- **Documentation**: [Project Wiki](https://github.com/lewbei/BBox-Polygon-Annotator/wiki)
- **Releases**: [GitHub Releases](https://github.com/lewbei/BBox-Polygon-Annotator/releases)

---

**Built with ❤️ for the Computer Vision Community**

*Star ⭐ this repository if you find it useful!*
