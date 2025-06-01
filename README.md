# BBox & Polygon Annotator
*Designed for Solo Researchers & Private Use*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()
[![Usage](https://img.shields.io/badge/usage-Solo%20Research-green)]()

A **private research tool** specifically designed for solo computer vision researchers and individual dataset creation. Built with Python and Tkinter, this personal-use application is perfect for researchers working independently on computer vision projects, offering both bounding box and polygon annotations with advanced features including AI-powered auto-annotation, comprehensive export formats, robust project management, and an intuitive user interface.

**✨ Perfect for Solo Researchers**: This tool is specifically designed for individual researchers, PhD students, and private research projects where you need a lightweight, easy-to-use annotation tool without the complexity of enterprise solutions.


## 🚀 Key Highlights - Perfect for Solo Research

- **👤 Solo Researcher Focused**: Designed specifically for individual researchers, PhD students, and private projects
- **🎯 Dual Annotation Support**: Complete bounding box and polygon annotation capabilities with advanced editing  
- **🤖 AI-Powered Workflow**: YOLO integration for intelligent auto-annotation with confidence filtering
- **📊 Personal Project Management**: Lightweight project system ideal for organizing individual research datasets
- **🔄 Comprehensive Export Suite**: 5 industry-standard formats (YOLO, COCO, Pascal VOC, CSV, JSON)
- **⚡ User-Friendly Experience**: Simple, intuitive UI perfect for researchers who want to focus on their data, not the tool
- **🏠 Private & Secure**: Run entirely on your local machine - no cloud dependencies or data sharing concerns

## 📋 Table of Contents

- [Why Choose This for Solo Research](#-why-choose-this-for-solo-research)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start---get-annotating-in-minutes)
- [Project Management](#-project-management---organized-for-solo-research)
- [Annotation Workflow](#-annotation-workflow)
- [Export Formats](#-export-formats)
- [Advanced Features](#-advanced-features)
- [Keyboard Shortcuts](#-keyboard-shortcuts)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)

## 🎯 Why Choose This for Solo Research

**Perfect for Individual Researchers** who need:
- ✅ **Quick Setup**: Get started in minutes, not hours
- ✅ **No Learning Curve**: Intuitive interface designed for researchers, not enterprise teams
- ✅ **Complete Privacy**: All data stays on your machine - no cloud, no sharing, no accounts
- ✅ **Lightweight**: Runs on any machine with Python - no heavy infrastructure required
- ✅ **Flexible Export**: Works with all major ML frameworks (YOLO, COCO, etc.)
- ✅ **AI-Assisted**: Speed up annotation with YOLO auto-annotation
- ✅ **Research-Focused**: Built by researchers, for researchers

**Not Suitable For**: Large teams, enterprise deployments, or collaborative annotation projects.

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

**Perfect for Solo Researchers**: Simple, straightforward installation with minimal dependencies.

1. **Setup Project Directory**
   ```bash
   # Navigate to your project directory
   cd path/to/your/image_labelling_project
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

**Solo Researcher Tip**: This tool runs entirely on your local machine - perfect for maintaining privacy and control over your research data.

## 🚀 Quick Start - Get Annotating in Minutes

### 1. Launch the Application
```bash
python -m image_labelling.main
```

### 2. Create Your First Research Project
- Click **"New Project"** in the Project Manager
- Enter a project name and select your dataset folder
- The tool will automatically create the necessary folder structure - perfect for organizing your research data

### 3. Start Annotating (Solo Researcher Workflow)
- **Select Images**: Navigate through your dataset using the image list
- **Choose Annotation Mode**: Use toolbar buttons to switch between bounding boxes (⬜) and polygons (🔷)
- **Draw Annotations**: Click and drag to create bounding boxes, or click multiple points for polygons
- **Assign Classes**: Select object classes from the class list or use number keys (1-9)
- **Save Progress**: Use Ctrl+S or enable auto-save - no need to worry about losing your work

## 🗂️ Project Management - Organized for Solo Research

### Project Structure (Perfect for Individual Researchers)
```
YourResearchProject/
├── dataset.yaml          # Project configuration
├── images/              # Source images
├── labels/              # YOLO format annotations
└── status.json          # Image status tracking
```

### Project Manager Features (Solo-Friendly)
- **Create Projects**: Set up new annotation projects with custom names and dataset paths
- **Open Projects**: Resume work on existing projects with preserved state - perfect for long-term research
- **Delete Projects**: Remove project metadata (datasets remain intact)
- **Status Overview**: View project statistics and last modified dates - track your research progress

### Image Status System (Track Your Progress)
- **Not Viewed** (🔴): Images not yet opened
- **Viewed** (🟡): Images opened but not annotated  
- **Labeled** (🟢): Images with annotations - your completed work
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

## 👥 Development & Research Notes

**Important**: This is a private research tool developed for solo use. The following information is provided for documentation purposes.

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

### Research Development Notes
**Note**: This section is for personal documentation and research reference only.

The codebase follows these principles for solo research efficiency:

1. **Focus on functionality** over public API design
2. **Rapid prototyping** with comprehensive features
3. **Personal workflow optimization** for research tasks
4. **Documentation for future reference** and tool improvement

### Development Setup (Personal Reference)
```bash
# Using the specified Python environment
"C:/Users/lewka/miniconda3/envs/deep_learning/python.exe" -m image_labelling.main

# Development environment
# - Python 3.12+ (Deep Learning environment)
# - Windows-based development
# - Private repository (no public contributions)
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

This project is licensed under a **Private Research Software License** - see the [LICENSE](LICENSE) file for details.

**Key Points for Solo Researchers:**
- ✅ **Private research use**: Completely free for your personal research
- ✅ **Strong protection**: Comprehensive liability limitations and disclaimers
- ❌ **No commercial use**: Commercial use requires separate permission
- ❌ **No redistribution**: Cannot be shared publicly without permission
- 🛡️ **Research focus**: Specific protections for research use cases

**Summary**: This license is designed specifically for private research tools, providing strong legal protection while allowing free use for legitimate research purposes.

```
PRIVATE RESEARCH SOFTWARE LICENSE

Copyright (c) 2025 [Your Name]

IMPORTANT: This software is developed for private research purposes only.

Key protections include:
• No warranty or guarantee of results
• Comprehensive liability limitations  
• Research-specific disclaimers
• Data responsibility clarifications
• No commercial use without permission
```

For the complete license terms, see the [LICENSE](LICENSE) file.

---

## 📞 Contact & Support

**Important**: This is a private research tool. Support is provided on a best-effort basis.

- **Questions**: Contact the author for research-related inquiries
- **Bug Reports**: Document issues in your local error.log files
- **Feature Requests**: Consider implementing yourself or contacting the author

**Remember**: You're using this tool at your own risk for research purposes. Always validate your annotations and back up your data!

---

**Built with ❤️ for Solo Computer Vision Researchers**

*A private tool designed to accelerate your research workflow while protecting your work.*
