# Image Labelling Tool
*Computer Vision Annotation Tool for Bounding Boxes and Polygons*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python-based image annotation tool built with Tkinter for creating bounding box and polygon annotations for computer vision datasets. Features project management, multiple export formats, YOLO model integration for auto-annotation, and a user-friendly interface designed for efficient dataset creation.

**Key Features**: Dual annotation modes (bounding boxes & polygons), YOLO auto-annotation, project-based workflow, multiple export formats (YOLO, COCO, Pascal VOC, CSV), and comprehensive annotation editing capabilities.


## ✨ Features

### Core Annotation Capabilities
- **Bounding Box Annotation**: Draw, resize, and move rectangular annotations
- **Polygon Annotation**: Create and edit polygon annotations with vertex manipulation
- **Multi-Class Support**: Define and manage multiple object classes with color coding
- **Annotation Editing**: Copy, paste, delete, move, and resize annotations with undo/redo support

### AI-Powered Workflow  
- **YOLO Auto-Annotation**: Automatic annotation using pre-trained or custom YOLO models
- **Confidence Filtering**: Configurable confidence thresholds for prediction filtering
- **Batch Processing**: Auto-annotate entire image sets with progress tracking

### Project Management
- **Project-Based Organization**: Manage multiple annotation projects with metadata
- **Image Status Tracking**: Track annotation progress (Not Viewed, Viewed, Labeled, Review Needed)
- **Progress Monitoring**: Real-time statistics and completion tracking

### User Interface
- **Modern UI**: Clean interface with Unicode icons and responsive design
- **Canvas Controls**: Zoom (mouse wheel) and pan (middle-mouse drag) functionality
- **Keyboard Shortcuts**: Efficient navigation and annotation with hotkeys
- **Error Handling**: Comprehensive error logging and user feedback

### Export & Integration
- **Multiple Export Formats**: YOLO, COCO JSON, Pascal VOC XML, CSV formats
- **Data Splitting**: Configurable train/validation/test splits
- **Flexible Export Options**: Include/exclude images and unannotated files

## 🔧 Installation

### Prerequisites
- **Python 3.8 or higher**
- **Operating System**: Windows, Linux, or macOS

### Required Dependencies
```bash
# Core dependencies
pip install opencv-python Pillow PyYAML

# Optional: For YOLO auto-annotation
pip install ultralytics
```

### Installation Steps

1. **Clone or Download the Project**
   ```bash
   # Clone the repository or download the source code
   cd path/to/your/image_labelling_project
   ```

2. **Install Dependencies**
   ```bash
   # Install required packages
   pip install opencv-python Pillow PyYAML
   
   # Optional: For YOLO auto-annotation features
   pip install ultralytics
   ```

3. **Run the Application**
   ```bash
   # Using the specific Python environment (as specified in instructions)
   "C:/Users/lewka/miniconda3/envs/deep_learning/python.exe" -m image_labelling.main
   
   # Or using standard Python
   python -m image_labelling.main
   ```

### Note on Python Environment
This tool is designed to work with your existing deep learning environment. The application runs entirely locally with no cloud dependencies.

## 🚀 Quick Start

### 1. Launch the Application
```bash
"C:/Users/lewka/miniconda3/envs/deep_learning/python.exe" -m image_labelling.main
```

### 2. Create Your First Project
- Click **"New Project"** in the Project Manager
- Enter a project name and select your dataset folder containing images
- The tool will create the necessary folder structure

### 3. Start Annotating
- **Navigate Images**: Use the image list or arrow keys to browse your dataset
- **Choose Annotation Mode**: 
  - Click **⬜** for bounding box mode
  - Click **🔷** for polygon mode
- **Create Annotations**: 
  - **Bounding boxes**: Click and drag to create rectangles
  - **Polygons**: Click to place vertices, right-click or Enter to complete
- **Assign Classes**: Select classes from the list or use number keys (1-9)
- **Save Progress**: Use Ctrl+S or enable auto-save

### 4. Export Your Dataset
- Click the **📤** export button
- Choose from YOLO, COCO, Pascal VOC, or CSV formats
- Configure train/validation splits as needed

## 🗂️ Project Management

### Project Structure
```
YourProject/
├── dataset.yaml          # Project configuration and class definitions
├── images/              # Source images (JPG, PNG, BMP, TIFF)
├── labels/              # YOLO format annotations (.txt files)
└── status.json          # Image annotation status tracking
```

### Project Manager Features
- **Create Projects**: Set up new annotation projects with custom names and dataset paths
- **Open Projects**: Resume work on existing projects with preserved state
- **Delete Projects**: Remove project metadata (original images remain intact)
- **Status Overview**: View project statistics and last modified timestamps

### Image Status System
- **🔴 Not Viewed**: Images not yet opened
- **🟡 Viewed**: Images opened but not annotated  
- **🟢 Labeled**: Images with annotations
- **🟠 Review Needed**: Auto-annotated images requiring manual verification

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

| Format | Use Case | Output | Features |
|--------|----------|--------|----------|
| **YOLO** | Direct YOLO training | `dataset.yaml` + organized folders | Data splitting, class mapping |
| **COCO JSON** | Industry standard | `annotations.json` | Full COCO compliance, metadata |
| **Pascal VOC** | Traditional CV | Individual `.xml` files | Per-image annotations |
| **CSV** | Analysis & review | `annotations.csv` | Spreadsheet-friendly format |

### Export Options
- **Data Splitting**: Configure train/validation/test ratios
- **Image Inclusion**: Option to copy images to export directory
- **Unannotated Images**: Include or exclude images without annotations
- **Custom Locations**: Export to any directory location

## 🔥 Advanced Features

### Auto-Annotation with YOLO
- **Model Loading**: Load pre-trained or custom YOLO models (.pt files)
- **Batch Processing**: Annotate entire datasets automatically
- **Confidence Thresholds**: Filter predictions by confidence scores
- **Progress Tracking**: Real-time progress with cancel capability
- **Review System**: Flag predictions for manual verification

### Canvas Controls
- **Zoom**: Mouse wheel or toolbar buttons
- **Pan**: Middle-mouse drag for navigation
- **Responsive Scaling**: Annotations scale correctly with zoom
- **Full Interface**: Resizable panels and responsive layout

### Annotation Management
- **Undo/Redo**: Full action history with Ctrl+Z/Ctrl+Y
- **Copy/Paste**: Duplicate annotations across images  
- **Class Management**: Add, edit, and color-code object classes
- **Auto-Save**: Configurable automatic saving intervals

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
- **Theme Support**: Automatic theme detection (clam, vista, default)
- **Window Sizing**: Responsive layout with configurable panels
- **Icon System**: Unicode-based icons for cross-platform compatibility

## 🛠️ Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
# Ensure you're in the correct directory
cd path/to/image_labelling

# Run with proper module syntax and specified Python environment
"C:/Users/lewka/miniconda3/envs/deep_learning/python.exe" -m image_labelling.main
```

#### YOLO auto-annotation not working
```bash
# Install ultralytics if not already installed
pip install ultralytics

# Check the console/logs for model loading errors
```

#### Images not loading
- **Supported formats**: JPG, JPEG, PNG, BMP, TIFF
- **Path issues**: Ensure no special characters in folder paths
- **Permissions**: Verify read access to image directories

#### Performance issues with large datasets
- **Enable auto-save**: Set reasonable intervals (300-600 seconds)
- **Close other applications**: Free up system memory
- **Check memory usage**: Monitor system resources

### Error Reporting
If you encounter bugs:
1. Check the `error.log` file in your project directory
2. Note the Python version and operating system
3. Provide steps to reproduce the issue
4. Include relevant log messages

## 🏗️ Project Architecture

The application is structured as a modular Python package:

```
image_labelling/
├── __init__.py           # Package initialization
├── main.py              # Application entry point with error handling
├── project_manager.py   # Project creation and management UI
├── editor.py           # Main annotation editor with canvas controls
├── exporter.py         # Export format converters (COCO, VOC, CSV)
├── helpers.py          # Utility functions for file operations
├── constants.py        # Application constants and logging setup
├── settings.py         # User preferences and window state persistence
└── startup_optimizer.py # Performance optimizations and lazy loading
```

### Key Components
- **Project Manager**: Handles project creation, listing, and selection
- **Annotation Editor**: Core annotation interface with bounding box and polygon tools  
- **Export System**: Converts annotations to various standard formats
- **Auto-Annotation**: YOLO model integration for automated labeling
- **Settings Management**: Persistent user preferences and window geometry

### Technical Stack
- **GUI Framework**: Tkinter with ttk styling
- **Image Processing**: OpenCV and Pillow (PIL)
- **ML Integration**: Ultralytics YOLO for auto-annotation
- **Data Formats**: YAML, JSON, XML, CSV export capabilities
- **Error Handling**: Comprehensive logging and user feedback systems

## 🗺️ Future Enhancements

### Performance Optimizations
- Image caching system for large datasets
- Lazy loading for improved startup performance
- Background processing for YOLO operations
- Memory usage optimizations

### User Interface Improvements
- Batch operations for multiple images
- Advanced polygon editing tools
- Image thumbnails in project list
- Customizable keyboard shortcuts

### Advanced Features
- Additional export formats and options
- Integration with more ML model types  
- Annotation quality validation tools
- Advanced project statistics and analytics

### Platform Enhancements
- Cross-platform optimization
- Alternative GUI framework evaluation
- Plugin architecture for extensibility

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### What this means:
- ✅ **Free to use** for any purpose (commercial, research, education, private)
- ✅ **Free to modify** and create derivative works
- ✅ **Free to distribute** original or modified versions  
- ✅ **Free to sell** or include in commercial products
- ✅ **No warranty** - software provided "as is"
- ⚠️ **Attribution required** - must include original copyright notice

## 🤝 Contributing

Contributions are welcome! This project follows standard open-source contribution practices.

### How to Contribute:
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/new-feature`)
3. **Commit** your changes (`git commit -m 'Add new feature'`)
4. **Push** to the branch (`git push origin feature/new-feature`)
5. **Open** a Pull Request

### Areas for Contribution:
- 🐛 Bug fixes and stability improvements
- ✨ New annotation features and tools
- 🎨 UI/UX improvements
- 📚 Documentation enhancements
- 🧪 Testing and platform compatibility
- 🚀 Performance optimizations

### Reporting Issues:
- Use GitHub Issues for bug reports
- Include detailed steps to reproduce
- Attach relevant error logs  
- Specify your OS and Python version

---

**Built for Computer Vision Dataset Creation**

*A practical tool for efficient image annotation workflows.*
