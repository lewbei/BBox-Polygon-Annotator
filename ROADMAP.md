# Project Roadmap: BBox & Polygon Annotator

This document outlines the development roadmap and to-do list for the BBox & Polygon Annotator application, based on its current state (v9.x) and planned enhancements.

## I. Current Status (as of May 2025)

The application (v9.x) is a feature-rich tool with:
- Dual annotation (bounding box and polygon).
- Integrated Project Manager with a project list panel.
- YOLO auto-annotation.
- Multi-format export (YOLO YAML, COCO JSON, Pascal VOC XML, CSV, Generic JSON).
- Undo/Redo, status tracking, and various UX features.
- Recent critical bug fixes for UI interactions and attribute errors.

## II. Overall Vision

To evolve the BBox & Polygon Annotator into a best-in-class, highly performant, user-friendly, and extensible tool for computer vision dataset creation, potentially migrating to a more robust UI framework for advanced capabilities and cross-platform support.

## III. Development Roadmap

### A. Short-Term Goals (Next 1-3 Months - Q2/Q3 2025)

Focus: UI/UX Polish, Core Stability, Completing "In Progress" items.

1.  **Visual Enhancements & Core UX:**
    *   **Button Icons**: Integrate intuitive icons for all toolbar and major action buttons.
    *   **Responsive Canvas**: Implement dynamic sizing for the annotation canvas, allowing it to adapt to window size.
    *   **Zoom & Pan Controls**: Add zoom (mouse wheel, buttons) and pan functionality to the canvas for detailed annotation work.
    *   **Keyboard Shortcut Display**: Create a help dialog or easily accessible list of keyboard shortcuts.
2.  **Performance & Stability:**
    *   **Finalize Polygon Editing**: Complete "Advanced Polygon Editing" features (vertex deletion, polygon movement if not fully stable).
    *   **Error Handling**: Implement more robust error handling and user feedback mechanisms throughout the application.
    *   **Auto-Save System**: Introduce configurable auto-save intervals to prevent data loss.
3.  **Code Quality & Minor Refinements:**
    *   Address minor "In Progress" items from `README.md` if any are still pending beyond polygon editing.
    *   Begin tackling high-priority technical debt items (e.g., start breaking down monolithic `__init__` methods).

### B. Mid-Term Goals (Next 3-6 Months - Q3/Q4 2025)

Focus: Advanced Features, Deeper Performance Optimization, Significant Refactoring.

1.  **Layout & Navigation Overhaul:**
    *   **Resizable Panels/Splitters**: Implement draggable dividers between UI panels (Image List, Canvas/Info, Class Panel) for flexible layout.
    *   **Image List Enhancements**: Add image thumbnail previews and search/filter functionality for large datasets.
2.  **Performance Optimization:**
    *   **Image Caching System**: Implement caching for recently accessed images to speed up navigation.
    *   **Lazy Loading**: Optimize loading for very large datasets.
    *   **Background Processing**: Ensure all potentially long operations (model loading, extensive auto-annotation) are threaded with clear progress indicators.
3.  **Advanced Annotation Features:**
    *   **Batch Operations**: Allow users to perform actions (e.g., delete, change status) on multiple selected images or annotations.
4.  **Technical Debt Reduction:**
    *   Continue refactoring: Focus on separating UI and business logic (MVC-like pattern), improving state management.
    *   Improve test coverage (Unit, Integration tests).

### C. Long-Term Vision (6+ Months - 2026 and beyond)

Focus: Platform Enhancement, Broader AI Integration, Extensibility.

1.  **Platform & Framework:**
    *   **UI Framework Evaluation**: Seriously evaluate and potentially migrate to a more modern UI framework like PyQt6/PySide6 for richer features, better styling, and performance.
    *   **Cross-Platform Compatibility**: Thorough testing and refinement for macOS and Linux.
2.  **Advanced Features & AI:**
    *   **Annotation Analytics**: Provide users with insights into their annotation progress and dataset statistics.
    *   **Plugin Architecture**: Design a system to allow for extensions and custom tools.
    *   **Advanced UI Themes**: Implement dark mode and further UI customization options.
    *   **Deeper AI Integration**:
        *   AI-powered annotation suggestions.
        *   Automated annotation quality validation.
        *   Active learning loop integration.
        *   Direct model training pipeline connection.
3.  **Collaboration & Scalability:**
    *   **Cloud Integration**: Explore options for remote project storage and basic collaboration features.

## IV. Actionable To-Do List (Immediate - Short Term)

### Core UX & Visual Polish
-   [ ] **Task 1**: Research and select an icon library or create custom SVG icons for toolbar buttons (`Auto Annotate`, `Save Labels`, `Load Model`, `Export Annotations`, `Mode Toggle`, `Undo`, `Redo`).
-   [ ] **Task 2**: Implement icon display on the respective `ttk.Button` widgets.
-   [ ] **Task 3**: Modify `BoundingBoxEditor.setup_canvas` and related image display logic to allow the canvas to resize with the window.
    -   [ ] Sub-task: Ensure annotation coordinates correctly scale/translate with canvas resize.
-   [ ] **Task 4**: Implement zoom functionality on the canvas:
    -   [ ] Bind mouse wheel (Ctrl + scroll or Alt + scroll) to zoom.
    -   [ ] Add zoom in/out buttons to the UI (perhaps near the canvas or in a view menu).
    -   [ ] Ensure annotations scale correctly with zoom.
-   [ ] **Task 5**: Implement pan functionality on the canvas (e.g., middle-mouse drag or spacebar + drag).
-   [ ] **Task 6**: Design and implement a modal dialog or a dedicated section to display all available keyboard shortcuts.

### Stability & Polygon Enhancements
-   [ ] **Task 7**: Review and complete any pending "Advanced Polygon Editing" features:
    -   [ ] Vertex deletion (if not fully implemented/stable).
    -   [ ] Polygon movement/dragging (if not fully implemented/stable).
-   [ ] **Task 8**: Implement a configurable auto-save mechanism:
    -   [ ] Add a setting (e.g., in a config file or future settings dialog) for auto-save interval.
    -   [ ] Implement a timer in `BoundingBoxEditor` to trigger `save_labels` and `save_statuses`.
-   [ ] **Task 9**: Enhance global error handling:
    -   [ ] Wrap critical operations in try-except blocks.
    -   [ ] Use `messagebox.showerror` for user-facing errors instead of just print statements where appropriate.

### Project Manager Refinements (Post-Upgrade)
-   [ ] **Task 10**: (Low Priority) Consider adding "Last Modified Date" to the Project Manager's Treeview by reading file system metadata for the project JSON files.

### Code Quality
-   [ ] **Task 11**: Begin refactoring `BoundingBoxEditor.__init__`:
    -   [ ] Identify logical blocks of UI setup (e.g., `_setup_main_layout`, `_init_variables`, `_load_initial_data`).
    -   [ ] Move these blocks into separate private helper methods called from `__init__`.
-   [ ] **Task 12**: Review all `print()` statements used for debugging and replace them with a proper logging mechanism (e.g., Python's `logging` module) or remove if no longer needed.

This roadmap and to-do list should provide a good guide for future development.