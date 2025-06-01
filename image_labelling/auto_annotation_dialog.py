"""
Auto Annotation Configuration Dialog
This module provides the UI for configuring auto-annotation settings based on model capabilities.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Tuple, Optional, Callable

class AutoAnnotationDialog:
    """Dialog for configuring auto-annotation settings."""
    
    def __init__(self, parent, model_analysis: Dict, image_files: List[str], confidence_threshold: float = 0.5):
        self.parent = parent
        self.model_analysis = model_analysis
        self.image_files = image_files
        self.confidence_threshold = confidence_threshold
        self.result = None
        
        self.dialog = None
        self.annotation_type_var = None
        self.confidence_var = None
        self.file_selection_vars = {}
        self.select_all_var = None
        
    def show_dialog(self) -> Optional[Dict]:
        """
        Show the auto-annotation configuration dialog.
        
        Returns:
            Configuration dictionary if user clicks OK, None if cancelled
        """
        self._create_dialog()
        self._setup_ui()
        self._center_dialog()
        
        # Wait for dialog to close
        self.dialog.wait_window()
        
        return self.result
    
    def _create_dialog(self):
        """Create the main dialog window."""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Auto Annotation Configuration")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        self.dialog.geometry("600x700")
        self.dialog.resizable(True, True)
        
        # Make dialog modal
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)
    
    def _setup_ui(self):
        """Setup the dialog UI components."""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Model Information Section
        self._create_model_info_section(main_frame)
        
        # Annotation Type Selection Section
        self._create_annotation_type_section(main_frame)
        
        # Configuration Section
        self._create_configuration_section(main_frame)
        
        # File Selection Section
        self._create_file_selection_section(main_frame)
        
        # Buttons Section
        self._create_buttons_section(main_frame)
    
    def _create_model_info_section(self, parent):
        """Create model information display section."""
        info_frame = ttk.LabelFrame(parent, text="🤖 Detected Model Information", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model description
        model_desc = self._get_model_description()
        desc_label = ttk.Label(info_frame, text=model_desc, font=("TkDefaultFont", 9))
        desc_label.pack(anchor=tk.W)
        
        # Model capabilities
        if self.model_analysis.get("error"):
            error_label = ttk.Label(info_frame, text=f"⚠️ {self.model_analysis['error']}", 
                                  foreground="red", font=("TkDefaultFont", 9))
            error_label.pack(anchor=tk.W, pady=(5, 0))
    
    def _create_annotation_type_section(self, parent):
        """Create annotation type selection section."""
        type_frame = ttk.LabelFrame(parent, text="📝 Annotation Type Selection", padding="10")
        type_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.annotation_type_var = tk.StringVar()
        
        # Get available options
        options = self.model_analysis.get("available_options", [
            ("bounding_boxes", "Bounding Boxes (Default)", True)
        ])
        
        for i, (value, label, is_recommended) in enumerate(options):
            radio = ttk.Radiobutton(type_frame, text=label, variable=self.annotation_type_var, value=value)
            radio.pack(anchor=tk.W, pady=2)
            
            # Set default to recommended option
            if is_recommended:
                self.annotation_type_var.set(value)
        
        # Add explanation
        explanation = self._get_annotation_explanation()
        if explanation:
            exp_label = ttk.Label(type_frame, text=explanation, font=("TkDefaultFont", 8), 
                                foreground="gray")
            exp_label.pack(anchor=tk.W, pady=(5, 0))
    
    def _create_configuration_section(self, parent):
        """Create configuration settings section."""
        config_frame = ttk.LabelFrame(parent, text="⚙️ Configuration Settings", padding="10")
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Confidence threshold
        conf_frame = ttk.Frame(config_frame)
        conf_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(conf_frame, text="Confidence Threshold:").pack(side=tk.LEFT)
        
        self.confidence_var = tk.DoubleVar(value=self.confidence_threshold)
        conf_scale = ttk.Scale(conf_frame, from_=0.1, to=0.9, variable=self.confidence_var, 
                              orient=tk.HORIZONTAL)
        conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        
        conf_value_label = ttk.Label(conf_frame, text=f"{self.confidence_threshold:.1f}")
        conf_value_label.pack(side=tk.RIGHT)
        
        # Update confidence label when scale changes
        def update_conf_label(*args):
            conf_value_label.config(text=f"{self.confidence_var.get():.1f}")
        self.confidence_var.trace('w', update_conf_label)
    
    def _create_file_selection_section(self, parent):
        """Create file selection section."""
        file_frame = ttk.LabelFrame(parent, text=f"📁 File Selection ({len(self.image_files)} files)", padding="10")
        file_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Select all checkbox
        select_frame = ttk.Frame(file_frame)
        select_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.select_all_var = tk.BooleanVar(value=True)
        select_all_cb = ttk.Checkbutton(select_frame, text="Select All Files", 
                                       variable=self.select_all_var, command=self._on_select_all)
        select_all_cb.pack(side=tk.LEFT)
        
        count_label = ttk.Label(select_frame, text=f"{len(self.image_files)} files")
        count_label.pack(side=tk.RIGHT)
        
        # File list with scrollbar
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for file selection
        columns = ("select", "filename")
        self.file_tree = ttk.Treeview(list_frame, columns=columns, show="tree headings", height=8)
        
        # Configure columns
        self.file_tree.heading("#0", text="")
        self.file_tree.column("#0", width=30, minwidth=30)
        self.file_tree.heading("select", text="✓")
        self.file_tree.column("select", width=30, minwidth=30)
        self.file_tree.heading("filename", text="Filename")
        self.file_tree.column("filename", width=400)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=scrollbar.set)
        
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate file list
        self._populate_file_list()
        
        # Bind click events for toggling selection
        self.file_tree.bind("<Button-1>", self._on_file_click)
    
    def _create_buttons_section(self, parent):
        """Create dialog buttons section."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Cancel button
        cancel_btn = ttk.Button(button_frame, text="Cancel", command=self._on_cancel)
        cancel_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # OK button
        ok_btn = ttk.Button(button_frame, text="Start Auto Annotation", command=self._on_ok)
        ok_btn.pack(side=tk.RIGHT)
        
        # Make OK button default
        ok_btn.focus_set()
        self.dialog.bind('<Return>', lambda e: self._on_ok())
        self.dialog.bind('<Escape>', lambda e: self._on_cancel())
    
    def _populate_file_list(self):
        """Populate the file selection list."""
        for i, filename in enumerate(self.image_files):
            item_id = self.file_tree.insert("", tk.END, values=("✓", filename))
            self.file_selection_vars[item_id] = True
    
    def _on_select_all(self):
        """Handle select all checkbox."""
        select_all = self.select_all_var.get()
        
        for item_id in self.file_selection_vars:
            self.file_selection_vars[item_id] = select_all
            check_mark = "✓" if select_all else ""
            self.file_tree.set(item_id, "select", check_mark)
    
    def _on_file_click(self, event):
        """Handle file list click for toggling selection."""
        item = self.file_tree.identify_row(event.y)
        column = self.file_tree.identify_column(event.x)
        
        if item and column == "#1":  # Clicked on select column
            current_state = self.file_selection_vars.get(item, False)
            new_state = not current_state
            self.file_selection_vars[item] = new_state
            
            check_mark = "✓" if new_state else ""
            self.file_tree.set(item, "select", check_mark)
            
            # Update select all checkbox
            all_selected = all(self.file_selection_vars.values())
            any_selected = any(self.file_selection_vars.values())
            
            if all_selected:
                self.select_all_var.set(True)
            elif not any_selected:
                self.select_all_var.set(False)
    
    def _get_model_description(self) -> str:
        """Get formatted model description."""
        if self.model_analysis.get("error"):
            return f"❌ Model Error: {self.model_analysis['error']}"
        
        model_name = self.model_analysis.get("model_name", "Unknown Model")
        model_type = self.model_analysis.get("model_type", "Unknown")
        
        desc_lines = [f"Model: {model_name}", f"Type: {model_type}"]
        
        # Add capabilities
        if self.model_analysis.get("supports_segmentation") and self.model_analysis.get("supports_detection"):
            desc_lines.append("Capabilities: Detection + Segmentation")
        elif self.model_analysis.get("supports_detection"):
            desc_lines.append("Capabilities: Detection Only")
        elif self.model_analysis.get("supports_segmentation"):
            desc_lines.append("Capabilities: Segmentation Only")
        
        # Add class count
        classes = self.model_analysis.get("classes", [])
        if classes:
            desc_lines.append(f"Classes: {len(classes)} ({', '.join(classes[:3])}{'...' if len(classes) > 3 else ''})")
        
        return "\n".join(desc_lines)
    
    def _get_annotation_explanation(self) -> str:
        """Get explanation text for annotation types."""
        annotation_type = self.annotation_type_var.get() if self.annotation_type_var else ""
        
        if annotation_type == "segmentation":
            return "💡 Segmentation creates precise pixel-level masks for objects"
        elif annotation_type == "bounding_boxes":
            return "💡 Bounding boxes create rectangular regions around objects"
        elif annotation_type == "both":
            return "💡 Creates both bounding boxes and segmentation masks"
        else:
            return ""
    
    def _center_dialog(self):
        """Center the dialog on the parent window."""
        self.dialog.update_idletasks()
        
        # Get parent dimensions
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Get dialog dimensions
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        # Calculate center position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
    
    def _on_ok(self):
        """Handle OK button click."""
        selected_files = []
        for item_id, is_selected in self.file_selection_vars.items():
            if is_selected:
                values = self.file_tree.item(item_id)["values"]
                if len(values) >= 2:
                    selected_files.append(values[1])  # filename is in column 1
        
        if not selected_files:
            tk.messagebox.showwarning("No Files Selected", 
                                    "Please select at least one file for auto-annotation.", 
                                    parent=self.dialog)
            return
        
        self.result = {
            "annotation_type": self.annotation_type_var.get(),
            "confidence_threshold": self.confidence_var.get(),
            "selected_files": selected_files,
            "model_analysis": self.model_analysis
        }
        
        self.dialog.destroy()
    
    def _on_cancel(self):
        """Handle Cancel button click."""
        self.result = None
        self.dialog.destroy()
