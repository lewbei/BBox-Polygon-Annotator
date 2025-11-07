import os
import shutil
import json
import logging
import yaml
import threading
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
import time
import random
from collections import OrderedDict

import tkinter as tk
from tkinter import ttk # Import ttk
from tkinter import filedialog, colorchooser, simpledialog, messagebox
# Defer heavy imports for faster startup - use lazy_importer instead
# from PIL import Image, ImageTk # Import Image and ImageTk from Pillow
# import numpy as np
# import json # Already imported

from image_labelling.constants import ICON_UNICODE, PROJECTS_DIR
from image_labelling.helpers import center_window, write_annotations_to_file, read_annotations_from_file, copy_files_recursive
from image_labelling.startup_optimizer import lazy_importer
from .exporter import convert_to_coco_format, convert_to_pascal_voc_format, convert_to_csv_format

# Get PIL components via lazy loader
def _get_pil():
    """Get PIL components lazily."""
    pil_components = lazy_importer.get_pil()
    return pil_components['Image'], pil_components['ImageTk']


class BoundingBoxEditor(tk.Frame):
    """
    This class provides a Tkinter-based interface for visualizing and editing bounding boxes
    on images. It also integrates a YOLO-based auto-annotation feature and a project-based 
    organizational structure for labeling tasks.
    """

    def __init__(self, master, project):
        super().__init__(master)
        self.root = master
        self.project = project
        self.root.title(f"Bounding Box Editor - Project: {project['project_name']}")

        # Model and concurrency handles
        self.model = None
        self.cancel_event = None

        # Folder paths from the project
        self.folder_path = project["dataset_path"]
        self.label_folder = os.path.join(self.folder_path, "labels")
        os.makedirs(self.label_folder, exist_ok=True)

        # YAML file path (dataset config)
        self.yaml_path = os.path.join(self.folder_path, "dataset.yaml")
        self.create_default_yaml_if_missing()

        # Load data from YAML
        with open(self.yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Class names from YAML
        raw_names = data.get("names", ["person"])
        if isinstance(raw_names, dict):
            # Convert dict to list sorted by integer keys
            self.class_names = [raw_names[k] for k in sorted(raw_names.keys(), key=lambda x: int(x))]
        else:
            self.class_names = raw_names

        # Paths used in the YAML (optional usage)
        self.paths = data.get("paths", {"dataset": self.folder_path, "train": "", "val": ""})
        self.validation = bool(self.paths.get("val"))
        self.auto_save_interval = data.get("auto_save_interval", 0)

        # Color mapping for classes
        self.update_class_colors()
        self.image_status = {}
        self.image_cache = OrderedDict()
        self.max_cache_size = data.get("image_cache_size", 20)

        # Performance: cache file existence checks
        self._file_exists_cache = {}

        # -----------------------------
        # Main UI Layout
        # -----------------------------
        # Top Bar: Buttons (Auto Annotate, Save, Load Model, Export)
        self.setup_top_bar() # This call requires methods below to be defined if used as commands

        # Main content frame (everything except top bar and bottom status bar)
        self.main_content_frame = tk.Frame(master)
        self.main_content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left Pane: Treeview for image list (inside main_content_frame)
        self.image_list_frame = tk.Frame(self.main_content_frame, width=200)
        self.image_list_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.setup_image_list_panel_widgets()

        # Middle Pane: Canvas (image) + Info Panel (inside main_content_frame)
        self.content_frame = tk.Frame(self.main_content_frame)
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.setup_canvas()
        self.setup_info_panel()
        
        # Right Pane: Class List + Actions (inside main_content_frame)
        self.class_frame = tk.Frame(self.content_frame, width=200)
        self.class_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 10), pady=10)
        self.setup_class_panel()

        # Status Bar: labeling progress counters (at the very bottom)
        self.setup_status_bar()
        
        # Initialize image-related variables
        self.image = None 
        self.image_path = None
        self.original_image = None 
        self.bboxes = [] 
        self.polygons = [] 
        
        self.current_bbox = None 
        self.current_bbox_orig_start = None 
        self.rect_start_canvas = None 
        self.rect = None 

        self.current_polygon_points = [] 
        self.polygon_drawing_active = False 
        
        self.dragging_point = False 
        self.drag_polygon_index = -1 
        self.drag_point_index = -1 
        self.hover_polygon_index = -1 
        self.hover_point_index = -1 
        self.dragging_whole_polygon = False 
        self.drag_whole_polygon_index = -1  
        self.polygon_move_start = (0, 0)
        
        self.polygon_just_completed = False    
        
        self.image_files = []
        self.current_image_index = -1
        self.selected_class_index = None
        self.annotation_mode = 'box' 
        self.zoom_level = 1.0 
        self.image_view_offset_x = 0 
        self.image_view_offset_y = 0
        self.image_offset_x = 0 
        self.image_offset_y = 0
        
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.pan_start_view_offset_x = 0
        self.pan_start_view_offset_y = 0

        self.history = []
        self.history_index = -1
        self.max_history_size = 20

        # Performance optimization: throttle canvas redraws
        self._pending_redraw = None
        self._redraw_throttle_ms = 16  # ~60 FPS max

        self.load_dataset_async()
        self.setup_bindings()
        self.save_history()
        if self.auto_save_interval and self.auto_save_interval > 0:
            self.start_auto_save()

    def _attempt_load_initial_image(self):
        """Attempts to load the last opened image or the first image in the dataset."""
        if not self.image_files: 
            return

        loaded_an_image = False
        if 'last_opened_image_relative' in self.project:
            last_image_relative_path = self.project['last_opened_image_relative']
            if last_image_relative_path:
                last_image_full_path = os.path.join(self.folder_path, last_image_relative_path)
                parent = os.path.dirname(last_image_relative_path)
                if parent:
                    parts = parent.split(os.sep)
                    acc = ''
                    for p in parts:
                        acc = p if not acc else os.path.join(acc, p)
                        folder_id = f'folder_{acc}'
                        if self.image_tree.exists(folder_id):
                            self.image_tree.item(folder_id, open=True)
                            self.on_folder_expand(None, folder_id)
                if os.path.exists(last_image_full_path) and last_image_relative_path in self.image_files:
                    try:
                        if self.image_tree.exists(last_image_relative_path):
                            self.image_tree.selection_set(last_image_relative_path)
                            self.image_tree.focus(last_image_relative_path)
                            self.image_tree.see(last_image_relative_path)
                            self.load_image(last_image_full_path)
                            loaded_an_image = True
                        else:
                            pass  # Last opened image not found in tree
                    except tk.TclError:
                        pass  # TclError while trying to select last opened image
                        
        if not loaded_an_image and self.image_files:
            first_image_relative_path = self.image_files[0]
            first_image_full_path = os.path.join(self.folder_path, first_image_relative_path)
            try:
                if self.image_tree.exists(first_image_relative_path):
                    self.image_tree.selection_set(first_image_relative_path)
                    self.image_tree.focus(first_image_relative_path)
                    self.image_tree.see(first_image_relative_path)
                    self.load_image(first_image_full_path)
                else:
                    pass  # First image not found in tree during fallback
            except tk.TclError as e:
                pass  # TclError while trying to select first image
    
    def auto_annotate_dataset_threaded(self):
        """Initiates the auto-annotation process with model detection and configuration dialog."""
        if self.model is None: 
            messagebox.showerror("Model Not Loaded", "Please load a YOLO model first.")
            return
        
        # Import the new modules
        try:
            from .model_analyzer import ModelAnalyzer
            from .auto_annotation_dialog import AutoAnnotationDialog
        except ImportError as e:
            messagebox.showerror("Import Error", f"Failed to import auto-annotation modules: {e}")
            return
        
        try:            # Analyze the loaded model
            analyzer = ModelAnalyzer()
            model_analysis = analyzer.analyze_model(
                getattr(self.model, 'model_path', 'Unknown'), 
                self.model
            )
            
            # Debug log setup
            debug_log_path = os.path.join(os.path.dirname(__file__), 'debug_auto_annotation.log')
            
            def debug_log(message):
                with open(debug_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{message}\n")
                print(message)  # Also print to console
            
            # Clear previous log and start fresh
            with open(debug_log_path, 'w', encoding='utf-8') as f:
                f.write("=== AUTO-ANNOTATION DEBUG LOG ===\n")
            
            debug_log(f"DEBUG MODEL ANALYSIS: {model_analysis}")
            debug_log(f"DEBUG MODEL ANALYSIS available_options: {model_analysis.get('available_options', 'NOT_FOUND')}")
            
            # Show configuration dialog
            dialog = AutoAnnotationDialog(
                parent=self.root,
                model_analysis=model_analysis,
                image_files=list(self.image_files),
                confidence_threshold=self.confidence_threshold.get()
            )
            
            config = dialog.show_dialog()
            
            # If user cancelled, return
            if config is None:
                return
            
            # Update confidence threshold
            self.confidence_threshold.set(config['confidence_threshold'])
            
            # Start auto-annotation with configuration
            self._start_auto_annotation_with_config(config)
            
        except Exception as e:
            messagebox.showerror("Auto Annotation Error", f"Failed to start auto annotation: {e}")
    
    def _start_auto_annotation_with_config(self, config):
        """Start the auto-annotation process with the given configuration."""
        self.auto_annotate_button.config(state=tk.DISABLED)
        
        # Create progress window
        self.progress_win = tk.Toplevel(self.root)
        self.progress_win.title("Auto Annotation Progress")
        self.progress_win.transient(self.root)
        self.progress_win.grab_set()
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_win, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(padx=20, pady=10, fill=tk.X, expand=True)
        
        self.progress_label = tk.Label(self.progress_win, text="0/0 images processed")
        self.progress_label.pack(pady=5)
        
        self.cancel_button = tk.Button(self.progress_win, text="Cancel", command=self.cancel_annotation)
        self.cancel_button.pack(pady=5)
        
        # Add annotation type info
        annotation_info = tk.Label(self.progress_win, 
                                 text=f"Mode: {config['annotation_type'].replace('_', ' ').title()}")
        annotation_info.pack(pady=2)
        
        self.progress_win.update_idletasks()
        
        # Center progress window
        main_width = self.root.winfo_width()
        main_height = self.root.winfo_height()
        main_x = self.root.winfo_x()
        main_y = self.root.winfo_y()
        progress_width = 350
        progress_height = 150
        x = main_x + (main_width - progress_width) // 2
        y = main_y + (main_height - progress_height) // 2
        self.progress_win.geometry(f"{progress_width}x{progress_height}+{x}+{y}")
        
        # Start annotation in thread
        self.cancel_event = threading.Event()
        self.annotation_config = config
        threading.Thread(target=self.auto_annotate_dataset, daemon=True).start()

    def show_shortcuts(self):
        shortcut_list = [
            ("Ctrl+S", "Save labels"), ("Ctrl+Z", "Undo"), ("Ctrl+Y", "Redo"),
            ("Esc", "Cancel polygon or clear selection"), ("Up/Down Arrow", "Navigate images"),
            ("Mouse Wheel", "Navigate images"), ("Ctrl + Mouse Wheel", "Zoom in/out"),
            ("Middle Mouse Drag", "Pan"), ("Digits 1-9", "Select class"),
        ]
        dlg = tk.Toplevel(self.root)
        dlg.title("Keyboard Shortcuts")
        dlg.transient(self.root)
        dlg.grab_set()
        frame = ttk.Frame(dlg, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        cols = ("Shortcut", "Description")
        tree = ttk.Treeview(frame, columns=cols, show="headings", selectmode="none")
        for col in cols: tree.heading(col, text=col); tree.column(col, anchor="w")
        for key, desc in shortcut_list: tree.insert("", "end", values=(key, desc))
        tree.pack(fill=tk.BOTH, expand=True)
        ttk.Button(frame, text="Close", command=dlg.destroy).pack(pady=(10, 0))
        dlg.update_idletasks()
        center_window(dlg, 400, 300)
        self.root.wait_window(dlg)

    def zoom_in(self):
        class _E: pass
        e = _E()
        e.delta = 120
        self.on_zoom(e)

    def zoom_out(self):
        class _E: pass
        e = _E()
        e.delta = -120
        self.on_zoom(e)

    def toggle_annotation_mode(self):
        if self.annotation_mode == 'box':
            self.annotation_mode = 'polygon'
            self.mode_toggle_button.config(text="Mode: Polygon")
            messagebox.showinfo("Mode Switched", "Switched to Polygon Annotation Mode.\nClick to add points, double-click to complete polygon.\nPress ESC or right-click to cancel current polygon.")
            self.canvas.delete(self.rect)
            self.current_bbox = None
        else:
            self.annotation_mode = 'box'
            self.mode_toggle_button.config(text="Mode: Box")
            messagebox.showinfo("Mode Switched", "Switched to Box Annotation Mode.")
            self.clear_current_polygon_drawing()
            self.current_polygon_points = []
            self.polygon_drawing_active = False

    def undo(self):
        if self.history_index > 0: self.history_index -= 1; self.restore_from_history()
    
    def redo(self):
        if self.history_index < len(self.history) - 1: self.history_index += 1; self.restore_from_history()

    def save_labels(self, *args):
        if not self.image_path or self.current_image_index == -1: return
        relative_image_path = os.path.relpath(self.image_path, self.folder_path)
        label_relative_path = os.path.splitext(relative_image_path)[0] + '.txt'
        label_path = os.path.join(self.label_folder, label_relative_path)
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        if self.original_image:
            original_shape = (self.original_image.height, self.original_image.width)            
            write_annotations_to_file(label_path, self.bboxes, self.polygons, original_shape)
        else:
            fallback_shape = (480, 640)
            if hasattr(self, 'image') and self.image is not None and hasattr(self.image, 'shape'):
                 Image, _ = _get_pil()
                 pil_image_from_numpy = Image.fromarray(self.image)
                 fallback_shape = (pil_image_from_numpy.height, pil_image_from_numpy.width)
            write_annotations_to_file(label_path, self.bboxes, self.polygons, fallback_shape)
        new_status = "edited" if (self.bboxes or self.polygons) else "viewed"
        self.image_status[relative_image_path] = new_status
        self.image_tree.item(relative_image_path, tags=(new_status,))
        self.save_statuses(); self.update_status_labels()

    def load_model(self):
        model_path = filedialog.askopenfilename(title="Select YOLO Model", filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")])
        if model_path:
            progress_win = tk.Toplevel(self.root)
            progress_win.title("Loading Model")
            progress_win.transient(self.root)
            progress_win.grab_set()
            progress_win.geometry("400x120")
            
            main_width = self.root.winfo_width()
            main_height = self.root.winfo_height()
            main_x = self.root.winfo_x()
            main_y = self.root.winfo_y()
            progress_width = 400
            progress_height = 120
            x = main_x + (main_width - progress_width) // 2
            y = main_y + (main_height - progress_height) // 2
            progress_win.geometry(f"{progress_width}x{progress_height}+{x}+{y}")
            
            progress_bar = ttk.Progressbar(progress_win, mode='indeterminate')
            progress_bar.pack(padx=20, pady=10, fill=tk.X)
            progress_bar.start(10)
            
            progress_label = tk.Label(progress_win, text=f"Loading YOLO model...\n{os.path.basename(model_path)}")
            progress_label.pack(pady=5)
            
            progress_win.update_idletasks()
            
            def load_model_thread():
                try:                    
                    YOLO = lazy_importer.get_yolo()
                    model = YOLO(model_path)
                    
                    def on_success():
                        try:
                            self.model = model
                            # Store model path for analysis
                            self.model.model_path = model_path
                            progress_win.destroy()
                            messagebox.showinfo("Success", f"Model loaded successfully from:\n{model_path}")
                        except:
                            pass
                    
                    self.root.after(0, on_success)
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    def on_error():
                        try:
                            progress_win.destroy()
                            messagebox.showerror("Error", f"Failed to load model:\n{error_msg}")
                        except:
                            pass 
                    
                    self.root.after(0, on_error)
            
            import threading
            loading_thread = threading.Thread(target=load_model_thread, daemon=True)
            loading_thread.start()

    def train_yolo_model(self):
        """Open a dialog for standard training configuration and execution"""
        annotated_count = 0
        for image_path_rel in self.image_files: # Iterate over relative paths
            label_path = os.path.join(self.label_folder, os.path.splitext(image_path_rel)[0] + '.txt')
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                annotated_count += 1
        
        if annotated_count < 10:
            messagebox.showwarning("Insufficient Data", 
                f"Only {annotated_count} annotated images found. Recommend at least 10+ images for training.")
            return

        self.open_training_dialog()

    def open_active_learning_dialog(self):
        """Open a dialog to configure and start an active learning annotation loop"""
        al_win = tk.Toplevel(self.root)
        al_win.title("Active Learning Loop")
        al_win.transient(self.root)
        al_win.grab_set()
        al_win.geometry("650x600")

        # Overview of active learning workflow
        msg = tk.Message(
            al_win,
            text=(
                "Active learning will iteratively train a model on initially labeled samples "
                "and select the most informative new images for annotation based on the chosen strategy."
            ),
            width=630,
            foreground="gray"
        )
        msg.pack(padx=10, pady=(10, 0))

        config_frame = tk.LabelFrame(al_win, text="Active Learning Configuration")
        config_frame.pack(fill=tk.X, padx=10, pady=5)

        # Task selection: choose detection (bounding boxes) or segmentation (masks)
        task_var = tk.StringVar(value="detect")
        tk.Label(config_frame, text="Task:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        tk.Radiobutton(
            config_frame, text="Detection", variable=task_var, value="detect"
        ).grid(row=0, column=1, sticky="w", padx=5, pady=2)
        tk.Radiobutton(
            config_frame, text="Segmentation", variable=task_var, value="segment"
        ).grid(row=0, column=2, sticky="w", padx=5, pady=2)
        tk.Label(
            config_frame,
            text="Choose Detection for bounding boxes or Segmentation for mask annotation",
            foreground="gray",
            wraplength=630
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=5)

        tk.Label(config_frame, text="Initial Seed Size:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        seed_var = tk.StringVar(value="20")
        tk.Entry(config_frame, textvariable=seed_var, width=10).grid(row=2, column=1, sticky="w", padx=5, pady=2)
        tk.Label(
            config_frame,
            text="# of initially labeled images to train the first model",
            foreground="gray"
        ).grid(row=3, column=0, columnspan=3, sticky="w", padx=5)

        tk.Label(config_frame, text="Iteration Budget:").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        budget_var = tk.StringVar(value="10")
        tk.Entry(config_frame, textvariable=budget_var, width=10).grid(row=4, column=1, sticky="w", padx=5, pady=2)
        tk.Label(
            config_frame,
            text="# of new images to select for annotation each round",
            foreground="gray"
        ).grid(row=5, column=0, columnspan=3, sticky="w", padx=5)

        tk.Label(config_frame, text="Query Strategy:").grid(row=6, column=0, sticky="w", padx=5, pady=2)
        strategy_var = tk.StringVar(value="Uncertainty")
        strategy_combo = ttk.Combobox(
            config_frame, textvariable=strategy_var,
            values=["Uncertainty", "Margin", "Random"], state="readonly", width=12
        )
        strategy_combo.grid(row=6, column=1, sticky="w", padx=5, pady=2)
        tk.Label(
            config_frame,
            text="Selection method: Uncertainty=lowest confidence, Margin=smallest class margin, Random=baseline",
            foreground="gray"
        ).grid(row=7, column=0, columnspan=3, sticky="w", padx=5)

        tk.Label(config_frame, text="Model Checkpoint:").grid(row=8, column=0, sticky="w", padx=5, pady=2)
        ckpt_var = tk.StringVar(value="")
        ckpt_entry = tk.Entry(config_frame, textvariable=ckpt_var, width=40, state="readonly")
        ckpt_entry.grid(row=8, column=1, sticky="ew", padx=5, pady=2)
        tk.Button(
            config_frame, text="Browse",
            command=lambda: ckpt_var.set(
                filedialog.askopenfilename(
                    title="Select YOLO Model",
                    filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
                )
            )
        ).grid(row=8, column=2, sticky="w", padx=5, pady=2)
        tk.Label(
            config_frame,
            text="YOLO model to warm-start learning; leave blank for random initialization",
            foreground="gray"
        ).grid(row=9, column=0, columnspan=3, sticky="w", padx=5)

        # Training hyperparameters for each iteration
        tk.Label(config_frame, text="Epochs:").grid(row=10, column=0, sticky="w", padx=5, pady=2)
        epoch_var = tk.StringVar(value="50")
        tk.Entry(config_frame, textvariable=epoch_var, width=10).grid(row=10, column=1, sticky="w", padx=5, pady=2)
        tk.Label(
            config_frame,
            text="# Number of training epochs per iteration",
            foreground="gray"
        ).grid(row=11, column=0, columnspan=3, sticky="w", padx=5)

        tk.Label(config_frame, text="Image Size:").grid(row=12, column=0, sticky="w", padx=5, pady=2)
        imgsz_var = tk.StringVar(value="640")
        tk.Entry(config_frame, textvariable=imgsz_var, width=10).grid(row=12, column=1, sticky="w", padx=5, pady=2)
        tk.Label(
            config_frame,
            text="# Training input image size (square)",
            foreground="gray"
        ).grid(row=13, column=0, columnspan=3, sticky="w", padx=5)

        tk.Label(config_frame, text="Batch Size:").grid(row=14, column=0, sticky="w", padx=5, pady=2)
        batch_var = tk.StringVar(value="16")
        tk.Entry(config_frame, textvariable=batch_var, width=10).grid(row=14, column=1, sticky="w", padx=5, pady=2)
        tk.Label(
            config_frame,
            text="# Number of images per training batch",
            foreground="gray"
        ).grid(row=15, column=0, columnspan=3, sticky="w", padx=5)

        tk.Label(config_frame, text="Learning Rate:").grid(row=16, column=0, sticky="w", padx=5, pady=2)
        lr_var = tk.StringVar(value="0.01")
        tk.Entry(config_frame, textvariable=lr_var, width=10).grid(row=16, column=1, sticky="w", padx=5, pady=2)
        tk.Label(
            config_frame,
            text="# Initial learning rate for training",
            foreground="gray"
        ).grid(row=17, column=0, columnspan=3, sticky="w", padx=5)

        progress_frame = tk.LabelFrame(al_win, text="Active Learning Progress")
        progress_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.active_learning_progress = tk.Text(progress_frame, height=6, state=tk.DISABLED)
        scroll = tk.Scrollbar(progress_frame, orient=tk.VERTICAL, command=self.active_learning_progress.yview)
        self.active_learning_progress.configure(yscrollcommand=scroll.set)
        self.active_learning_progress.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.active_learning_stop_flag = threading.Event()
        btn_frame = tk.Frame(al_win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        start_btn = tk.Button(
            btn_frame, text="Start Active Learning",
            command=lambda: self._start_active_learning(
                task_var, seed_var, budget_var, strategy_var, ckpt_var, epoch_var,
                imgsz_var, batch_var, lr_var, al_win, start_btn
            )
        )
        start_btn.pack(side=tk.LEFT, padx=5)
        stop_btn = tk.Button(btn_frame, text="🛑 Stop", command=lambda: self.active_learning_stop_flag.set())
        stop_btn.pack(side=tk.LEFT, padx=5)
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=al_win.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=5)

        center_window(al_win, 650, 600)

    def _start_active_learning(self, task_var, seed_var, budget_var, strategy_var, ckpt_var, epoch_var, imgsz_var, batch_var, lr_var, window, start_btn):
        """Handler for starting one iteration of the active learning loop"""
        start_btn.config(state=tk.DISABLED)
        # Clear stop flag and read configuration
        self.active_learning_stop_flag.clear()
        task = task_var.get()
        seed_size = int(seed_var.get())
        budget = int(budget_var.get())
        strategy = strategy_var.get()
        ckpt = ckpt_var.get()
        epochs = int(epoch_var.get())
        imgsz = int(imgsz_var.get())
        batch_size = int(batch_var.get())
        lr = float(lr_var.get())
        self._log_active_learning(
            f"Starting Active Learning: task={task}, seed={seed_size}, budget={budget}, "
            f"strategy={strategy}, epochs={epochs}, imgsz={imgsz}, batch={batch_size}, lr={lr}, "
            f"checkpoint={ckpt or 'random init'}"
        )

        def _worker():
            iteration = 1
            # Initial seed selection: ask user to label seed images if no labels yet
            labeled = [img for img in self.image_files
                       if os.path.exists(os.path.join(self.label_folder, os.path.splitext(img)[0] + '.txt'))]
            if not labeled and seed_size > 0:
                unlabeled = list(self.image_files)
                seed_imgs = random.sample(unlabeled, min(seed_size, len(unlabeled)))
                for rel_img in seed_imgs:
                    if self.image_tree.exists(rel_img):
                        self.image_tree.item(rel_img, tags=("review_needed",))
                        self.image_tree.see(rel_img)
                        self.image_status[rel_img] = "review_needed"
                self.save_statuses()
                self._log_active_learning(
                    f"Initial seed selected ({len(seed_imgs)} images). Please annotate these before training."
                )
                self.root.after(0, start_btn.config, {"state": tk.NORMAL})
                return


            # Determine device (prefer GPU if available)
            devices = ["cpu"]
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        devices.append(f"cuda:{i}")
                    devices.append("cuda")
            except ImportError:
                pass
            device = devices[-1]

            # Run one training iteration
            model_name = ckpt or ("yolov8n.pt" if task == "detect" else "yolov8n-seg.pt")
            self._log_active_learning(f"Iteration {iteration}: training model '{model_name}'...")
            save_dir = self.execute_training(
                model_name,
                epochs, imgsz, batch_size, lr,
                os.path.join(self.folder_path, "active_learning_runs", f"iter_{iteration:02d}"),
                True, "train_only",
                start_btn, window, device,
                active=True, stop_flag=self.active_learning_stop_flag,
            )
            if self.active_learning_stop_flag.is_set():
                self._log_active_learning("🛑 Active Learning training stopped by user.")
                self.root.after(0, start_btn.config, {"state": tk.NORMAL})
                return
            if not save_dir:
                self._log_active_learning("❌ Training did not complete successfully.")
                self.root.after(0, start_btn.config, {"state": tk.NORMAL})
                return

            # Load trained model for inference
            YOLO = lazy_importer.get_yolo()
            model_instance = YOLO(os.path.join(save_dir, "weights", "best.pt"))

            # Score unlabeled images by uncertainty
            unlabeled = [img for img in self.image_files
                         if self.image_status.get(img) not in ("edited", "review_needed")]
            scores = {}
            for img in unlabeled:
                if self.active_learning_stop_flag.is_set():
                    break
                results = model_instance(os.path.join(self.folder_path, img))
                confs = [box.conf[0].item() for box in results[0].boxes]
                scores[img] = 1.0 - max(confs) if confs else 1.0
            if self.active_learning_stop_flag.is_set():
                self._log_active_learning("🛑 Active Learning inference stopped by user.")
                self.root.after(0, start_btn.config, {"state": tk.NORMAL})
                return

            # Select top-K most uncertain images
            selected = [img for img, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:budget]
            for rel_img in selected:
                if self.image_tree.exists(rel_img):
                    self.image_tree.item(rel_img, tags=("review_needed",))
                    self.image_tree.see(rel_img)
                    self.image_status[rel_img] = "review_needed"
            self.save_statuses()
            self._log_active_learning(
                f"Iteration {iteration}: selected {len(selected)} images for annotation."
            )
            self._log_active_learning("Iteration complete. Please annotate the selected images.")
            self.root.after(0, start_btn.config, {"state": tk.NORMAL})

        threading.Thread(target=_worker, daemon=True).start()

    def _log_active_learning(self, message):
        """Write a message to the active learning progress widget"""
        self.active_learning_progress.config(state=tk.NORMAL)
        self.active_learning_progress.insert(tk.END, message + "\n")
        self.active_learning_progress.config(state=tk.DISABLED)
        self.active_learning_progress.see(tk.END)

    def export_format_selection_window(self):
        export_win = tk.Toplevel(self.root)
        export_win.title("Select Export Format")
        export_win.transient(self.root)
        export_win.grab_set()

        tk.Label(export_win, text="Choose an export format:").pack(pady=10, padx=10)

        export_format_var = tk.StringVar(value="coco")

        formats = [
            ("COCO (.json for all images)", "coco"),
            ("Pascal VOC (.xml per image)", "pascal_voc"),
            ("CSV (.csv for all images)", "csv"),
            ("YOLO (existing .txt files)", "yolo")
        ]

        for text, value in formats:
            tk.Radiobutton(export_win, text=text, variable=export_format_var, value=value).pack(anchor=tk.W, padx=20)

        def on_export():
            selected_format = export_format_var.get()
            export_win.destroy() 
            self._execute_export(selected_format)

        button_frame = tk.Frame(export_win)
        button_frame.pack(pady=10, padx=10, fill=tk.X)
        
        tk.Button(button_frame, text="Export", command=on_export).pack(side=tk.LEFT, expand=True, padx=5)
        tk.Button(button_frame, text="Cancel", command=export_win.destroy).pack(side=tk.RIGHT, expand=True, padx=5)
        
        export_win.update_idletasks() 
        center_window(export_win, 350, 230)

    # --------------------------------------------------
    # Setup / Layout Methods
    # --------------------------------------------------    
    def setup_image_list_panel_widgets(self): 
        header_frame = tk.Frame(self.image_list_frame)
        header_frame.pack(fill=tk.X, padx=2, pady=2)
        
        btn_frame = tk.Frame(header_frame)
        btn_frame.pack(side=tk.LEFT)
        
        expand_all_btn = tk.Button(btn_frame, text="⊞", command=self.expand_all_folders, width=3)
        expand_all_btn.pack(side=tk.LEFT, padx=1)
        
        collapse_all_btn = tk.Button(btn_frame, text="⊟", command=self.collapse_all_folders, width=3)
        collapse_all_btn.pack(side=tk.LEFT, padx=1)
        
        tk.Label(header_frame, text="Images", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(10, 0))
        
        self.image_tree = ttk.Treeview(
            self.image_list_frame,
            columns=("filename",),
            show="tree headings",
            selectmode="extended"
        )
        self.image_tree.heading("#0", text="Folder Structure")
        self.image_tree.heading("filename", text="File Info")
        self.image_tree.column("#0", width=200, anchor=tk.W)
        self.image_tree.column("filename", width=100, anchor=tk.W)
        self.image_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.image_tree.tag_configure("edited", background="lightgreen")
        self.image_tree.tag_configure("viewed", background="lightblue")
        self.image_tree.tag_configure("not_viewed", background="white")
        self.image_tree.tag_configure("review_needed", background="red")
        
        self.image_tree.tag_configure("folder", background="lightgray", font=("Arial", 9, "bold"))

        scrollbar = tk.Scrollbar(self.image_list_frame, orient="vertical", command=self.image_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_tree.configure(yscrollcommand=scrollbar.set)
        self.image_tree.bind("<<TreeviewSelect>>", self.on_image_select)
        self.image_tree.bind("<<TreeviewOpen>>", self.on_folder_expand)
        self.image_tree.bind("<<TreeviewClose>>", self.on_folder_collapse)

        # Batch operations: right-click context menu for multi-image status changes
        self.batch_menu = tk.Menu(self.root, tearoff=0)
        self.batch_menu.add_command(label="Mark as Not Viewed", command=lambda: self._batch_mark_status("not_viewed"))
        self.batch_menu.add_command(label="Mark as Viewed", command=lambda: self._batch_mark_status("viewed"))
        self.batch_menu.add_command(label="Mark as Edited", command=lambda: self._batch_mark_status("edited"))
        self.batch_menu.add_command(label="Mark as Review Needed", command=lambda: self._batch_mark_status("review_needed"))
        self.batch_menu.add_separator()
        self.batch_menu.add_command(label="Delete Annotations", command=self._batch_delete_annotations)
        self.image_tree.bind("<Button-3>", self._on_image_tree_right_click)

    def setup_canvas(self):
        self.canvas = tk.Canvas(self.content_frame, width=500, height=720)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_width = 500 
        self.canvas_height = 720
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel) 
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)  
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_pan_drag) 
        self.canvas.bind("<ButtonRelease-1>", self.on_pan_release) 
        self.canvas.bind("<Motion>", self.on_motion) 
        self.canvas.bind("<Double-Button-1>", self.on_double_click) 
        self.canvas.bind("<Button-3>", self.on_right_click) 
        self.canvas.bind("<Leave>", self._on_canvas_leave)

    def setup_info_panel(self):
        self.info_frame = tk.Frame(self.content_frame, width=300)
        self.info_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.info_label = tk.Label(self.info_frame, text="Annotations Info", font=("Arial", 14, "bold"))
        self.info_label.pack(pady=10)
        self.image_name_label = tk.Label(self.info_frame, text="", font=("Arial", 10))
        self.image_name_label.pack(pady=5)
        self.info_canvas = tk.Canvas(self.info_frame)
        self.info_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.info_scrollbar = tk.Scrollbar(self.info_frame, orient="vertical", command=self.info_canvas.yview)
        self.info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_canvas.configure(yscrollcommand=self.info_scrollbar.set)
        self.bbox_info_frame = tk.Frame(self.info_canvas)
        self.info_canvas.create_window((0, 0), window=self.bbox_info_frame, anchor="nw")
        self.bbox_info_frame.bind("<Configure>", lambda e: self.info_canvas.configure(scrollregion=self.info_canvas.bbox("all")))

    def setup_class_panel(self):
        self.class_label = tk.Label(self.class_frame, text="Classes", font=("Arial", 14))
        self.class_label.pack(pady=10)
        self.class_listbox = tk.Listbox(self.class_frame, exportselection=False) 
        self.class_listbox.pack(pady=10, fill=tk.BOTH, expand=True)
        for cls in self.class_names:
            self.class_listbox.insert(tk.END, cls)
        self.class_listbox.bind("<<ListboxSelect>>", self.on_class_select)
        self.class_listbox.bind("<ButtonRelease-1>", self.on_class_select) 
        self.class_entry = tk.Entry(self.class_frame)
        self.class_entry.pack(pady=5, fill=tk.X, padx=5)
        btn_frame = tk.Frame(self.class_frame)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Add", command=self.add_class).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Update", command=self.update_class).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Remove", command=self.remove_class).pack(side=tk.LEFT, padx=2)
        btn_frame2 = tk.Frame(self.class_frame)
        btn_frame2.pack(pady=5)
        tk.Button(btn_frame2, text="Reload from YAML", command=self.reload_classes_from_yaml).pack(padx=2)
        self.clear_selection_button = tk.Button(self.class_frame, text="Clear Selection", command=self.clear_class_selection)
        self.clear_selection_button.pack(pady=10)
        self.paste_all_button = tk.Button(self.class_frame, text="Paste All", command=self.paste_all_bboxes)
        self.paste_all_button.pack(pady=10)
        self.delete_image_button = tk.Button(self.class_frame, text="Delete Image", command=self.delete_image)
        self.delete_image_button.pack(pady=10)
        self.copy_frame = tk.Frame(self.class_frame)
        self.copy_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        self.copy_frame.bind("<MouseWheel>", self.on_mouse_wheel)
        self.copy_frame.bind("<Button-4>", self.on_mouse_wheel)
        self.copy_frame.bind("<Button-5>", self.on_mouse_wheel)
        self.copied_bbox_list = []
        self.update_copied_bbox_display()
        self.confidence_threshold = tk.DoubleVar(value=0.5)
        self.confidence_scale = tk.Scale(
            self.class_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
            label="Confidence Threshold", variable=self.confidence_threshold
        )
        self.confidence_scale.pack(pady=10)

    def setup_top_bar(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        buttons_frame = tk.Frame(top_frame)
        buttons_frame.pack(side=tk.LEFT, pady=5)
        self.auto_annotate_button = tk.Button(buttons_frame, text=f"{ICON_UNICODE['auto_annotate']} Auto Annotate", command=self.auto_annotate_dataset_threaded)
        self.auto_annotate_button.pack(side=tk.LEFT, padx=5)
        self.save_button = tk.Button(buttons_frame, text=f"{ICON_UNICODE['save']} Save Labels", command=self.save_labels)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.load_model_button = tk.Button(buttons_frame, text=f"{ICON_UNICODE['load_model']} Load Model", command=self.load_model)
        self.load_model_button.pack(side=tk.LEFT, padx=5)
        self.export_button = tk.Button(buttons_frame, text=f"{ICON_UNICODE['export']} Export Annotations", command=self.export_format_selection_window)
        self.export_button.pack(side=tk.LEFT, padx=5)
        self.train_button = tk.Button(
            buttons_frame,
            text=f"{ICON_UNICODE['train']} Standard Training",
            command=self.train_yolo_model
        )
        self.train_button.pack(side=tk.LEFT, padx=5)
        separator = ttk.Separator(buttons_frame, orient=tk.VERTICAL)
        separator.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        self.active_learning_button = tk.Button(
            buttons_frame,
            text=f"{ICON_UNICODE['active_learning']} Active-Learning Training",
            command=self.open_active_learning_dialog
        )
        self.active_learning_button.pack(side=tk.LEFT, padx=5)
        self.mode_toggle_button = tk.Button(buttons_frame, text=f"{ICON_UNICODE['mode_box']} Mode: Box", command=self.toggle_annotation_mode)
        self.mode_toggle_button.pack(side=tk.LEFT, padx=15)
        self.undo_button = tk.Button(buttons_frame, text=f"{ICON_UNICODE['undo']} Undo", command=self.undo, state=tk.DISABLED)
        self.undo_button.pack(side=tk.LEFT, padx=5)
        self.redo_button = tk.Button(buttons_frame, text=f"{ICON_UNICODE['redo']} Redo", command=self.redo, state=tk.DISABLED)
        self.redo_button.pack(side=tk.LEFT, padx=5)
        self.zoom_in_button = tk.Button(buttons_frame, text=f"{ICON_UNICODE['zoom_in']}", command=self.zoom_in)
        self.zoom_in_button.pack(side=tk.LEFT, padx=5)
        self.zoom_out_button = tk.Button(buttons_frame, text=f"{ICON_UNICODE['zoom_out']}", command=self.zoom_out)
        self.zoom_out_button.pack(side=tk.LEFT, padx=5)
        self.shortcuts_button = tk.Button(buttons_frame, text=f"{ICON_UNICODE['shortcuts']} Shortcuts", command=self.show_shortcuts)
        self.shortcuts_button.pack(side=tk.LEFT, padx=5)

    def setup_status_bar(self):
        self.status_frame = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(2,0), padx=2)
        self.status_labels = {}
        statuses = [("Viewed", "viewed"), ("Labeled", "edited"), ("Review Needed", "review_needed"), ("Non-viewed", "not_viewed")]
        for display_name, tag in statuses:
            frame = tk.Frame(self.status_frame)
            frame.pack(side=tk.LEFT, padx=10, pady=2)
            label = tk.Label(frame, text=f"{display_name}: 0", font=("Arial", 9))
            label.pack()
            self.status_labels[display_name] = label

        self.progress = ttk.Progressbar(
            self.status_frame,
            orient=tk.HORIZONTAL,
            length=200,
            mode='indeterminate'
        )

    def setup_bindings(self):
        self.root.bind("<Control-s>", lambda event: self.save_labels())
        self.root.bind("<Control-z>", lambda event: self.undo())
        self.root.bind("<Control-y>", lambda event: self.redo())
        self.root.bind("<Control-Shift-Z>", lambda event: self.redo()) 
        self.root.bind("<Escape>", self.on_escape_key)
        self.root.bind("<Down>", lambda event: self.navigate_image(+1))
        self.root.bind("<Up>", lambda event: self.navigate_image(-1))
        self.class_listbox.bind("<Down>", lambda e: "break")
        self.class_listbox.bind("<Up>", lambda e: "break")        
        self.root.bind("<Key>", self.on_key_press)
        self.root.bind("<Delete>", self.on_delete_vertex)
        self.root.bind("<BackSpace>", self.on_delete_vertex)
        self.canvas.bind("<Control-MouseWheel>", self.on_zoom)
        self.canvas.bind("<Control-Button-4>", self.on_zoom) 
        self.canvas.bind("<Control-Button-5>", self.on_zoom)  
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_release)

    def on_zoom(self, event):
        if hasattr(event, "delta"): delta = event.delta
        elif hasattr(event, "num") and event.num == 4: delta = 120
        elif hasattr(event, "num") and event.num == 5: delta = -120
        else: return
        factor = 1.1 if delta > 0 else 0.9
        new_zoom = self.zoom_level * factor
        new_zoom = max(0.1, min(new_zoom, 10.0))
        if new_zoom <= 1.0 and self.zoom_level > 1.0:
            self.image_view_offset_x = 0
            self.image_view_offset_y = 0
        self.zoom_level = new_zoom
        class _E: pass
        e = _E()
        e.width = self.canvas_width
        e.height = self.canvas_height
        self.on_canvas_resize(e)

    def on_mouse_wheel(self, event):
        if not self.image_files: return
        if hasattr(event, 'delta'): delta = event.delta
        elif event.num == 4: delta = 120
        elif event.num == 5: delta = -120
        else: return
        if delta > 0: self.navigate_image(-1)
        elif delta < 0: self.navigate_image(1)
        self.display_image()

    def on_canvas_resize(self, event):
        if hasattr(self, 'original_image') and self.original_image is not None:
            # Use throttled display for smooth resize/zoom
            self._schedule_display_image()

    def on_pan_start(self, event):
        if self.zoom_level > 1.0:
            self.panning = True
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.pan_start_view_offset_x = self.image_view_offset_x
            self.pan_start_view_offset_y = self.image_view_offset_y
            self.canvas.config(cursor="hand1")

    def on_pan_drag(self, event):
        if self.panning and self.zoom_level > 1.0 and self.original_image is not None:
            dx = self.pan_start_x - event.x
            dy = self.pan_start_y - event.y
            new_view_offset_x = self.pan_start_view_offset_x + dx
            new_view_offset_y = self.pan_start_view_offset_y + dy
            zoomed_width = int(self.original_image.width * self.zoom_level)
            zoomed_height = int(self.original_image.height * self.zoom_level)
            max_offset_x = max(0, zoomed_width - self.canvas.winfo_width())
            max_offset_y = max(0, zoomed_height - self.canvas.winfo_height())
            self.image_view_offset_x = max(0, min(new_view_offset_x, max_offset_x))
            self.image_view_offset_y = max(0, min(new_view_offset_y, max_offset_y))
            # Use throttled display for smooth panning
            self._schedule_display_image()
        elif self.annotation_mode == 'box' and self.current_bbox and self.rect and hasattr(self, 'current_bbox_orig_start') and self.current_bbox_orig_start:
            canvas_x_current, canvas_y_current = event.x, event.y
            image_x_current, image_y_current = self.canvas_to_image_coords(canvas_x_current, canvas_y_current)
            if image_x_current is not None and image_y_current is not None:
                x_orig_start, y_orig_start = self.current_bbox_orig_start
                new_x_orig = min(x_orig_start, image_x_current)
                new_y_orig = min(y_orig_start, image_y_current)
                new_w_orig = abs(x_orig_start - image_x_current)
                new_h_orig = abs(y_orig_start - image_y_current)
                self.current_bbox[0] = int(new_x_orig)
                self.current_bbox[1] = int(new_y_orig)
                self.current_bbox[2] = int(new_w_orig)
                self.current_bbox[3] = int(new_h_orig)
            canvas_x_start, canvas_y_start = self.rect_start_canvas
            self.canvas.coords(self.rect, canvas_x_start, canvas_y_start, canvas_x_current, canvas_y_current)
        elif self.dragging_point and self.annotation_mode == 'polygon' and self.drag_polygon_index != -1 and self.drag_point_index != -1:
            image_x_current, image_y_current = self.canvas_to_image_coords(event.x, event.y)
            if image_x_current is not None and image_y_current is not None:
                if 0 <= self.drag_polygon_index < len(self.polygons) and \
                   0 <= self.drag_point_index < len(self.polygons[self.drag_polygon_index]['points']):
                    
                    self.polygons[self.drag_polygon_index]['points'][self.drag_point_index] = (image_x_current, image_y_current)
                    
                    if len(self.polygons[self.drag_polygon_index]['points']) > 1 and \
                       self.polygons[self.drag_polygon_index]['points'][0] == self.polygons[self.drag_polygon_index]['points'][-1]:
                        if self.drag_point_index == 0:
                             self.polygons[self.drag_polygon_index]['points'][-1] = (image_x_current, image_y_current)                   
                        elif self.drag_point_index == len(self.polygons[self.drag_polygon_index]['points']) -1:
                             self.polygons[self.drag_polygon_index]['points'][0] = (image_x_current, image_y_current)

                    self.display_annotations() 

    def on_pan_release(self, event):
        if self.panning:
            self.panning = False
            self.canvas.config(cursor="")
        elif self.annotation_mode == 'box' and self.current_bbox and self.rect:
            self.canvas.delete(self.rect)
            self.rect = None
            self.current_bbox_orig_start = None
            self.rect_start_canvas = None
            self.display_annotations()
            self.save_history()        
        elif self.dragging_point and self.annotation_mode == 'polygon':
            self.dragging_point = False
            self.drag_polygon_index = -1
            self.drag_point_index = -1
            self.hover_polygon_index = -1 
            self.hover_point_index = -1 
            self._ignore_hover_until = time.perf_counter() + 0.15
            
            self.save_history()
            self.display_annotations() 
            self.canvas.config(cursor="")
        
        elif self.annotation_mode == 'polygon' and not self.polygon_drawing_active and not self.dragging_point:
            found_hover = False
            for poly_idx, poly_data in enumerate(self.polygons):
                points_orig = poly_data['points']
                for point_idx, (px_orig, py_orig) in enumerate(points_orig):
                    canvas_px, canvas_py = self.image_to_canvas_coords(px_orig, py_orig)
                    if canvas_px is not None and canvas_py is not None:
                        distance = ((event.x - canvas_px) ** 2 + (event.y - canvas_py) ** 2) ** 0.5
                        if distance <= 8: 
                            found_hover = True
                            break
                if found_hover:
                    break
            if not found_hover and (self.hover_polygon_index != -1 or self.hover_point_index != -1):
                self.clear_polygon_hover_state()
        
        if not self.panning and not self.dragging_point:
             self.canvas.config(cursor="")

    def clear_current_polygon_drawing(self):
        self.canvas.delete("polygon_drawing") 
        self.canvas.delete("polygon_hover_point") 
    
    def cancel_current_polygon(self):
        self.clear_current_polygon_drawing()
        self.current_polygon_points = []
        self.polygon_drawing_active = False
        self.display_annotations() 

    def clear_polygon_hover_state(self):
        if self.hover_polygon_index != -1 or self.hover_point_index != -1:
            self.hover_polygon_index = -1
            self.hover_point_index = -1
            self.canvas.config(cursor="")
            self._ignore_hover_until = time.perf_counter() + 0.1
            
            self.display_annotations()
            return True
        return False

    def on_escape_key(self, event):
        if self.annotation_mode == 'polygon' and self.polygon_drawing_active:
            self.cancel_current_polygon()
        elif self.annotation_mode == 'polygon':
            if not self.clear_polygon_hover_state():
                self.clear_class_selection()
        else:
            self.clear_class_selection()

    def on_right_click(self, event):
        if self.annotation_mode == 'polygon' and self.polygon_drawing_active:
            self.cancel_current_polygon()

    def on_delete_vertex(self, event):
        if self.annotation_mode == 'polygon' and not self.polygon_drawing_active:
            idx = self.hover_polygon_index
            vidx = self.hover_point_index
            if 0 <= idx < len(self.polygons) and 0 <= vidx < len(self.polygons[idx]['points']):
                points = self.polygons[idx]['points']
                if len(points) > 3:
                    del points[vidx]
                else:
                    if messagebox.askyesno(
                            "Delete Polygon",
                            "Deleting this vertex will remove the whole polygon. Proceed?"):
                        del self.polygons[idx]
                        self.hover_polygon_index = -1
                        self.hover_point_index = -1
                self.display_annotations()
                self.save_history()

    # --------------------------------------------------
    # YAML / Dataset / Project Setup
    # --------------------------------------------------

    def create_default_yaml_if_missing(self):
        if not os.path.exists(self.yaml_path):
            default_yaml = {
                "train": os.path.join(self.folder_path, 'train'), "val": os.path.join(self.folder_path, 'val'),
                "nc": 1, "names": ["person"], "auto_save_interval": 120
            }
            with open(self.yaml_path, "w") as f: yaml.dump(default_yaml, f, sort_keys=False)

    def load_dataset_async(self):
        """Load dataset in background to avoid blocking the UI."""
        for item in self.image_tree.get_children():
            self.image_tree.delete(item)
        self.progress.pack(side=tk.RIGHT, padx=10)
        self.progress.start()
        threading.Thread(target=self._load_dataset_worker, daemon=True).start()

    def _load_dataset_worker(self):
        if not self.folder_path:
            self.root.after(0, lambda: messagebox.showerror("Error", "Dataset folder not set."))
            self.root.after(0, self._stop_progress)
            return
        image_files = []
        folder_structure = {}
        for root_dir, _, files in os.walk(self.folder_path):
            for file in files:
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    relative_path = os.path.relpath(
                        os.path.join(root_dir, file), self.folder_path)
                    image_files.append(relative_path)
                    dir_part = os.path.dirname(relative_path) or "/"
                    folder_structure.setdefault(dir_part, []).append(relative_path)
        image_files.sort()
        if not image_files:
            self.root.after(0, lambda: messagebox.showinfo(
                "No Images", "No images found in the selected folder."))
            self.root.after(0, self._stop_progress)
            return
        self.load_statuses()
        self.root.after(0, lambda: self._finish_dataset_load(folder_structure, image_files))

    def _finish_dataset_load(self, folder_structure, image_files):
        self.progress.stop()
        self.progress.pack_forget()
        self.image_files = image_files
        self.folder_structure = folder_structure
        root_key = "/"
        for relative_image_path in sorted(self.folder_structure.get(root_key, [])):
            status = self.image_status.get(relative_image_path, "not_viewed")
            self.image_tree.insert(
                "", tk.END, iid=relative_image_path,
                text=os.path.basename(relative_image_path),
                values=(f"Status: {status}",), tags=(status,)
            )
        for folder_path_key in sorted(self.folder_structure.keys()):
            if folder_path_key == root_key:
                continue
            if os.path.dirname(folder_path_key):
                continue
            files_in_folder = self.folder_structure.get(folder_path_key, [])
            total_files = len(files_in_folder)
            status_counts = {"not_viewed": 0, "viewed": 0,
                             "edited": 0, "review_needed": 0}
            for file_path in files_in_folder:
                status_counts[self.image_status.get(file_path, "not_viewed")] += 1
            status_text = f"{total_files} files"
            if status_counts["edited"] > 0:
                status_text += f" ({status_counts['edited']} labeled)"
            folder_id = f"folder_{folder_path_key}"
            self.image_tree.insert(
                "", tk.END, iid=folder_id,
                text=f"📁 {os.path.basename(folder_path_key)}",
                values=(status_text,), tags=("folder",)
            )
            if self._has_children_folder(folder_path_key):
                self.image_tree.insert(
                    folder_id, tk.END, text="", values=("",),
                    tags=("dummy",)
                )
        if not self.folder_structure.get(root_key):
            for folder_path_key in sorted(self.folder_structure.keys()):
                if folder_path_key == root_key or os.path.dirname(folder_path_key):
                    continue
                folder_id = f"folder_{folder_path_key}"
                self.image_tree.item(folder_id, open=True)
                self.on_folder_expand(None, folder_id)
        self.save_statuses()
        self.update_status_labels()
        # After dataset load completes, restore last opened image selection
        self.root.after_idle(self._attempt_load_initial_image)

    def _stop_progress(self):
        self.progress.stop()
        self.progress.pack_forget()
    
    def load_dataset(self):
        if not self.folder_path: 
            messagebox.showerror("Error", "Dataset folder not set.")
            return
            
        for item in self.image_tree.get_children(): 
            self.image_tree.delete(item)
            
        self.image_files = []
        folder_structure = {} 
        
        for root_dir, _, files in os.walk(self.folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    relative_path = os.path.relpath(os.path.join(root_dir, file), self.folder_path)
                    self.image_files.append(relative_path)
                    
                    dir_part = os.path.dirname(relative_path)
                    if dir_part == "": 
                        dir_part = "/"
                    
                    if dir_part not in folder_structure:
                        folder_structure[dir_part] = []
                    folder_structure[dir_part].append(relative_path)
        
        self.image_files.sort()
        if not self.image_files:
            messagebox.showinfo("No Images", "No images found in the selected folder.")
            return

        self.load_statuses()
        self.folder_structure = folder_structure
        root_key = "/"
        for relative_image_path in sorted(self.folder_structure.get(root_key, [])):
            status = self.image_status.get(relative_image_path, "not_viewed")
            self.image_tree.insert("", tk.END, iid=relative_image_path,
                                   text=os.path.basename(relative_image_path),
                                   values=(f"Status: {status}",), tags=(status,))

        for folder_path_key in sorted(self.folder_structure.keys()):
            if folder_path_key == root_key:
                continue
            if os.path.dirname(folder_path_key):
                continue
            files_in_folder = self.folder_structure.get(folder_path_key, [])
            total_files = len(files_in_folder)
            status_counts = {"not_viewed": 0, "viewed": 0, "edited": 0, "review_needed": 0}
            for file_path in files_in_folder:
                status_counts[self.image_status.get(file_path, "not_viewed")] += 1
            status_text = f"{total_files} files"
            if status_counts["edited"] > 0:
                status_text += f" ({status_counts['edited']} labeled)"
            folder_id = f"folder_{folder_path_key}"
            self.image_tree.insert("", tk.END, iid=folder_id,
                                   text=f"📁 {os.path.basename(folder_path_key)}",
                                   values=(status_text,), tags=("folder",))
            if self._has_children_folder(folder_path_key):
                self.image_tree.insert(folder_id, tk.END, text="", values=("",), tags=("dummy",))

        self.save_statuses()
        self.update_status_labels()

    # --------------------------------------------------
    # Folder Management for Hierarchical Image List
    # --------------------------------------------------
    
    def expand_all_folders(self):
        def expand_recursive(item):
            if self.image_tree.item(item)["tags"] and "folder" in self.image_tree.item(item)["tags"]:
                self.image_tree.item(item, open=True)
            for child in self.image_tree.get_children(item):
                expand_recursive(child)
        
        for child in self.image_tree.get_children():
            expand_recursive(child)
    
    def collapse_all_folders(self):
        def collapse_recursive(item):
            if self.image_tree.item(item)["tags"] and "folder" in self.image_tree.item(item)["tags"]:
                self.image_tree.item(item, open=False)
            for child in self.image_tree.get_children(item):
                collapse_recursive(child)
        
        for child in self.image_tree.get_children():
            collapse_recursive(child)
    
    def _has_children_folder(self, folder_key):
        """
        Determine if a folder_key has any direct subfolders or images in the full folder_structure.
        """
        for key in self.folder_structure.keys():
            if key != "/" and os.path.dirname(key) == folder_key:
                return True
        if self.folder_structure.get(folder_key):
            return True
        return False
    
    def on_folder_expand(self, event=None, folder_id=None):
        """
        Lazy-load children of a folder node when expanded.
        """
        if folder_id is not None:
            item = folder_id
        else:
            item = self.image_tree.focus()
        if not item or "folder" not in self.image_tree.item(item).get("tags", []):
            return
        dummy_found = False
        for child in self.image_tree.get_children(item):
            if "dummy" in self.image_tree.item(child).get("tags", []):
                self.image_tree.delete(child)
                dummy_found = True
                break
        if not dummy_found:
            return
        folder_key = item.replace("folder_", "", 1)
        for relative_image_path in sorted(self.folder_structure.get(folder_key, [])):
            status = self.image_status.get(relative_image_path, "not_viewed")
            self.image_tree.insert(item, tk.END, iid=relative_image_path,
                                   text=os.path.basename(relative_image_path),
                                   values=(f"Status: {status}",), tags=(status,))
        for child_folder_key in sorted(self.folder_structure.keys()):
            if os.path.dirname(child_folder_key) != folder_key:
                continue
            files_in_folder = self.folder_structure.get(child_folder_key, [])
            total_files = len(files_in_folder)
            status_counts = {"not_viewed": 0, "viewed": 0, "edited": 0, "review_needed": 0}
            for p in files_in_folder:
                status_counts[self.image_status.get(p, "not_viewed")] += 1
            status_text = f"{total_files} files"
            if status_counts["edited"] > 0:
                status_text += f" ({status_counts['edited']} labeled)"
            sub_id = f"folder_{child_folder_key}"
            self.image_tree.insert(item, tk.END, iid=sub_id,
                                   text=f"📁 {os.path.basename(child_folder_key)}",
                                   values=(status_text,), tags=("folder",))
            if self._has_children_folder(child_folder_key):
                self.image_tree.insert(sub_id, tk.END, text="", values=("",), tags=("dummy",))
        self.update_folder_status_display()
    
    def on_folder_collapse(self, event):
        pass
    
    def update_folder_status_display(self):
        def update_folder_recursive(folder_id):
            children = self.image_tree.get_children(folder_id)
            if not children:
                return
            
            total_files = 0
            status_counts = {"not_viewed": 0, "viewed": 0, "edited": 0, "review_needed": 0}
            
            for child in children:
                if self.image_tree.item(child)["tags"] and "folder" in self.image_tree.item(child)["tags"]:
                    update_folder_recursive(child)
                else:
                    total_files += 1
                    child_tags = self.image_tree.item(child)["tags"]
                    if child_tags:
                        status = child_tags[0]
                        if status in status_counts:
                            status_counts[status] += 1
            
            if total_files > 0:
                status_text = f"{total_files} files"
                if status_counts["edited"] > 0:
                    status_text += f" ({status_counts['edited']} labeled)"
                
                self.image_tree.item(folder_id, values=(status_text,))
        
        for child in self.image_tree.get_children():
            if self.image_tree.item(child)["tags"] and "folder" in self.image_tree.item(child)["tags"]:
                update_folder_recursive(child)
    
    # --------------------------------------------------
    # Status Persistence
    # --------------------------------------------------

    def save_statuses(self):
        if self.folder_path:
            status_file = os.path.join(self.folder_path, "image_status.json")
            with open(status_file, "w") as f: json.dump(self.image_status, f)

    def load_statuses(self):
        if self.folder_path:
            status_file = os.path.join(self.folder_path, "image_status.json")
            if os.path.exists(status_file):
                with open(status_file, "r") as f: self.image_status = json.load(f)
            else: self.image_status = {}

    def update_status_labels(self):
        counts = {"Viewed": 0, "Labeled": 0, "Review Needed": 0, "Non-viewed": 0}
        for relative_image_path in self.image_files:
            status = self.image_status.get(relative_image_path, "not_viewed")
            if status == "edited": counts["Labeled"] += 1; counts["Viewed"] += 1
            elif status == "viewed": counts["Viewed"] += 1
            elif status == "review_needed": counts["Review Needed"] += 1
            elif status == "not_viewed": counts["Non-viewed"] += 1
        for display_name in counts: self.status_labels[display_name].config(text=f"{display_name}: {counts[display_name]}")

    # --------------------------------------------------
    # Auto-Save Mechanism
    # --------------------------------------------------

    def start_auto_save(self):
        self.auto_save_id = self.root.after(self.auto_save_interval * 1000, self._auto_save_callback)

    def _auto_save_callback(self): self.save_labels(); self.start_auto_save()

    # --------------------------------------------------
    # Class Management
    # --------------------------------------------------

    def update_class_colors(self):
        predefined_colors = ["red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple", "brown", "pink"]
        self.class_colors = {i: predefined_colors[i % len(predefined_colors)] for i in range(len(self.class_names))}

    def add_class(self):
        new_class = self.class_entry.get().strip()
        if new_class:
            self.class_listbox.insert(tk.END, new_class)
            self.class_names.append(new_class)
            self.update_class_colors(); self.update_yaml_classes(); self.class_entry.delete(0, tk.END)

    def update_class(self):
        selection = self.class_listbox.curselection()
        if selection:
            index = selection[0]; new_val = self.class_entry.get().strip()
            if new_val:
                self.class_listbox.delete(index); self.class_listbox.insert(index, new_val)
                self.class_names[index] = new_val
                self.update_class_colors(); self.update_yaml_classes(); self.class_entry.delete(0, tk.END)
    
    def remove_class(self):
        selection = self.class_listbox.curselection()
        if selection:
            if len(self.class_names) == 1: messagebox.showwarning("Warning", "You must have at least one class."); return
            index = selection[0]
            self.class_listbox.delete(index); self.class_names.pop(index)
            self.update_class_colors(); self.update_yaml_classes()
            
            updated_bboxes = []
            max_idx = len(self.class_names) - 1
            for x, y, w, h, class_id in self.bboxes:
                if class_id > max_idx: class_id = 0
                updated_bboxes.append((x, y, w, h, class_id))
            self.bboxes = updated_bboxes
            
            updated_polygons = []
            for poly_data in self.polygons:
                class_id = poly_data['class_id']
                if class_id > max_idx: class_id = 0
                updated_polygons.append({'class_id': class_id, 'points': poly_data['points']})
            self.polygons = updated_polygons
            
            self.display_annotations()

    def update_yaml_classes(self):
        try:
            with open(self.yaml_path, "r") as f: data = yaml.safe_load(f)
        except Exception: data = {}
        data["nc"] = len(self.class_names); data["names"] = self.class_names
        data["train"] = os.path.join(self.folder_path, 'train')
        data["val"] = os.path.join(self.folder_path, 'val')        
        with open(self.yaml_path, "w") as f: yaml.dump(data, f, sort_keys=False)

    # --------------------------------------------------
    # History Management (Undo/Redo)
    # --------------------------------------------------

    def save_history(self):
        if self.current_image_index == -1: return
        current_state = {
            'bboxes': [bbox[:] for bbox in self.bboxes], 
            'polygons': [{'class_id': p['class_id'], 'points': p['points'][:]} for p in self.polygons], 
            'image_index': self.current_image_index
        }
        if self.history_index < len(self.history) - 1: self.history = self.history[:self.history_index + 1]
        self.history.append(current_state)
        self.history_index = len(self.history) - 1
        if len(self.history) > self.max_history_size: 
            self.history.pop(0)
            self.history_index -= 1
        self.update_undo_redo_buttons()
    
    def restore_from_history(self):
        if 0 <= self.history_index < len(self.history):
            state = self.history[self.history_index]
            
            if 'image_index' in state and state['image_index'] != self.current_image_index:
                if 0 <= state['image_index'] < len(self.image_files):
                    target_image_path = os.path.join(self.folder_path, self.image_files[state['image_index']])
                    self.load_image(target_image_path)
                    relative_image_path = self.image_files[state['image_index']]
                    try:
                        if self.image_tree.exists(relative_image_path):
                            self.image_tree.selection_set(relative_image_path)
                            self.image_tree.focus(relative_image_path)
                            self.image_tree.see(relative_image_path)
                    except tk.TclError:
                        pass 
            
            self.bboxes = [bbox[:] for bbox in state['bboxes']]
            self.polygons = [{'class_id': p['class_id'], 'points': p['points'][:]} for p in state['polygons']]
            self.display_annotations()
            self.update_undo_redo_buttons()

    def update_undo_redo_buttons(self):
        self.undo_button.config(state=tk.NORMAL if self.history_index > 0 else tk.DISABLED)
        self.redo_button.config(state=tk.NORMAL if self.history_index < len(self.history) - 1 else tk.DISABLED)

    # --------------------------------------------------
    # Image Navigation / Display
    # --------------------------------------------------    
    def on_image_select(self, event):
        selected = self.image_tree.selection()
        if selected:
            selected_item = selected[0]
            item_tags = self.image_tree.item(selected_item)["tags"]
            
            if item_tags and "folder" in item_tags:
                if self.image_tree.item(selected_item, "open"):
                    self.image_tree.item(selected_item, open=False)
                else:
                    self.image_tree.item(selected_item, open=True)
                return
            
            if selected_item.startswith("folder_"):
                return 
                
            relative_image_path = selected_item
            image_path = os.path.join(self.folder_path, relative_image_path)
            
            if os.path.exists(image_path) and relative_image_path in self.image_files:
                self.load_image(image_path)

    def load_image(self, image_path=None):
        if image_path:
            self.image_path = image_path
            self.current_image_index = self.image_files.index(os.path.relpath(image_path, self.folder_path))
        else:
            messagebox.showwarning("Manual Load", "Manually loaded images are not part of the project's dataset structure and won't be saved.")
            self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
            if not self.image_path: return
            self.current_image_index = -1 

        if self.image_path in self.image_cache:
            self.original_image = self.image_cache.pop(self.image_path)
            self.image_cache[self.image_path] = self.original_image
        else:
            cv2_module = lazy_importer.get_cv2()
            original_image_cv = cv2_module.imread(self.image_path)
            if original_image_cv is None:
                messagebox.showerror("Error", f"Failed to load image: {self.image_path}\nFile might be missing, corrupted, or in an unsupported format.")
                self.image = None
                self.original_image = None
                self.image_name_label.config(text=f"Error loading: {os.path.basename(self.image_path)}")
                self.bboxes = []
                self.polygons = []
                self.display_image()
                self.display_annotations()
                return

            original_image_cv = cv2_module.cvtColor(original_image_cv, cv2_module.COLOR_BGR2RGB)
            Image, _ = _get_pil()
            self.original_image = Image.fromarray(original_image_cv)
            self.image_cache[self.image_path] = self.original_image
            if len(self.image_cache) > self.max_cache_size:
                self.image_cache.popitem(last=False)
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = self.canvas_width, self.canvas_height

        width_scale = canvas_width / self.original_image.width
        height_scale = canvas_height / self.original_image.height
        
        initial_fit_zoom = min(width_scale, height_scale)
        
        self.zoom_level = max(0.1, initial_fit_zoom)
        if self.zoom_level > 1.0: 
            self.zoom_level = 1.0
        
        self.image_view_offset_x = 0; self.image_view_offset_y = 0
        
        self.display_image()

        relative_image_path_for_label = os.path.relpath(self.image_path, self.folder_path)
        label_relative_path = os.path.splitext(relative_image_path_for_label)[0] + '.txt'
        label_path = os.path.join(self.label_folder, label_relative_path)
        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        self.bboxes, self.polygons = read_annotations_from_file(label_path, (self.original_image.height, self.original_image.width))
        self.display_annotations()

        relative_image_path = os.path.relpath(self.image_path, self.folder_path)
        new_status = "edited" if (self.bboxes or self.polygons) else "viewed"
        self.image_status[relative_image_path] = new_status
        self.image_tree.item(relative_image_path, tags=(new_status,))
        self.save_statuses(); self.update_status_labels()
        self.image_name_label.config(text=relative_image_path)
        if self.selected_class_index is not None: self.class_listbox.selection_set(self.selected_class_index)

        if self.original_image is not None and self.image_path and self.current_image_index != -1:
            relative_image_path = os.path.relpath(self.image_path, self.folder_path)
            self.project['last_opened_image_relative'] = relative_image_path
            self._save_project_config()

    def _save_project_config(self):
        if not hasattr(self, 'project') or 'project_name' not in self.project:
            return 
        project_name = self.project['project_name']
        safe_project_filename = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in project_name).rstrip()
        if not safe_project_filename: safe_project_filename = "Untitled_Project"
        project_file_path = os.path.join(PROJECTS_DIR, f"{safe_project_filename}.json")
        try:
            with open(project_file_path, "w") as f: json.dump(self.project, f, indent=4)
        except Exception as e: 
            pass 

    def _schedule_display_image(self):
        """Throttled version of display_image to avoid excessive redraws."""
        if self._pending_redraw is not None:
            return  # Already scheduled
        self._pending_redraw = self.root.after(self._redraw_throttle_ms, self._execute_display_image)

    def _execute_display_image(self):
        """Execute the actual display update."""
        self._pending_redraw = None
        self.display_image()

    def display_image(self):
        if self.original_image is None: return
        canvas_width = self.canvas.winfo_width(); canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1: canvas_width, canvas_height = self.canvas_width, self.canvas_height

        zoomed_img_width = int(self.original_image.width * self.zoom_level)
        zoomed_img_height = int(self.original_image.height * self.zoom_level)

        if (self.zoom_level > 1.0 and self.image_view_offset_x == 0 and self.image_view_offset_y == 0 and
            zoomed_img_width > canvas_width and zoomed_img_height > canvas_height):
            self.image_view_offset_x = (zoomed_img_width - canvas_width) // 2
            self.image_view_offset_y = (zoomed_img_height - canvas_height) // 2

        max_offset_x = max(0, zoomed_img_width - canvas_width)
        max_offset_y = max(0, zoomed_img_height - canvas_height)
        self.image_view_offset_x = max(0, min(self.image_view_offset_x, max_offset_x))
        self.image_view_offset_y = max(0, min(self.image_view_offset_y, max_offset_y))

        crop_x1 = self.image_view_offset_x; crop_y1 = self.image_view_offset_y
        crop_x2 = self.image_view_offset_x + canvas_width; crop_y2 = self.image_view_offset_y + canvas_height

        Image, ImageTk = _get_pil()
        scaled_image = self.original_image.resize((zoomed_img_width, zoomed_img_height), Image.Resampling.NEAREST)
        display_crop_x1 = int(crop_x1); display_crop_y1 = int(crop_y1)
        display_crop_x2 = int(min(crop_x2, zoomed_img_width)); display_crop_y2 = int(min(crop_y2, zoomed_img_height))

        if display_crop_x1 >= zoomed_img_width or display_crop_y1 >= zoomed_img_height:
            self.canvas.delete("image"); self.tk_image = None; return

        cropped_image_pil = scaled_image.crop((display_crop_x1, display_crop_y1, display_crop_x2, display_crop_y2))

        if zoomed_img_width < canvas_width: self.image_offset_x = (canvas_width - zoomed_img_width) // 2
        else: self.image_offset_x = 0
        if zoomed_img_height < canvas_height: self.image_offset_y = (canvas_height - zoomed_img_height) // 2
        else: self.image_offset_y = 0

        self.canvas.delete("image")
        self.tk_image = ImageTk.PhotoImage(cropped_image_pil)
        self.canvas.create_image(self.image_offset_x, self.image_offset_y, anchor=tk.NW, image=self.tk_image, tags="image")
        self.display_annotations()

    def display_annotations(self):
        self.canvas.delete("bbox"); self.canvas.delete("polygon")
        for widget in self.bbox_info_frame.winfo_children(): widget.destroy()
 
        for i, (x_orig, y_orig, w_orig, h_orig, class_id) in enumerate(self.bboxes):
            color = self.class_colors.get(class_id, "red")
            canvas_x1, canvas_y1 = self.image_to_canvas_coords(x_orig, y_orig)
            canvas_x2, canvas_y2 = self.image_to_canvas_coords(x_orig + w_orig, y_orig + h_orig)
            if canvas_x1 is not None and canvas_y1 is not None and canvas_x2 is not None and canvas_y2 is not None:
                self.canvas.create_rectangle(canvas_x1, canvas_y1, canvas_x2, canvas_y2, outline=color, width=2, tags="bbox")
                self.canvas.create_text(canvas_x1, canvas_y1 - 10, text=self.class_names[class_id], fill=color, anchor=tk.NW, tags="bbox", font=("Arial", 8, "bold"))
            bbox_info_row = tk.Frame(self.bbox_info_frame, bd=1, relief="solid", padx=2, pady=2); bbox_info_row.pack(fill=tk.X, pady=2)
            tk.Label(bbox_info_row, text=f"Box: {self.class_names[class_id]}", font=("Arial", 9)).grid(row=0, column=0, sticky="w")
            tk.Label(bbox_info_row, text=f"Pos:({x_orig},{y_orig}) Size:({w_orig},{h_orig})", font=("Arial", 8)).grid(row=1, column=0, sticky="w")
            tk.Button(bbox_info_row, text="Copy", command=lambda bbox=(x_orig,y_orig,w_orig,h_orig,class_id): self.copy_bbox(bbox), font=("Arial",8)).grid(row=0,column=1,padx=2,sticky="e")
            tk.Button(bbox_info_row, text="Delete", command=lambda i=i, type='bbox': self.delete_annotation(i, type), font=("Arial",8)).grid(row=1,column=1,padx=2,sticky="e")
            bbox_info_row.grid_columnconfigure(0, weight=1)

        for i, poly_data in enumerate(self.polygons):
            class_id = poly_data['class_id']; points_orig = poly_data['points']; color = self.class_colors.get(class_id, "blue")
            if len(points_orig) > 1:
                canvas_coords_flat = []
                for p_x_orig, p_y_orig in points_orig:
                    c_x, c_y = self.image_to_canvas_coords(p_x_orig, p_y_orig)
                    if c_x is not None and c_y is not None: canvas_coords_flat.extend([c_x, c_y])
                if len(canvas_coords_flat) >= 4:
                    self.canvas.create_polygon(canvas_coords_flat, outline=color, fill="", width=2, tags="polygon")
                    if canvas_coords_flat: self.canvas.create_text(canvas_coords_flat[0], canvas_coords_flat[1] - 10, text=self.class_names[class_id], fill=color, anchor=tk.NW, tags="polygon", font=("Arial", 8, "bold"))
                
                for point_idx, (px_orig, py_orig) in self._iter_poly_vertices(points_orig):
                    canvas_px, canvas_py = self.image_to_canvas_coords(px_orig, py_orig)
                    if canvas_px is not None and canvas_py is not None:
                        is_hovered = (i == self.hover_polygon_index and point_idx == self.hover_point_index)
                        if not is_hovered: 
                            self.canvas.create_oval(canvas_px-3, canvas_py-3, canvas_px+3, canvas_py+3, fill=color, outline="white", width=1, tags="polygon")
                
                for point_idx, (px_orig, py_orig) in self._iter_poly_vertices(points_orig):
                    canvas_px, canvas_py = self.image_to_canvas_coords(px_orig, py_orig)
                    if canvas_px is not None and canvas_py is not None:
                        is_hovered = (i == self.hover_polygon_index and point_idx == self.hover_point_index)
                        if is_hovered: 
                            self.canvas.create_oval(canvas_px-5, canvas_py-5, canvas_px+5, canvas_py+5, fill="yellow", outline="orange", width=2, tags="polygon")
            poly_info_row = tk.Frame(self.bbox_info_frame, bd=1, relief="solid", padx=2, pady=2); poly_info_row.pack(fill=tk.X, pady=2)
            tk.Label(poly_info_row, text=f"Poly: {self.class_names[class_id]}", font=("Arial",9)).grid(row=0,column=0,sticky="w")
            tk.Label(poly_info_row, text=f"Points: {len(points_orig)}", font=("Arial",8)).grid(row=1,column=0,sticky="w")
            tk.Button(poly_info_row, text="Delete", command=lambda i=i, type='polygon': self.delete_annotation(i, type), font=("Arial",8)).grid(row=0,column=1,rowspan=2,padx=2,sticky="ns")
            poly_info_row.grid_columnconfigure(0, weight=1)

    def canvas_to_image_coords(self, canvas_x, canvas_y):
        if not self.original_image or self.original_image is None: return None, None
        x_on_displayed_image = canvas_x - self.image_offset_x
        y_on_displayed_image = canvas_y - self.image_offset_y
        x_on_scaled_image = x_on_displayed_image + self.image_view_offset_x
        y_on_scaled_image = y_on_displayed_image + self.image_view_offset_y
        original_x = x_on_scaled_image / self.zoom_level
        original_y = y_on_scaled_image / self.zoom_level
        if 0 <= original_x < self.original_image.width and 0 <= original_y < self.original_image.height:
            return original_x, original_y
        return None, None

    def image_to_canvas_coords(self, image_x, image_y):
        if not self.original_image or self.original_image is None: return None, None
        scaled_x = image_x * self.zoom_level; scaled_y = image_y * self.zoom_level
        panned_x = scaled_x - self.image_view_offset_x; panned_y = scaled_y - self.image_view_offset_y
        canvas_x = panned_x + self.image_offset_x; canvas_y = panned_y + self.image_offset_y
        return canvas_x, canvas_y

    def is_click_on_polygon_edge(self, click_x, click_y):
        threshold = 5.0 

        for poly_data in self.polygons:
            points_orig = poly_data['points']
            if len(points_orig) < 2:
                continue

            canvas_points = []
            for px_orig, py_orig in points_orig:
                c_x, c_y = self.image_to_canvas_coords(px_orig, py_orig)
                if c_x is None or c_y is None: 
                    break 
                canvas_points.append((c_x, c_y))
            
            if len(canvas_points) < len(points_orig): 
                continue

            for i in range(len(canvas_points) - 1):
                p1 = canvas_points[i]
                p2 = canvas_points[i+1]
                x1, y1 = p1
                x2, y2 = p2

                L2 = (x2 - x1)**2 + (y2 - y1)**2
                
                if L2 == 0: 
                    dist = ((click_x - x1)**2 + (click_y - y1)**2)**0.5
                else:
                    dot_product = (click_x - x1) * (x2 - x1) + (click_y - y1) * (y2 - y1)
                    t = dot_product / L2

                    if 0 <= t <= 1: 
                        proj_x = x1 + t * (x2 - x1)
                        proj_y = y1 + t * (y2 - y1)
                        dist = ((click_x - proj_x)**2 + (click_y - proj_y)**2)**0.5
                    else: 
                        dist_to_p1 = ((click_x - x1)**2 + (click_y - y1)**2)**0.5
                        dist_to_p2 = ((click_x - x2)**2 + (click_y - y2)**2)**0.5
                        dist = min(dist_to_p1, dist_to_p2)
                
                if dist < threshold:
                    return True
        
        return False 
    
    # --------------------------------------------------
    # Event Handlers
    # --------------------------------------------------
    
    def on_click(self, event):
        
        if self.annotation_mode == 'box' and not self.polygon_drawing_active:
            if self.selected_class_index is None:
                messagebox.showwarning("No Class Selected", "Please select a class from the list before drawing a bounding box.", parent=self.root); return
            image_x_start, image_y_start = self.canvas_to_image_coords(event.x, event.y)
            if image_x_start is not None and image_y_start is not None:
                self.current_bbox_orig_start = (image_x_start, image_y_start)
                self.current_bbox = [image_x_start, image_y_start, 0, 0, self.selected_class_index]
                self.bboxes.append(self.current_bbox)
                self.rect_start_canvas = (event.x, event.y)
                self.rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="blue", width=2, tags="bbox_drawing")
            else:
                self.current_bbox = None; self.current_bbox_orig_start = None; self.rect_start_canvas = None; self.rect = None
        elif self.annotation_mode == 'polygon':
            image_x, image_y = self.canvas_to_image_coords(event.x, event.y)
            if image_x is not None and image_y is not None:
                if not self.polygon_drawing_active:
                    if self.hover_polygon_index != -1 and self.hover_point_index != -1 and \
                       0 <= self.hover_polygon_index < len(self.polygons) and \
                       0 <= self.hover_point_index < len(self.polygons[self.hover_polygon_index]['points']):

                        self.dragging_point = True
                        self.drag_polygon_index = self.hover_polygon_index
                        self.drag_point_index = self.hover_point_index
                        self.canvas.config(cursor="fleur")
                        return                    
                    if self.is_click_on_polygon_edge(event.x, event.y):
                        return
                    
                    current_selection_tuple = self.class_listbox.curselection()
                    if not current_selection_tuple:
                        messagebox.showwarning("No Class Selected", "Please select a class before drawing a polygon.", parent=self.root)
                        return
                    self.selected_class_index = current_selection_tuple[0]
                    
                    self.current_polygon_points = [(image_x, image_y)]
                    self.polygon_drawing_active = True
                    self.draw_current_polygon_drawing() 
                else: 
                    self.current_polygon_points.append((image_x, image_y))
                    self.draw_current_polygon_drawing()
    
    def _iter_poly_vertices(self, points):
        if len(points) > 1 and points[0] == points[-1]:
            return enumerate(points[:-1])          
        return enumerate(points)                   
    
    def _update_hover_state(self, canvas_x: int, canvas_y: int) -> None:
        if (hasattr(self, "_ignore_hover_until") and
            self._ignore_hover_until > time.perf_counter()):
            return

        thresh_sq = 8 ** 2
        new_poly = new_point = -1

        for poly_idx, poly in enumerate(self.polygons):
            for pt_idx, (px, py) in self._iter_poly_vertices(poly["points"]):
                cx, cy = self.image_to_canvas_coords(px, py)
                if cx is None:
                    continue
                if (canvas_x - cx) ** 2 + (canvas_y - cy) ** 2 <= thresh_sq:
                    new_poly, new_point = poly_idx, pt_idx
                    break        
            if new_poly != -1:
                break

        if (new_poly, new_point) != (self.hover_polygon_index, self.hover_point_index):
            self.hover_polygon_index, self.hover_point_index = new_poly, new_point
            self.canvas.config(cursor="hand2" if new_poly != -1 else "")
            self.display_annotations()              
    
    def _on_canvas_leave(self, event):
        if self.hover_polygon_index != -1:
            self.hover_polygon_index = self.hover_point_index = -1
            self.canvas.config(cursor="")
            self.display_annotations()
    
    def on_motion(self, event):
        if self.annotation_mode == 'polygon' and self.polygon_drawing_active and self.current_polygon_points:
            self.draw_current_polygon_drawing(live_canvas_x=event.x, live_canvas_y=event.y)
        elif self.annotation_mode == "polygon" and not self.dragging_point:
            self._update_hover_state(event.x, event.y)

    def draw_current_polygon_drawing(self, live_canvas_x=None, live_canvas_y=None):
        self.canvas.delete("polygon_drawing") 

        if not self.current_polygon_points: 
            return

        committed_canvas_points = []
        for x_orig, y_orig in self.current_polygon_points:
            cx, cy = self.image_to_canvas_coords(x_orig, y_orig)
            if cx is not None and cy is not None: 
                committed_canvas_points.append((cx, cy))
            else: 
                return 

        if not committed_canvas_points:
            return

        for x_c, y_c in committed_canvas_points:
            self.canvas.create_oval(
                x_c - 3, y_c - 3, x_c + 3, y_c + 3,
                fill="red", outline="red", tags="polygon_drawing"
            )

        if len(committed_canvas_points) > 1:
            flat_committed_coords = [coord for point_tuple in committed_canvas_points for coord in point_tuple]
            self.canvas.create_line(
                flat_committed_coords, fill="red", width=2, tags="polygon_drawing"
            )

        if live_canvas_x is not None and live_canvas_y is not None and self.polygon_drawing_active:
            if committed_canvas_points: 
                last_committed_x, last_committed_y = committed_canvas_points[-1]
                
                self.canvas.create_line(
                    last_committed_x, last_committed_y, live_canvas_x, live_canvas_y,
                    fill="red", width=2, tags="polygon_drawing"
                )

                if len(committed_canvas_points) >= 1: 
                    first_committed_x, first_committed_y = committed_canvas_points[0]
                    self.canvas.create_line(
                        live_canvas_x, live_canvas_y, first_committed_x, first_committed_y,
                        fill="red", width=2, dash=(4, 2), tags="polygon_drawing"
                    )
        elif self.polygon_drawing_active and len(committed_canvas_points) > 2:
            x_last, y_last = committed_canvas_points[-1]
            x_first, y_first = committed_canvas_points[0]
            self.canvas.create_line(
                x_last, y_last, x_first, y_first,
                fill="red", width=2, dash=(4, 2), tags="polygon_drawing"
        )    
    
    def on_double_click(self, event):
        if self.annotation_mode == 'polygon' and self.polygon_drawing_active:
            if self.current_polygon_points and len(self.current_polygon_points) > 2:
                
                if self.selected_class_index is None:
                    messagebox.showwarning("No Class Selected", "Please select a class before completing the polygon.", parent=self.root)
                    self.cancel_current_polygon() 
                    return

                self.polygons.append({
                    "class_id": self.selected_class_index, 
                    "points": self.current_polygon_points[:] 
                })
                self.current_polygon_points = [] 
                self.polygon_drawing_active = False
                self.clear_current_polygon_drawing() 
                
                self.hover_polygon_index = -1
                self.hover_point_index = -1
                
                self.polygon_just_completed = True
                self.root.after(100, self._reset_polygon_completion_flag) 
                
                self.display_annotations() 
                self.save_history() 
            else: 
                self.cancel_current_polygon()
       
    def _reset_polygon_completion_flag(self):
        self.polygon_just_completed = False
    
    def navigate_image(self, direction):
        if not self.image_files:
            return
        # prevent moving past the first or last image
        if self.current_image_index == 0 and direction < 0:
            return
        if self.current_image_index == len(self.image_files) - 1 and direction > 0:
            return
        self.save_history()
        self.current_image_index += direction
        if self.current_image_index < 0:
            self.current_image_index = 0
        elif self.current_image_index >= len(self.image_files):
            self.current_image_index = len(self.image_files) - 1
        
        image_path = os.path.join(self.folder_path, self.image_files[self.current_image_index])
        self.load_image(image_path)
        
        relative_image_path = self.image_files[self.current_image_index]
        # Ensure the image node is loaded in the tree (expand parent folders if needed)
        if not self.image_tree.exists(relative_image_path):
            parent = os.path.dirname(relative_image_path)
            if parent:
                parts = parent.split(os.sep)
                acc = ''
                for p in parts:
                    acc = p if not acc else os.path.join(acc, p)
                    folder_id = f'folder_{acc}'
                    if self.image_tree.exists(folder_id):
                        self.image_tree.item(folder_id, open=True)
                        self.on_folder_expand(None, folder_id)
        # Select and scroll to the image if present
        if self.image_tree.exists(relative_image_path):
            try:
                self.image_tree.selection_set(relative_image_path)
                self.image_tree.focus(relative_image_path)
                self.image_tree.see(relative_image_path)
            except tk.TclError:
                pass

    # --------------------------------------------------
    # Copy/Paste Features
    # --------------------------------------------------

    def copy_bbox(self, bbox): self.copied_bbox_list.append(bbox); self.update_copied_bbox_display()

    def paste_all_bboxes(self):
        if self.copied_bbox_list:
            for bbox in self.copied_bbox_list: self.bboxes.append(bbox)
            self.display_annotations(); self.save_history()
        else: messagebox.showinfo("Info", "No bounding boxes copied to paste.")

    def update_copied_bbox_display(self):
        for widget in self.copy_frame.winfo_children(): widget.destroy()
        if not self.copied_bbox_list: tk.Label(self.copy_frame, text="Copied Bounding Boxes: None", font=("Arial", 12)).pack(pady=10)
        else:
            for bbox in self.copied_bbox_list:
                x, y, w, h, class_id = bbox
                label_text = f"Class {self.class_names[class_id]}, ({x}, {y}), ({w}, {h})"
                tk.Label(self.copy_frame, text=label_text, font=("Arial", 12)).pack(pady=5)

    # --------------------------------------------------
    # Save / Delete Image
    # --------------------------------------------------

    def delete_image(self):
        if self.current_image_index == -1: messagebox.showwarning("Warning", "No image selected to delete."); return
        if self.current_image_index == -1: messagebox.showwarning("Warning", "Cannot delete manually loaded image."); return # This condition is redundant
        relative_image_path = self.image_files[self.current_image_index]
        image_path = os.path.join(self.folder_path, relative_image_path)
        label_relative_path = os.path.splitext(relative_image_path)[0] + '.txt'
        label_path = os.path.join(self.label_folder, label_relative_path)
        if not messagebox.showyesno("Confirm Delete", f"Delete {relative_image_path} and its label?"): return
        try:
            os.remove(image_path)
            if os.path.exists(label_path): os.remove(label_path)
        except Exception as e: messagebox.showerror("Error", f"Error deleting files: {e}"); return
        del self.image_files[self.current_image_index]
        self.image_tree.delete(relative_image_path)
        if relative_image_path in self.image_status: del self.image_status[relative_image_path]
        self.canvas.delete("all"); self.image_name_label.config(text="")
        self.bboxes = []; self.polygons = []
        for widget in self.bbox_info_frame.winfo_children(): widget.destroy()
        self.bbox_info_frame.destroy(); self.bbox_info_frame = tk.Frame(self.info_canvas)
        self.info_canvas.create_window((0, 0), window=self.bbox_info_frame, anchor="nw")
        self.info_canvas.configure(scrollregion=self.info_canvas.bbox("all"))
        if self.image_files:
            self.current_image_index = min(self.current_image_index, len(self.image_files) - 1)
            if self.current_image_index >=0: # Ensure index is valid before loading
                self.load_image(os.path.join(self.folder_path, self.image_files[self.current_image_index]))
            else: # No images left or index became invalid
                self.current_image_index = -1 
                self.image = None; self.original_image = None; self.image_path = None
                self.display_image() # Clear canvas
        else: 
            self.current_image_index = -1
            self.image = None; self.original_image = None; self.image_path = None
            self.display_image() # Clear canvas
        self.update_status_labels(); self.save_history()

    # --------------------------------------------------
    # Class/Editor Utilities
    # --------------------------------------------------

    def delete_annotation(self, index, annotation_type):
        if annotation_type == 'bbox':
            if 0 <= index < len(self.bboxes):
                del self.bboxes[index]
        elif annotation_type == 'polygon':
            if 0 <= index < len(self.polygons):
                del self.polygons[index]
        
        self.display_annotations()
        self.save_history()
        if self.image_path and self.current_image_index != -1:
            relative_image_path = os.path.relpath(self.image_path, self.folder_path)
            new_status = "edited" if (self.bboxes or self.polygons) else "viewed"
            self.image_status[relative_image_path] = new_status
            self.image_tree.item(relative_image_path, tags=(new_status,))
            self.save_statuses()
            self.update_status_labels()

    def clear_class_selection(self):
        self.class_listbox.selection_clear(0, tk.END); self.selected_class_index = None
        self.copied_bbox_list = []; self.update_copied_bbox_display(); self.root.focus_set()

    def on_class_select(self, event):
        selection = self.class_listbox.curselection()
        if selection:
            self.selected_class_index = selection[0]
        else:
            self.selected_class_index = None

    def on_key_press(self, event):
        if event.char.isdigit():
            idx = int(event.char) - 1
            if 0 <= idx < len(self.class_names):
                self.class_listbox.selection_clear(0, tk.END)
                self.class_listbox.selection_set(idx)
                self.selected_class_index = idx

    # --------------------------------------------------
    # YOLO Model Loading / Auto-Annotation (related methods)
    # --------------------------------------------------
    def cancel_annotation(self):
        if self.cancel_event: self.cancel_event.set()
        if hasattr(self, 'progress_win') and self.progress_win.winfo_exists(): self.progress_win.destroy()
        self.auto_annotate_button.config(state=tk.NORMAL)

    def update_progress(self, percent, current, total):
        if hasattr(self, 'progress_win') and self.progress_win.winfo_exists():
            self.progress_var.set(percent)
            if hasattr(self, 'progress_label') and self.progress_label.winfo_exists():
                self.progress_label.config(text=f"{current}/{total} images processed")
            self.progress_win.update_idletasks()      
    def auto_annotate_dataset(self):
        """Auto-annotate dataset based on configuration from dialog."""
        # Initialize debug log file (overwrite each time)
        debug_log_path = os.path.join(os.path.dirname(__file__), 'debug_auto_annotation.log')
        
        def debug_log(message):
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")
            print(message)  # Also print to console
        
        # Clear previous log
        with open(debug_log_path, 'w', encoding='utf-8') as f:
            f.write("=== AUTO-ANNOTATION DEBUG LOG ===\n")
        
        # Get configuration from the stored config
        config = getattr(self, 'annotation_config', {})
        annotation_type = config.get('annotation_type', 'bounding_boxes')
        debug_log(f"DEBUG AUTO-ANNOTATE: annotation_type = '{annotation_type}'")
        debug_log(f"DEBUG AUTO-ANNOTATE: config = {config}")
        conf_threshold = config.get('confidence_threshold', self.confidence_threshold.get())
        selected_files = config.get('selected_files', list(self.image_files))
        
        flagged_images = []
        processed_count = 0
        total_images = len(selected_files)
        
        try:
            for idx, image_file in enumerate(selected_files):
                processed_count = idx + 1
                if self.cancel_event and self.cancel_event.is_set(): 
                    break
                
                image_path = os.path.join(self.folder_path, image_file)
                label_filename = os.path.splitext(image_file)[0] + '.txt'
                label_path = os.path.join(self.label_folder, label_filename)
                
                # Run model inference
                results = self.model(image_path, conf=conf_threshold, verbose=False)
                relative_image_path = image_file
                
                # Process results based on annotation type
                if annotation_type == "segmentation":
                    success = self._process_segmentation_results(results, label_path, image_file, conf_threshold)
                elif annotation_type == "both":
                    success = self._process_both_results(results, label_path, image_file, conf_threshold)
                else:  # bounding_boxes
                    success = self._process_detection_results(results, label_path, image_file, conf_threshold)
                
                # Update image status
                if success.get('has_annotations'):
                    if success.get('uncertain'):
                        flagged_images.append(relative_image_path)
                        self.image_status[relative_image_path] = "review_needed"
                    else:
                        self.image_status[relative_image_path] = "edited"
                else:
                    self.image_status[relative_image_path] = "viewed"
                
                # Update progress
                progress_percent = (processed_count / total_images) * 100
                self.root.after(0, self.update_progress, progress_percent, processed_count, total_images)
                
                if self.cancel_event.is_set(): 
                    break
                    
        except Exception as e:
            error_message = str(e)
            self.root.after(0, lambda err_msg=error_message: messagebox.showerror("Error", f"Annotation failed: {err_msg}"))
        finally:
            self.save_statuses()
            self.root.after(0, self.update_status_labels)
              # Update image tree tags
            for relative_image_path in selected_files:
                if self.image_tree.exists(relative_image_path):
                    status = self.image_status.get(relative_image_path, "not_viewed")
                    self.image_tree.item(relative_image_path, tags=(status,))
            
            if hasattr(self, 'progress_win') and self.progress_win.winfo_exists(): 
                self.root.after(0, self.progress_win.destroy)
            self.root.after(0, lambda: self.auto_annotate_button.config(state=tk.NORMAL))
            
            # Refresh the current image if it was part of the auto-annotation
            if self.image_path and selected_files:
                current_relative_path = os.path.relpath(self.image_path, self.folder_path)
                if current_relative_path in selected_files:
                    self.root.after(0, lambda: self.load_image(self.image_path))
            
            # Show completion message
            if self.cancel_event and self.cancel_event.is_set():
                self.root.after(0, lambda: messagebox.showinfo("Cancelled", f"Annotation cancelled. Processed {processed_count}/{total_images} images."))
            elif flagged_images:
                self.root.after(0, lambda: messagebox.showwarning("Review Needed", f"{len(flagged_images)} images have low-confidence detections requiring review."))
            else:
                self.root.after(0, lambda: messagebox.showinfo("Complete", "Auto-annotation finished successfully!"))
    
    def _process_detection_results(self, results, label_path, image_file, conf_threshold):
        """Process detection results for bounding box annotations."""
        detections = results[0].boxes
        bboxes = []
        uncertain = False
        img_h, img_w = None, None
        
        if not detections:
            return {"has_annotations": False, "uncertain": False}
        
        for box in detections:
            if self.cancel_event and self.cancel_event.is_set(): 
                break
                
            conf_score = box.conf[0].item()
            class_id = int(box.cls[0])
            
            if img_h is None or img_w is None: 
                img_h, img_w = results[0].orig_shape[:2]
            
            if class_id >= len(self.class_names): 
                continue
            
            np_module = lazy_importer.get_numpy()
            x_center, y_center, width, height = box.xywhn[0].cpu().numpy()
            x_center_abs = x_center * img_w
            y_center_abs = y_center * img_h
            width_abs = width * img_w
            height_abs = height * img_h
            x_min = int(x_center_abs - width_abs / 2)
            y_min = int(y_center_abs - height_abs / 2)
            
            bboxes.append((x_min, y_min, int(width_abs), int(height_abs), class_id, conf_score))
            
            if conf_score < conf_threshold * 1.2:  # Mark uncertain if close to threshold
                uncertain = True
        
        if bboxes:
            # Create directory structure for label file if it doesn't exist
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            existing_bboxes, existing_polygons = read_annotations_from_file(label_path, (img_h, img_w))
            new_bboxes_for_file = [(x, y, w, h, cid) for (x, y, w, h, cid, _) in bboxes]
            write_annotations_to_file(label_path, new_bboxes_for_file, existing_polygons, (img_h, img_w))
            return {"has_annotations": True, "uncertain": uncertain}
        
        return {"has_annotations": False, "uncertain": False}
    
    def _process_segmentation_results(self, results, label_path, image_file, conf_threshold):
        """Process segmentation results for polygon annotations."""
        # Debug log setup
        debug_log_path = os.path.join(os.path.dirname(__file__), 'debug_auto_annotation.log')
        
        def debug_log(message):
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")
            print(message)  # Also print to console
        
        debug_log(f"DEBUG SEGMENTATION: Processing {image_file}")
        debug_log(f"DEBUG SEGMENTATION: results[0] type: {type(results[0])}")
        debug_log(f"DEBUG SEGMENTATION: hasattr(results[0], 'masks'): {hasattr(results[0], 'masks')}")
        if hasattr(results[0], 'masks'):
            debug_log(f"DEBUG SEGMENTATION: results[0].masks: {results[0].masks}")
            debug_log(f"DEBUG SEGMENTATION: results[0].masks is None: {results[0].masks is None}")
        
        # Check if model supports segmentation
        if not hasattr(results[0], 'masks') or results[0].masks is None:
            # Fall back to bounding boxes if no masks available
            return self._process_detection_results(results, label_path, image_file, conf_threshold)
        
        masks = results[0].masks
        detections = results[0].boxes
        polygons = []
        uncertain = False
        img_h, img_w = None, None
        
        if not detections:
            debug_log(f"DEBUG SEGMENTATION: No detections found")
            return {"has_annotations": False, "uncertain": False}
        
        debug_log(f"DEBUG SEGMENTATION: Starting polygon conversion for {len(detections)} detections")
        
        # YOLOv8 provides polygon coordinates directly in masks.xy
        if hasattr(masks, 'xy') and masks.xy is not None:
            debug_log(f"DEBUG SEGMENTATION: Using pre-computed polygon coordinates from masks.xy")
            debug_log(f"DEBUG SEGMENTATION: masks.xy has {len(masks.xy)} polygon sets")
            
            for i, (box, polygon_coords) in enumerate(zip(detections, masks.xy)):
                if self.cancel_event and self.cancel_event.is_set(): 
                    break
                    
                conf_score = box.conf[0].item()
                class_id = int(box.cls[0])
                
                debug_log(f"DEBUG SEGMENTATION: Processing detection {i}: class_id={class_id}, conf_score={conf_score}")
                
                if img_h is None or img_w is None: 
                    img_h, img_w = results[0].orig_shape[:2]
                
                if class_id >= len(self.class_names): 
                    debug_log(f"DEBUG SEGMENTATION: Skipping invalid class_id {class_id} (max: {len(self.class_names)-1})")
                    continue
                
                # Convert numpy array to list of (x, y) tuples
                try:
                    debug_log(f"DEBUG SEGMENTATION: Converting polygon coordinates for detection {i}")
                    debug_log(f"DEBUG SEGMENTATION: polygon_coords shape: {polygon_coords.shape}")
                    debug_log(f"DEBUG SEGMENTATION: First few coordinates: {polygon_coords[:5] if len(polygon_coords) > 5 else polygon_coords}")
                    
                    # Convert to integer pixel coordinates
                    polygon_points = []
                    for coord_pair in polygon_coords:
                        x, y = coord_pair
                        polygon_points.append((int(x), int(y)))
                    
                    if len(polygon_points) >= 3:  # Valid polygon needs at least 3 points
                        debug_log(f"DEBUG SEGMENTATION: Created polygon with {len(polygon_points)} points: {polygon_points[:5]}...")
                        polygons.append((polygon_points, class_id, conf_score))
                        
                        if conf_score < conf_threshold * 1.2:  # Mark uncertain if close to threshold
                            uncertain = True
                    else:
                        debug_log(f"DEBUG SEGMENTATION: Polygon has too few points ({len(polygon_points)}), skipping")
                        
                except Exception as e:
                    debug_log(f"DEBUG SEGMENTATION: Exception during polygon conversion for detection {i}: {e}")
                    continue
        else:
            debug_log(f"DEBUG SEGMENTATION: masks.xy not available, falling back to bounding boxes")
            return self._process_detection_results(results, label_path, image_file, conf_threshold)
        
        debug_log(f"DEBUG SEGMENTATION: Generated {len(polygons)} polygons total")
        
        if polygons:
            # Create directory structure for label file if it doesn't exist
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            existing_bboxes, existing_polygons = read_annotations_from_file(label_path, (img_h, img_w))
            
            # Convert polygons to the correct format for write_annotations_to_file
            new_polygons_for_file = []
            for (points, cid, _) in polygons:
                new_polygons_for_file.append({
                    'class_id': cid,
                    'points': points
                })
            
            debug_log(f"DEBUG SEGMENTATION: Writing {len(new_polygons_for_file)} polygons to label file")
            write_annotations_to_file(label_path, existing_bboxes, new_polygons_for_file, (img_h, img_w))
            return {"has_annotations": True, "uncertain": uncertain}
        
        return {"has_annotations": False, "uncertain": False}
    
    def _process_both_results(self, results, label_path, image_file, conf_threshold):
        """Process results for both bounding boxes and segmentation."""
        # Process bounding boxes first
        bbox_result = self._process_detection_results(results, label_path, image_file, conf_threshold)
        
        # Then process segmentation if available
        seg_result = self._process_segmentation_results(results, label_path, image_file, conf_threshold)
        
        # Combine results
        has_annotations = bbox_result.get("has_annotations", False) or seg_result.get("has_annotations", False)
        uncertain = bbox_result.get("uncertain", False) or seg_result.get("uncertain", False)
        
        return {"has_annotations": has_annotations, "uncertain": uncertain}

    # --------------------------------------------------
    # YOLO Training Functionality (related methods)
    # --------------------------------------------------

    def reload_classes_from_yaml(self):
        try:
            with open(self.yaml_path, "r") as f: data = yaml.safe_load(f)
            raw_names = data.get("names", ["person"])
            self.class_names = [raw_names[k] for k in sorted(raw_names.keys(), key=lambda x:int(x))] if isinstance(raw_names,dict) else raw_names
            self.class_listbox.delete(0, tk.END)
            for class_name in self.class_names: self.class_listbox.insert(tk.END, class_name)
            self.update_class_colors(); self.display_annotations()
            messagebox.showinfo("Classes Reloaded", f"Successfully reloaded {len(self.class_names)} classes from YAML file.")
        except Exception as e: messagebox.showerror("Error", f"Failed to reload classes from YAML: {str(e)}")

    def open_training_dialog(self):
        train_win = tk.Toplevel(self.root)
        train_win.title("Standard Training")
        train_win.transient(self.root)
        train_win.grab_set()
        train_win.geometry("700x900") 
        
        model_frame = tk.LabelFrame(train_win, text="🎯 Standard Training Setup")
        model_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(model_frame, text="Initial Weights:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        model_var = tk.StringVar(value="yolov8n.pt")
        model_options = [
            "None (random init)",
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt"
        ]
        model_combo = ttk.Combobox(model_frame, textvariable=model_var, values=model_options, state="readonly")
        model_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=2)

        params_frame = tk.LabelFrame(train_win, text="Training Parameters")
        params_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        epochs_var = tk.StringVar(value="100")
        tk.Entry(params_frame, textvariable=epochs_var, width=10).grid(row=0, column=1, sticky="w", padx=5, pady=2)

        tk.Label(params_frame, text="Image Size:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        imgsz_var = tk.StringVar(value="640")
        tk.Entry(params_frame, textvariable=imgsz_var, width=10).grid(row=0, column=3, sticky="w", padx=5, pady=2)

        tk.Label(params_frame, text="Batch Size:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        batch_var = tk.StringVar(value="16")
        tk.Entry(params_frame, textvariable=batch_var, width=10).grid(row=1, column=1, sticky="w", padx=5, pady=2)

        tk.Label(params_frame, text="Learning Rate:").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        lr_var = tk.StringVar(value="0.01")
        tk.Entry(params_frame, textvariable=lr_var, width=10).grid(row=1, column=3, sticky="w", padx=5, pady=2)
        
        device_frame = tk.LabelFrame(train_win, text="⚡ Device Selection")
        device_frame.pack(fill=tk.X, padx=10, pady=5)
        
        def detect_available_devices():
            devices = ["cpu"]
            gpu_info = ""
            try:
                import torch
                if torch.cuda.is_available():
                    count = torch.cuda.device_count()
                    for i in range(count):
                        devices.append(f"cuda:{i}")
                        name = torch.cuda.get_device_name(i)
                        mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        gpu_info += f"GPU {i}: {name} ({mem:.1f} GB)\n"
                    if count > 0:
                        devices.append("cuda")
                else:
                    gpu_info = "No CUDA-compatible GPU detected"
            except ImportError:
                gpu_info = "PyTorch not available for GPU detection"
            except Exception as e:
                gpu_info = f"GPU detection failed: {e}"
            return devices, gpu_info

        tk.Label(device_frame, text="Training Device:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        device_var = tk.StringVar(value="Detecting...")
        device_combo = ttk.Combobox(
            device_frame,
            textvariable=device_var,
            values=["Detecting..."],
            state="disabled",
            width=15
        )
        device_combo.grid(row=0, column=1, sticky="w", padx=5, pady=2)

        gpu_info_label = tk.Label(
            device_frame,
            text="Detecting GPU devices...",
            font=("TkDefaultFont", 8),
            fg="gray50",
            justify="left"
        )
        gpu_info_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        def _detect_devices_worker():
            devs, info = detect_available_devices()
            def _on_done():
                default = "cpu"
                if "cuda" in devs:
                    default = "cuda"
                else:
                    for d in devs:
                        if d.startswith("cuda"):
                            default = d
                            break
                device_combo.config(values=devs, state="readonly")
                device_var.set(default)
                gpu_info_label.config(text=info)
            train_win.after(0, _on_done)

        threading.Thread(target=_detect_devices_worker, daemon=True).start()
        
        device_tips_frame = tk.Frame(device_frame)
        device_tips_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        tk.Label(device_tips_frame, text="💡 Device Tips:", 
                font=("TkDefaultFont", 9, "bold"), fg="darkorange").pack(anchor="w")
        tk.Label(device_tips_frame, text="   • CPU: Stable but slower, good for small datasets", 
                font=("TkDefaultFont", 8), fg="darkorange").pack(anchor="w")
        tk.Label(device_tips_frame, text="   • GPU: Faster training, requires sufficient VRAM (8GB+ recommended)", 
                font=("TkDefaultFont", 8), fg="darkorange").pack(anchor="w")
        tk.Label(device_tips_frame, text="   • If training crashes, try CPU or reduce batch size", 
                font=("TkDefaultFont", 8), fg="darkorange").pack(anchor="w")
        
        data_frame = tk.LabelFrame(train_win, text="Data Configuration")
        data_frame.pack(fill=tk.X, padx=10, pady=5)
        
        auto_export_var = tk.BooleanVar(value=True)
        tk.Checkbutton(data_frame, text="Auto-export dataset for training", variable=auto_export_var).pack(anchor="w", padx=5, pady=2)
        
        split_var = tk.StringVar(value="split")
        tk.Radiobutton(data_frame, text="Split data (80/20 train/val)", variable=split_var, value="split").pack(anchor="w", padx=5, pady=2)
        tk.Radiobutton(data_frame, text="Use existing train/val split", variable=split_var, value="existing").pack(anchor="w", padx=5, pady=2)
        tk.Radiobutton(data_frame, text="Train only (no validation)", variable=split_var, value="train_only").pack(anchor="w", padx=5, pady=2)
        
        output_frame = tk.LabelFrame(train_win, text="Output Location")
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        
        output_var = tk.StringVar(value=os.path.join(self.folder_path, "runs", "train"))
        tk.Label(output_frame, text="Output Directory:").pack(anchor="w", padx=5, pady=2)
        output_entry = tk.Entry(output_frame, textvariable=output_var, width=60)
        output_entry.pack(fill=tk.X, padx=5, pady=2)
        
        progress_frame = tk.LabelFrame(train_win, text="Training Progress")
        progress_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.train_progress = tk.Text(progress_frame, height=8, state=tk.DISABLED)
        scrollbar = tk.Scrollbar(progress_frame, orient="vertical", command=self.train_progress.yview)
        self.train_progress.configure(yscrollcommand=scrollbar.set)
        self.train_progress.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        button_frame = tk.Frame(train_win)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def start_training():
            start_btn.config(state=tk.DISABLED)

            # Determine selected model
            model_val = model_var.get()
            if not model_val or model_val.startswith("🎯") or model_val.startswith("None"):
                messagebox.showerror(
                    "Invalid Selection",
                    "Please select a valid pretrained model or 'None' for random init.",
                    parent=train_win
                )
                start_btn.config(state=tk.NORMAL)
                return

            epochs = int(epochs_var.get())
            imgsz = int(imgsz_var.get())
            batch = int(batch_var.get())
            lr = float(lr_var.get())
            output_dir = output_var.get()
            device = device_var.get()

            self.training_stop_flag = threading.Event()
            stop_btn.config(state=tk.NORMAL)

            active = False
            training_thread = threading.Thread(
                target=self.execute_training,
                args=(
                    model_val,
                    epochs,
                    imgsz,
                    batch,
                    lr,
                    output_dir,
                    auto_export_var.get(),
                    split_var.get(),
                    start_btn,
                    train_win,
                    device,
                    active,
                    self.training_stop_flag,
                )
            )
            training_thread.start()
            self.current_training_thread = training_thread
        
        def safe_cancel():
            if hasattr(self, 'training_stop_flag'):
                self.training_stop_flag.set()
            if hasattr(self, 'current_training_thread') and self.current_training_thread.is_alive():
                time.sleep(0.5)
            train_win.destroy()
        
        start_btn = tk.Button(button_frame, text="🚀 Start Training", command=start_training)
        start_btn.pack(side=tk.LEFT, padx=5)

        stop_btn = tk.Button(button_frame, text="🛑 Stop Training", state=tk.DISABLED,
                             command=lambda: (self.training_stop_flag.set(), stop_btn.config(state=tk.DISABLED)))
        stop_btn.pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Cancel", command=safe_cancel).pack(side=tk.RIGHT, expand=True, padx=5)

    def bbox_to_polygon(self, x_center, y_center, width, height):
        x1 = x_center - width/2
        y1 = y_center - height/2
        x2 = x_center + width/2
        y2 = y_center + height/2
        return [x1, y1, x2, y1, x2, y2, x1, y2]

    def convert_label_file_to_segmentation(self, input_file, output_file):
        try:
            with open(input_file, 'r') as f:
                lines = f.readlines()
            
            converted_lines = []
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5: 
                    class_id = parts[0]
                    x_center, y_center, width, height = map(float, parts[1:5])
                    polygon_coords = self.bbox_to_polygon(x_center, y_center, width, height)
                    converted_line = f"{class_id} " + " ".join(map(str, polygon_coords)) + "\n"
                    converted_lines.append(converted_line)
                elif len(parts) > 5 and len(parts) % 2 == 1: 
                    converted_lines.append(line)
                else:
                    logging.warning(f"Skipping invalid annotation in {input_file}: {line.strip()}")
            
            with open(output_file, 'w') as f:
                f.writelines(converted_lines)
            return True
        except Exception as e:
            logging.error(f"Failed to convert label file {input_file}: {e}")
            return False

    def convert_dataset_to_segmentation(self, log_callback=None):
        def log_msg(msg):
            if log_callback:
                log_callback(msg)
            else:
                logging.info(msg)
        
        try:
            from pathlib import Path
            
            source_dir = Path("yolo_prepared_dataset")
            output_dir = Path("yolo_prepared_dataset_segmentation")
            
            if not source_dir.exists():
                log_msg("❌ Error: yolo_prepared_dataset directory not found!")
                return False
            
            output_dir.mkdir(exist_ok=True)
            (output_dir / "images").mkdir(exist_ok=True)
            (output_dir / "labels").mkdir(exist_ok=True)
            (output_dir / "images" / "train").mkdir(exist_ok=True)
            (output_dir / "images" / "val").mkdir(exist_ok=True)
            (output_dir / "labels" / "train").mkdir(exist_ok=True)
            (output_dir / "labels" / "val").mkdir(exist_ok=True)
            
            log_msg("📁 Copying images...")
            for split in ["train", "val"]:
                source_img_dir = source_dir / "images" / split
                output_img_dir = output_dir / "images" / split
                
                if source_img_dir.exists():
                    for img_file in source_img_dir.glob("*"):
                        if img_file.is_file():
                            shutil.copy2(img_file, output_img_dir / img_file.name)
            
            log_msg("🔄 Converting labels...")
            converted_count = 0
            for split in ["train", "val"]:
                source_label_dir = source_dir / "labels" / split
                output_label_dir = output_dir / "labels" / split
                
                if source_label_dir.exists():
                    for label_file in source_label_dir.glob("*.txt"):
                        if self.convert_label_file_to_segmentation(label_file, output_label_dir / label_file.name):
                            converted_count += 1
                        else:
                            log_msg(f"⚠️ Failed to convert {label_file.name}")
            
            dataset_yaml_path = output_dir / "dataset.yaml" # Renamed
            yaml_content = f"""# YOLO segmentation dataset configuration
path: {output_dir.absolute().as_posix()}
train: images/train
val: images/val
nc: {len(self.class_names)}
names: {self.class_names}
""" # Used self.class_names
            
            with open(dataset_yaml_path, 'w') as f:
                f.write(yaml_content)
            
            log_msg("✅ Conversion completed!")
            log_msg(f"📊 Output directory: {output_dir.absolute()}")
            log_msg(f"📝 Converted {converted_count} label files")
            log_msg(f"📄 Dataset YAML: {dataset_yaml_path.absolute()}")
            
            return True
            
        except Exception as e:
            log_msg(f"❌ Failed to convert dataset: {e}")
            logging.error(f"Dataset conversion error: {e}", exc_info=True)
            return False

    def _export_yaml_logic(self, split_type):
        prepared_dataset_root = os.path.join(os.getcwd(), "yolo_prepared_dataset") 
        
        if os.path.exists(prepared_dataset_root):
            shutil.rmtree(prepared_dataset_root)
        os.makedirs(prepared_dataset_root, exist_ok=True)

        train_images_dir = os.path.join(prepared_dataset_root, "images", "train")
        val_images_dir = os.path.join(prepared_dataset_root, "images", "val")
        train_labels_dir = os.path.join(prepared_dataset_root, "labels", "train")
        val_labels_dir = os.path.join(prepared_dataset_root, "labels", "val")

        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(val_images_dir, exist_ok=True)
        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(val_labels_dir, exist_ok=True)

        dataset_yaml_path_local = os.path.join(prepared_dataset_root, "dataset.yaml") # Renamed
        
        all_image_files_in_project = list(self.image_files) 
        
        labeled_image_files_relative_to_original_dataset = []
        for relative_image_path in all_image_files_in_project:
            original_label_filename_part = os.path.splitext(relative_image_path)[0]
            if original_label_filename_part.startswith(os.path.sep) or original_label_filename_part.startswith('/'):
                 original_label_filename_part = original_label_filename_part[1:]
            original_label_file_path = os.path.join(self.label_folder, original_label_filename_part + ".txt")

            if os.path.exists(original_label_file_path) and os.path.getsize(original_label_file_path) > 0:
                labeled_image_files_relative_to_original_dataset.append(relative_image_path)
            else:
                logging.info(f"Skipping image {relative_image_path} for training YAML as its label file is missing or empty at {original_label_file_path}")

        if not labeled_image_files_relative_to_original_dataset:
            logging.error("No labeled images with non-empty label files found to create dataset.yaml for training.")
            messagebox.showerror("Dataset YAML Error", "No labeled images found for training. Ensure images are annotated and labels saved.", parent=self.root)
            return None

        source_train_files_rel = []
        source_val_files_rel = []

        if split_type == "split":
            import random
            random.shuffle(labeled_image_files_relative_to_original_dataset)
            split_idx = int(len(labeled_image_files_relative_to_original_dataset) * 0.8)
            source_train_files_rel = labeled_image_files_relative_to_original_dataset[:split_idx]
            source_val_files_rel = labeled_image_files_relative_to_original_dataset[split_idx:]
            
            if not source_train_files_rel and labeled_image_files_relative_to_original_dataset:
                source_train_files_rel = labeled_image_files_relative_to_original_dataset
                source_val_files_rel = []
                logging.warning("Too few labeled images for a train/val split. Using all for training.")
        elif split_type == "existing":
            logging.warning("Split type 'existing' with auto_export. This logic assumes 'train' and 'val' subdirectories exist with images in the original dataset.")
            for rel_path in labeled_image_files_relative_to_original_dataset:
                if rel_path.startswith("train" + os.path.sep) or rel_path.startswith("train/"):
                    source_train_files_rel.append(rel_path)
                elif rel_path.startswith("valid" + os.path.sep) or rel_path.startswith("valid/") or \
                     rel_path.startswith("val" + os.path.sep) or rel_path.startswith("val/"):
                    source_val_files_rel.append(rel_path)
            if not source_train_files_rel:
                logging.warning("No labeled images found in 'train' subdirectory for 'existing' split. Using all labeled images for training.")
                source_train_files_rel = labeled_image_files_relative_to_original_dataset
                source_val_files_rel = []
        elif split_type == "train_only":
            source_train_files_rel = labeled_image_files_relative_to_original_dataset
            source_val_files_rel = []
            # For train-only split, use training images also as validation to avoid empty val set
            source_val_files_rel = source_train_files_rel.copy()
            logging.warning("Train-only split: using all labeled images for validation as well.")
        else:
            logging.error(f"Unknown split_type: {split_type}")
            messagebox.showerror("Dataset YAML Error", f"Unknown split type: {split_type}", parent=self.root)
            return None
            
        if not source_train_files_rel and not source_val_files_rel and labeled_image_files_relative_to_original_dataset:
            logging.warning("Train/Val split resulted in empty lists, using all labeled images for training as fallback.")
            source_train_files_rel = labeled_image_files_relative_to_original_dataset
            source_val_files_rel = []

        def copy_and_get_relative_paths(source_files_relative_to_original, dest_image_dir, dest_label_dir):
            yaml_image_paths = []
            for original_rel_img_path in source_files_relative_to_original:
                original_abs_img_path = os.path.join(self.folder_path, original_rel_img_path)
                
                original_label_filename_part = os.path.splitext(original_rel_img_path)[0]
                if original_label_filename_part.startswith(os.path.sep) or original_label_filename_part.startswith('/'):
                    original_label_filename_part = original_label_filename_part[1:]
                original_abs_label_path = os.path.join(self.label_folder, original_label_filename_part + ".txt")

                img_basename = os.path.basename(original_abs_img_path)
                label_basename = os.path.basename(original_abs_label_path)

                dest_abs_img_path = os.path.join(dest_image_dir, img_basename)
                dest_abs_label_path = os.path.join(dest_label_dir, label_basename)

                try:
                    shutil.copy2(original_abs_img_path, dest_abs_img_path)
                    shutil.copy2(original_abs_label_path, dest_abs_label_path)
                    yaml_image_paths.append(os.path.join("images", os.path.basename(dest_image_dir), img_basename).replace("\\", "/"))
                except Exception as e:
                    logging.error(f"Error copying file {original_abs_img_path} or {original_abs_label_path}: {e}")
            return yaml_image_paths

        yaml_train_image_paths = copy_and_get_relative_paths(source_train_files_rel, train_images_dir, train_labels_dir)
        yaml_val_image_paths = copy_and_get_relative_paths(source_val_files_rel, val_images_dir, val_labels_dir)

        yaml_data = {
            'path': prepared_dataset_root.replace("\\", "/"), 
            'train': 'images/train', 
            'val': 'images/val',     
            'nc': len(self.class_names),
            'names': self.class_names
        }
        logging.info(f"Generated dataset.yaml with folder paths in {prepared_dataset_root}.")

        try:
            with open(dataset_yaml_path_local, 'w') as f:
                yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=None, width=float("inf"))
            logging.info(f"Generated dataset.yaml at {dataset_yaml_path_local}")
            return dataset_yaml_path_local 
        except Exception as e:
            logging.error(f"Failed to write dataset.yaml: {e}", exc_info=True)
            messagebox.showerror("Dataset YAML Error", f"Failed to write dataset.yaml:\n{e}", parent=self.root)
            return None

    def execute_training(self, model_name_arg, epochs, imgsz, batch, lr, output_dir, auto_export, split_type, start_btn, train_win, device, active=False, stop_flag=None):
        def log_message(msg):
            try:
                def update_gui():
                    try:
                        self.train_progress.config(state=tk.NORMAL)
                        self.train_progress.insert(tk.END, f"{msg}\n")
                        self.train_progress.see(tk.END)
                        self.train_progress.config(state=tk.DISABLED)
                    except:
                        pass 
                train_win.after(0, update_gui)
            except:
                print(f"Log: {msg}") 
        
        try:
            if stop_flag and stop_flag.is_set():
                log_message("🛑 Training cancelled by user")
                return
            
            log_message("🚀 Starting YOLO training...")
            log_message(f"Training Mode: {'Active Learning' if active else 'Standard Training'}")
            log_message(f"Model: {model_name_arg}")
            log_message(f"Epochs: {epochs}, Image Size: {imgsz}, Batch: {batch}, LR: {lr}")
            log_message(f"Device: {device}")
            
            try:
                import psutil
                memory = psutil.virtual_memory()
                available_gb = memory.available / (1024**3)
                log_message(f"💾 System check: {available_gb:.1f} GB RAM available")
                if available_gb < 2:
                    log_message("⚠️ WARNING: Low memory detected - adjusting parameters to prevent crashes")
                    batch = 1; imgsz = min(imgsz, 320)
                    log_message(f"🔧 Auto-adjusted: batch_size={batch}, imgsz={imgsz}")
                elif available_gb < 4:
                    log_message("⚠️ CAUTION: Limited memory - using conservative settings")
                    batch = min(batch, 2); imgsz = min(imgsz, 416)
                    log_message(f"🔧 Auto-adjusted: batch_size={batch}, imgsz={imgsz}")
                else:
                    log_message("✅ Good memory available for training")
            except ImportError: log_message("⚠️ Cannot check system resources (psutil not available)")
            except Exception as e_psutil: log_message(f"⚠️ System check failed: {e_psutil}") # Renamed
            
            dataset_yaml_local = None
            is_segmentation_model = ("-seg.pt" in model_name_arg.lower())
            if is_segmentation_model:
                log_message("🎯 SEGMENTATION MODE: Converting all annotations to polygons...")
            else:
                log_message("🎯 OBJECT DETECTION MODE: Using bounding boxes for detection...")
            
            if auto_export:
                log_message("📤 Exporting dataset for training...")
                if is_segmentation_model:
                    log_message("🔄 Step 1: Creating base dataset...")
                    dataset_yaml_local = self._export_yaml_logic(split_type)
                    if not dataset_yaml_local:
                        log_message("❌ Failed to prepare base dataset.yaml for training.")
                        messagebox.showerror("Training Error", "Failed to prepare dataset.yaml.", parent=train_win)
                        start_btn.config(state=tk.NORMAL); return
                    log_message("🔄 Step 2: Converting to pure segmentation format...")
                    if self.convert_dataset_to_segmentation(log_callback=log_message):
                        current_dir = os.getcwd()
                        segmentation_dataset_root = os.path.join(current_dir, "yolo_prepared_dataset_segmentation")
                        dataset_yaml_local = os.path.join(segmentation_dataset_root, "dataset.yaml")
                        log_message(f"📊 Using segmentation dataset: {dataset_yaml_local}")
                    else:
                        log_message("❌ Failed to convert dataset to segmentation format")
                        messagebox.showerror("Dataset Error", "Segmentation training failed: Dataset conversion error.\n\nTry detection mode instead.", parent=train_win)
                        start_btn.config(state=tk.NORMAL); return
                else:
                    dataset_yaml_local = self._export_yaml_logic(split_type) 
                    if not dataset_yaml_local:
                        log_message("❌ Failed to prepare dataset.yaml for training.")
                        messagebox.showerror("Training Error", "Failed to prepare dataset.yaml.", parent=train_win)
                        start_btn.config(state=tk.NORMAL); return
                    log_message(f"📊 Using detection dataset: {dataset_yaml_local}")
            else: # No auto-export
                current_dir = os.getcwd()
                if is_segmentation_model:
                    segmentation_dataset_root = os.path.join(current_dir, "yolo_prepared_dataset_segmentation")
                    segmentation_yaml = os.path.join(segmentation_dataset_root, "dataset.yaml")
                    if os.path.exists(segmentation_yaml):
                        dataset_yaml_local = segmentation_yaml
                        log_message(f"📊 Using existing segmentation dataset: {dataset_yaml_local}")
                    else:
                        log_message("⚠️ No segmentation dataset found. Auto-converting from detection dataset...")
                        detection_dataset_root = os.path.join(current_dir, "yolo_prepared_dataset")
                        if not os.path.exists(detection_dataset_root):
                            log_message("❌ No datasets found. Please enable auto-export or create datasets first.")
                            messagebox.showerror("Dataset Error", "No datasets found for segmentation training.\n\nSolutions:\n1. Enable 'Auto-export dataset' option\n2. Create datasets manually", parent=train_win)
                            start_btn.config(state=tk.NORMAL); return
                        if self.convert_dataset_to_segmentation(log_callback=log_message):
                            dataset_yaml_local = segmentation_yaml
                            log_message(f"📊 Using converted segmentation dataset: {dataset_yaml_local}")
                        else:
                            log_message("❌ Failed to convert dataset to segmentation format")
                            messagebox.showerror("Dataset Error", "Segmentation training failed: Dataset conversion error.\n\nTry detection mode instead.", parent=train_win)
                            start_btn.config(state=tk.NORMAL); return
                else: # Detection mode, no auto-export
                    prepared_dataset_root = os.path.join(current_dir, "yolo_prepared_dataset")
                    dataset_yaml_local = os.path.join(prepared_dataset_root, "dataset.yaml")
                    if os.path.exists(dataset_yaml_local):
                        log_message(f"📊 Using existing detection dataset: {dataset_yaml_local}")
                    else:
                        log_message("❌ No detection dataset found. Please enable auto-export.")
                        messagebox.showerror("Dataset Error", "No detection dataset found.\n\nPlease enable 'Auto-export dataset' option.", parent=train_win)
                        start_btn.config(state=tk.NORMAL); return
            
            log_message(f"🔍 Validating dataset format for {'segmentation' if is_segmentation_model else 'detection'} model...")
            labels_dir = os.path.join(os.path.dirname(dataset_yaml_local), "labels", "train")
            if os.path.exists(labels_dir):
                label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
                if label_files:
                    try:
                        has_detection = False; has_segmentation = False
                        total_detection_annotations = 0; total_segmentation_annotations = 0
                        files_to_check = label_files[:min(5, len(label_files))]
                        for file_name in files_to_check:
                            file_path = os.path.join(labels_dir, file_name)
                            try:
                                with open(file_path, 'r') as f_label_check: # Renamed
                                    file_lines = f_label_check.readlines()
                                for line in file_lines:
                                    parts = line.strip().split()
                                    if len(parts) == 5: has_detection = True; total_detection_annotations += 1
                                    elif len(parts) > 5 and len(parts) % 2 == 1: has_segmentation = True; total_segmentation_annotations += 1
                            except Exception: continue # Skip file on error
                        log_message(f"📋 Dataset analysis: {total_detection_annotations} bounding boxes, {total_segmentation_annotations} polygons")
                        if has_detection and has_segmentation:
                            if is_segmentation_model: log_message("✅ Mixed dataset with segmentation model: Will use polygons for segmentation, ignore bounding boxes.")
                            else: log_message("✅ Mixed dataset with detection model: Will convert polygons to bounding boxes automatically.")
                        elif is_segmentation_model and has_detection and not has_segmentation:
                            log_message("❌ Error: Segmentation model requires polygon annotations, but only bounding boxes found.")
                            messagebox.showerror("Dataset Format Error", "This segmentation model requires polygon annotations, but your dataset only contains bounding boxes.\n\nSolutions:\n1. Use a detection model (yolov8n.pt)\n2. Add polygon annotations to your dataset", parent=train_win)
                            start_btn.config(state=tk.NORMAL); return
                        elif not is_segmentation_model and has_segmentation and not has_detection:
                            log_message("✅ Detection model with polygon dataset: Will convert polygons to bounding boxes.")
                        else: log_message("✅ Dataset format matches model type perfectly.")
                    except Exception as e_validate: log_message(f"⚠️ Could not validate dataset format: {e_validate}") # Renamed
            
            try:
                YOLO = lazy_importer.get_yolo()
                log_message("🤖 Loading YOLO model...")
                import gc, torch
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
                os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'; os.environ['NUMEXPR_NUM_THREADS'] = '1'
                torch.set_num_threads(1)
                if hasattr(torch.cuda, 'empty_cache'): torch.cuda.empty_cache()
                gc.collect()
                
                model_instance = YOLO(model_name_arg) # Use renamed argument
                log_message("🏋️ Starting training...")
                log_message("🧠 Enhanced memory optimization applied")
                
                safe_batch = max(1, min(batch, 2)); safe_imgsz = min(imgsz, 320); safe_workers = 0
                log_message(f"🛡️ Safety parameters: batch={safe_batch}, imgsz={safe_imgsz}, workers={safe_workers}")
                
                if stop_flag and stop_flag.is_set():
                    log_message("🛑 Training cancelled before start")
                    start_btn.config(state=tk.NORMAL); return
                
                class TrainingCallback:
                    def __init__(self, stop_flag_cb, log_func_cb, total_epochs_cb): # Renamed
                        self.stop_flag = stop_flag_cb
                        self.log_func = log_func_cb
                        self.train_start_time = None
                        self.total_epochs = total_epochs_cb
                        self.epoch_count_for_avg = 0

                    def on_train_epoch_start(self, trainer):
                        if trainer.epoch == 0 and self.train_start_time is None:
                            self.train_start_time = time.time()
                            self.log_func(f"🚀 Training initiated for {self.total_epochs} epochs...")
                            self.log_func(f"   Device: {trainer.device}")

                    def on_train_epoch_end(self, trainer):
                        if self.stop_flag and self.stop_flag.is_set():
                            self.log_func("🛑 Training stopped by user.")
                            trainer.stop = True; return True

                        current_epoch_num = trainer.epoch + 1
                        self.epoch_count_for_avg = current_epoch_num

                        if self.train_start_time is None:
                            if current_epoch_num == 1:
                                self.train_start_time = time.time() - (trainer.times.get('epoch', 60.0))
                                self.log_func(f"🚀 Training started (approximated at end of epoch 1) for {self.total_epochs} epochs...")
                            else:
                                self.log_func(f"Epoch {current_epoch_num}/{self.total_epochs} completed. (ETA unavailable)")
                                return False

                        time_now = time.time(); time_elapsed_total_seconds = time_now - self.train_start_time
                        avg_time_per_epoch_seconds = time_elapsed_total_seconds / self.epoch_count_for_avg if self.epoch_count_for_avg > 0 else 0
                        epochs_remaining = self.total_epochs - current_epoch_num
                        estimated_time_remaining_seconds = epochs_remaining * avg_time_per_epoch_seconds if avg_time_per_epoch_seconds > 0 else 0

                        def format_seconds_to_hms(seconds):
                            if seconds < 0: seconds = 0
                            h = int(seconds // 3600); m = int((seconds % 3600) // 60); s = int(seconds % 60)
                            return f"{h:02d}:{m:02d}:{s:02d}"

                        log_lines = [f"✅ Epoch {current_epoch_num}/{self.total_epochs} Completed", f"   Total Time Elapsed: {format_seconds_to_hms(time_elapsed_total_seconds)}"]
                        if avg_time_per_epoch_seconds > 0: log_lines.append(f"   Avg. Time/Epoch: {avg_time_per_epoch_seconds:.2f}s")
                        if epochs_remaining > 0 and avg_time_per_epoch_seconds > 0 : log_lines.append(f"   Est. Time Remaining: {format_seconds_to_hms(estimated_time_remaining_seconds)}")
                        elif current_epoch_num == self.total_epochs: log_lines.append("   🏁 Training complete!")
                        else: log_lines.append("   Est. Time Remaining: Calculating...")

                        metrics_to_log = {}
                        if hasattr(trainer, 'metrics') and trainer.metrics:
                            if 'loss' in trainer.metrics: metrics_to_log['Loss'] = f"{trainer.metrics['loss']:.4f}"
                            for mk, dn in [('box_loss', 'BoxLoss'), ('cls_loss', 'ClsLoss'), ('dfl_loss', 'DFLLoss'), ('metrics/precision(B)', 'Precision(B)'), ('metrics/recall(B)', 'Recall(B)'), ('metrics/mAP50(B)', 'mAP50(B)'), ('metrics/mAP50-95(B)', 'mAP50-95(B)')]:
                                if mk in trainer.metrics: metrics_to_log[dn] = f"{trainer.metrics[mk]:.4f}"
                            for mk, dn in [('seg_loss', 'SegLoss'), ('metrics/precision(M)', 'Precision(M)'), ('metrics/recall(M)', 'Recall(M)'), ('metrics/mAP50(M)', 'mAP50(M)'), ('metrics/mAP50-95(M)', 'mAP50-95(M)')]:
                                if mk in trainer.metrics: metrics_to_log[dn] = f"{trainer.metrics[mk]:.4f}"
                        if metrics_to_log: log_lines.append(f"   Metrics: {', '.join([f'{name}: {val}' for name, val in metrics_to_log.items()])}")
                        self.log_func("\n".join(log_lines))
                        return False 
                
                callback = TrainingCallback(stop_flag, log_message, epochs)
                if hasattr(model_instance, 'add_callback'):
                    model_instance.add_callback('on_train_epoch_start', callback.on_train_epoch_start)
                    model_instance.add_callback('on_train_epoch_end', callback.on_train_epoch_end)
                
                log_message("🚀 Beginning YOLO training...")
                results = model_instance.train(data=dataset_yaml_local, epochs=epochs, imgsz=safe_imgsz, batch=safe_batch, lr0=lr, project=output_dir, device=device, workers=safe_workers, patience=20, save_period=5, cache=False, amp=False, verbose=True, exist_ok=True)
                
                log_message("✅ Training completed successfully!")
                save_dir_path = "Unknown" 
                if hasattr(results, 'save_dir'):
                    save_dir_path = results.save_dir
                    self.last_train_save_dir = save_dir_path
                    log_message(f"📁 Results saved to: {save_dir_path}")
                else:
                    log_message(f"📁 Results object does not have 'save_dir' attribute. Full results object: {results}")

                del model_instance, results; gc.collect()
                if hasattr(torch.cuda, 'empty_cache'): torch.cuda.empty_cache()
                log_message("🧹 Memory cleanup completed")
                messagebox.showinfo("Training Complete", f"Training completed successfully!\nResults saved to: {save_dir_path}", parent=train_win)
                
            except KeyboardInterrupt: log_message("⏹️ Training interrupted by user"); messagebox.showinfo("Training Interrupted", "Training was stopped by user.", parent=train_win)
            except MemoryError: log_message("💾 Out of memory! Try reducing batch size or image size."); messagebox.showerror("Memory Error", "Training failed due to insufficient memory.\n\nSolutions:\n1. Reduce batch size (try 1-2)\n2. Reduce image size (try 320 or 416)\n3. Close other applications\n4. Use a smaller model (yolov8n instead of yolov8l)", parent=train_win)
            except Exception as e_train: # Renamed
                error_msg = str(e_train)
                log_message(f"❌ Training failed: {error_msg}")
                solution_msg = f"Training failed:\n{error_msg}"
                if "segmentation fault" in error_msg.lower() or "access violation" in error_msg.lower(): solution_msg = "Training crashed due to a system-level error.\n\nSolutions:\n1. Restart the application\n2. Reduce batch size to 1\n3. Use smaller image size (320)\n4. Close other memory-intensive applications\n5. Try detection mode instead of segmentation"
                elif "out of memory" in error_msg.lower() or "memory" in error_msg.lower(): solution_msg = "Memory error during training.\n\nSolutions:\n1. Reduce batch size\n2. Reduce image size\n3. Use CPU instead of GPU"
                messagebox.showerror("Training Error", solution_msg, parent=train_win)
            
        except Exception as e_outer: # Renamed
            log_message(f"❌ Error: {str(e_outer)}")
            messagebox.showerror("Error", f"An error occurred:\n{str(e_outer)}", parent=train_win)
        finally:
            try: import gc, torch; gc.collect()
            except: pass
            if 'torch' in locals() and hasattr(torch.cuda, 'empty_cache'): torch.cuda.empty_cache()
            if start_btn.winfo_exists(): start_btn.config(state=tk.NORMAL)
        # Return the directory where the model was saved (or None if unavailable)
        return getattr(self, 'last_train_save_dir', None)

    # --------------------------------------------------
    # Annotation Export Functionality (related methods)
    # --------------------------------------------------

    def _execute_export(self, export_format):
        if export_format == "coco": self._export_coco()
        elif export_format == "pascal_voc": self._export_pascal_voc()
        elif export_format == "csv": self._export_csv()
        elif export_format == "yolo": self._export_yolo()
        else: messagebox.showerror("Export Error", f"Unknown export format: {export_format}", parent=self.root)

    def _get_all_annotations_data(self):
        all_bboxes_map = {}; all_polygons_map = {}
        cv2_module = lazy_importer.get_cv2()

        for image_relative_path in self.image_files:
            full_image_path = os.path.join(self.folder_path, image_relative_path)
            label_relative_path = os.path.splitext(image_relative_path)[0] + '.txt'
            label_path = os.path.join(self.label_folder, label_relative_path)

            if not os.path.exists(label_path): continue 

            try:
                height, width = -1, -1
                if self.image_path and os.path.normpath(full_image_path) == os.path.normpath(self.image_path) and self.original_image:
                    height, width = self.original_image.height, self.original_image.width
                else:
                    img_cv = cv2_module.imread(full_image_path)
                    if img_cv is None: logging.warning(f"Could not read image {full_image_path} to get dimensions for export."); continue
                    height, width = img_cv.shape[:2]
                
                if height == -1 or width == -1: logging.warning(f"Could not determine dimensions for {full_image_path}"); continue

                bboxes, polygons = read_annotations_from_file(label_path, (height, width))
                if bboxes: all_bboxes_map[image_relative_path] = bboxes
                if polygons: all_polygons_map[image_relative_path] = polygons
            except Exception as e: logging.error(f"Error processing annotations for {image_relative_path} during export prep: {e}", exc_info=True)
        return all_bboxes_map, all_polygons_map

    def _export_coco(self):
        try:
            all_bboxes_map, all_polygons_map = self._get_all_annotations_data()
            if not self.image_files: messagebox.showinfo("Export COCO", "No images in the project to export.", parent=self.root); return

            coco_data = convert_to_coco_format(self.image_files, all_bboxes_map,all_polygons_map,self.class_names,self.folder_path )
            save_path = filedialog.asksaveasfilename(defaultextension=".json",filetypes=[("COCO JSON files", "*.json"), ("All files", "*.*")],title="Save COCO Annotations",parent=self.root)
            if not save_path: return
            with open(save_path, 'w') as f: json.dump(coco_data, f, indent=4)
            messagebox.showinfo("Export Successful", f"Annotations exported to COCO format at:\n{save_path}", parent=self.root)
        except Exception as e: messagebox.showerror("Export Error", f"Failed to export to COCO format:\n{e}", parent=self.root); logging.error("Failed to export COCO", exc_info=True)

    def _export_pascal_voc(self):
        try:
            if not self.image_files: messagebox.showinfo("Export Pascal VOC", "No images in the project to export.", parent=self.root); return
            output_dir = filedialog.askdirectory(title="Select Directory to Save Pascal VOC XML Files",parent=self.root)
            if not output_dir: return

            cv2_module = lazy_importer.get_cv2(); exported_count = 0
            for image_relative_path in self.image_files:
                full_image_path = os.path.join(self.folder_path, image_relative_path)
                label_relative_path = os.path.splitext(image_relative_path)[0] + '.txt'
                label_path = os.path.join(self.label_folder, label_relative_path)
                image_shape = None
                if self.image_path and os.path.normpath(full_image_path) == os.path.normpath(self.image_path) and self.original_image:
                    image_shape = (self.original_image.height, self.original_image.width, 3) 
                else:
                    img_cv = cv2_module.imread(full_image_path)
                    if img_cv is not None: image_shape = img_cv.shape
                    else: logging.warning(f"Could not read image {full_image_path} for Pascal VOC export."); continue
                if image_shape is None: continue
                current_bboxes, current_polygons = [], []
                if os.path.exists(label_path): current_bboxes, current_polygons = read_annotations_from_file(label_path, image_shape[:2])
                xml_data_str = convert_to_pascal_voc_format(image_relative_path, current_bboxes,current_polygons,self.class_names,image_shape)
                xml_filename = os.path.splitext(os.path.basename(image_relative_path))[0] + ".xml"
                save_path = os.path.join(output_dir, xml_filename)
                with open(save_path, 'w', encoding='utf-8') as f: f.write(xml_data_str)
                exported_count +=1
            if exported_count > 0: messagebox.showinfo("Export Successful", f"{exported_count} XML files exported to Pascal VOC format in:\n{output_dir}", parent=self.root)
            else: messagebox.showinfo("Export Pascal VOC", "No annotations found or images processed for Pascal VOC export.", parent=self.root)
        except Exception as e: messagebox.showerror("Export Error", f"Failed to export to Pascal VOC format:\n{e}", parent=self.root); logging.error("Failed to export Pascal VOC", exc_info=True)

    def _export_csv(self):
        try:
            all_bboxes_map, all_polygons_map = self._get_all_annotations_data()
            if not self.image_files: messagebox.showinfo("Export CSV", "No images in the project to export.", parent=self.root); return
            csv_rows = convert_to_csv_format(self.image_files,all_bboxes_map,all_polygons_map,self.class_names)
            if len(csv_rows) <= 1: messagebox.showinfo("Export CSV", "No annotations found to export to CSV.", parent=self.root); return
            save_path = filedialog.asksaveasfilename(defaultextension=".csv",filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],title="Save Annotations as CSV",parent=self.root)
            if not save_path: return
            with open(save_path, 'w', newline='', encoding='utf-8') as f: writer = csv.writer(f); writer.writerows(csv_rows)
            messagebox.showinfo("Export Successful", f"Annotations exported to CSV format at:\n{save_path}", parent=self.root)
        except Exception as e: messagebox.showerror("Export Error", f"Failed to export to CSV format:\n{e}", parent=self.root); logging.error("Failed to export CSV", exc_info=True)

    def _export_yolo(self):
        try:
            if not os.path.isdir(self.label_folder) or not os.listdir(self.label_folder): messagebox.showinfo("Export YOLO", "No label files found in the 'labels' directory.", parent=self.root); return
            if not os.path.exists(self.yaml_path): messagebox.showinfo("Export YOLO", f"Dataset YAML file not found at {self.yaml_path}.", parent=self.root); return
            save_path = filedialog.asksaveasfilename(defaultextension=".zip",filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],title="Save YOLO Dataset as ZIP",parent=self.root)
            if not save_path: return
            temp_dir_for_zip = os.path.join(self.folder_path, "_temp_yolo_export")
            if os.path.exists(temp_dir_for_zip): shutil.rmtree(temp_dir_for_zip)
            os.makedirs(temp_dir_for_zip)
            temp_labels_dir = os.path.join(temp_dir_for_zip, "labels")
            shutil.copytree(self.label_folder, temp_labels_dir)
            shutil.copy2(self.yaml_path, os.path.join(temp_dir_for_zip, "dataset.yaml"))
            shutil.make_archive(os.path.splitext(save_path)[0], 'zip', temp_dir_for_zip)
            shutil.rmtree(temp_dir_for_zip)
            messagebox.showinfo("Export Successful", f"YOLO dataset (labels and dataset.yaml) zipped to:\n{save_path}", parent=self.root)
        except Exception as e: 
            messagebox.showerror("Export Error", f"Failed to export YOLO dataset as ZIP:\n{e}", parent=self.root)
            logging.error("Failed to export YOLO ZIP", exc_info=True)
            if 'temp_dir_for_zip' in locals() and os.path.exists(temp_dir_for_zip): # Check if var defined
                try: shutil.rmtree(temp_dir_for_zip)
                except Exception as e_clean: logging.error(f"Failed to cleanup temp export dir: {e_clean}")

    # --------------------------------------------------
    # Batch Operations for Image Status Management
    # --------------------------------------------------
    def _on_image_tree_right_click(self, event):
        """Show batch operations menu on right-click over image(s)."""
        item = self.image_tree.identify_row(event.y)
        if item:
            current_selection = self.image_tree.selection()
            if item not in current_selection:
                self.image_tree.selection_set(item)
            # Only show menu for images, not folders
            if "folder" in self.image_tree.item(item).get("tags", []):
                return
            self.batch_menu.tk_popup(event.x_root, event.y_root)

    def _batch_mark_status(self, status):
        """Mark all selected images with the given status tag."""
        for item in self.image_tree.selection():
            if "folder" in self.image_tree.item(item).get("tags", []):
                continue
            self.image_status[item] = status
            self.image_tree.item(item, tags=(status,))
            self.image_tree.set(item, "filename", f"Status: {status}")
        self.save_statuses()
        self.update_folder_status_display()
        self.update_status_labels()

    def _batch_delete_annotations(self):
        """Delete annotation files for all selected images and reset status."""
        for item in self.image_tree.selection():
            if "folder" in self.image_tree.item(item).get("tags", []):
                continue
            label_file = os.path.join(self.label_folder, os.path.splitext(item)[0] + ".txt")
            try:
                if os.path.exists(label_file):
                    os.remove(label_file)
            except Exception as e:
                logging.error(f"Failed to delete annotation for {item}: {e}")
            # Reset status to not_viewed
            self.image_status[item] = "not_viewed"
            self.image_tree.item(item, tags=("not_viewed",))
            self.image_tree.set(item, "filename", "Status: not_viewed")
        self.save_statuses()
        self.update_folder_status_display()
        self.update_status_labels()
