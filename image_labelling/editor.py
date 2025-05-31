import os
import shutil
import json
import logging
import yaml
import threading
import csv
import xml.etree.ElementTree as ET
from datetime import datetime

import tkinter as tk
from tkinter import ttk # Import ttk
from tkinter import filedialog, colorchooser, simpledialog, messagebox
from PIL import Image, ImageTk # Import Image and ImageTk from Pillow
import numpy as np
# import json # Already imported

from image_labelling.constants import ICON_UNICODE, PROJECTS_DIR
from image_labelling.helpers import center_window, write_annotations_to_file, read_annotations_from_file, copy_files_recursive
from image_labelling.startup_optimizer import lazy_importer

# --------------------------------------------------
# BoundingBoxEditor Class
# --------------------------------------------------

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
        self.update_class_colors()        # Dictionary to hold status of each image
        self.image_status = {}

        # -----------------------------
        # Main UI Layout
        # -----------------------------
        # Top Bar: Buttons (Auto Annotate, Save, Load Model, Export)
        self.setup_top_bar()

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
        self.image = None # This will hold the PhotoImage for the currently displayed (possibly cropped/scaled) image
        self.image_path = None
        self.original_image = None # This will hold the PIL Image object of the full, original image
        self.bboxes = [] # Stores (x_orig, y_orig, w_orig, h_orig, class_id) for boxes in original image coordinates
        self.polygons = [] # Stores {'class_id': int, 'points': [(x1_orig, y1_orig), ...]} for polygons in original image coordinates
        
        self.current_bbox = None # Stores [x_orig, y_orig, w_orig, h_orig, class_id] of the box being drawn
        self.current_bbox_orig_start = None # Stores (x_orig_start, y_orig_start) for the current box
        self.rect_start_canvas = None # Stores (canvas_x, canvas_y) where current box drawing started
        self.rect = None # Canvas item ID for the temporary rectangle being drawn for a box

        self.current_polygon_points = [] # Stores [(x_orig, y_orig), ...] for the polygon currently being drawn
        self.polygon_drawing_active = False # True if a polygon is currently being drawn
        
        # Polygon point editing state
        self.dragging_point = False 
        self.drag_polygon_index = -1 
        self.drag_point_index = -1 
        self.hover_polygon_index = -1 
        self.hover_point_index = -1 
        # Polygon movement state
        self.dragging_whole_polygon = False 
        self.drag_whole_polygon_index = -1  
        self.polygon_move_start = (0, 0)
        
        # Flag to prevent immediate polygon creation after double-click completion
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

        self.load_dataset()
        self.setup_bindings()
        self.save_history()
        if self.auto_save_interval and self.auto_save_interval > 0:
            self.start_auto_save()

        self.root.after_idle(self._attempt_load_initial_image)


    def _attempt_load_initial_image(self):
        """Attempts to load the last opened image or the first image in the dataset."""
        if not self.image_files: 
            return

        loaded_an_image = False
        if 'last_opened_image_relative' in self.project:
            last_image_relative_path = self.project['last_opened_image_relative']
            if last_image_relative_path: 
                last_image_full_path = os.path.join(self.folder_path, last_image_relative_path)
                if os.path.exists(last_image_full_path) and last_image_relative_path in self.image_files:
                    try:
                        if self.image_tree.exists(last_image_relative_path):
                            self.image_tree.selection_set(last_image_relative_path)
                            self.image_tree.focus(last_image_relative_path)
                            self.image_tree.see(last_image_relative_path)
                            self.load_image(last_image_full_path)
                            loaded_an_image = True
                        else:
                            print(f"Info: Last opened image '{last_image_relative_path}' not found in tree. Tree items: {self.image_tree.get_children()}")
                    except tk.TclError as e:
                        print(f"Info: TclError while trying to select last opened image '{last_image_relative_path}': {e}")
                        
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
                     print(f"Info: First image '{first_image_relative_path}' not found in tree during fallback.")
            except tk.TclError as e:
                print(f"Info: TclError while trying to select first image '{first_image_relative_path}': {e}")

    # --------------------------------------------------
    # Setup / Layout Methods
    # --------------------------------------------------

    def setup_image_list_panel_widgets(self): 
        self.image_tree = ttk.Treeview(
            self.image_list_frame,
            columns=("filename",),
            show="headings",
            selectmode="browse"
        )
        self.image_tree.heading("filename", text="Images")
        self.image_tree.column("filename", anchor=tk.W)
        self.image_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.image_tree.tag_configure("edited", background="lightgreen")
        self.image_tree.tag_configure("viewed", background="lightblue")
        self.image_tree.tag_configure("not_viewed", background="white")
        self.image_tree.tag_configure("review_needed", background="red")

        scrollbar = tk.Scrollbar(self.image_list_frame, orient="vertical", command=self.image_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_tree.configure(yscrollcommand=scrollbar.set)
        self.image_tree.bind("<<TreeviewSelect>>", self.on_image_select)

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
        self.class_listbox = tk.Listbox(self.class_frame, exportselection=False) # Modified this line
        self.class_listbox.pack(pady=10, fill=tk.BOTH, expand=True)
        for cls in self.class_names:
            self.class_listbox.insert(tk.END, cls)
        self.class_listbox.bind("<<ListboxSelect>>", self.on_class_select)
        self.class_listbox.bind("<ButtonRelease-1>", self.on_class_select) # Add this line
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
            self.display_image()

    def on_pan_start(self, event):
        if self.zoom_level > 1.0:
            self.panning = True
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.pan_start_view_offset_x = self.image_view_offset_x
            self.pan_start_view_offset_y = self.image_view_offset_y
            self.canvas.config(cursor="hand1")

    def on_pan_drag(self, event):
        if self.panning and self.zoom_level > 1.0 and self.original_image is not None: # Middle mouse button drag for panning
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
            self.display_image()
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
                # Ensure indices are valid before attempting to update
                if 0 <= self.drag_polygon_index < len(self.polygons) and \
                   0 <= self.drag_point_index < len(self.polygons[self.drag_polygon_index]['points']):
                    
                    print(f"DEBUG: Dragging polygon {self.drag_polygon_index}, point {self.drag_point_index} to ({image_x_current}, {image_y_current})")
                    
                    # Update the specific point being dragged
                    self.polygons[self.drag_polygon_index]['points'][self.drag_point_index] = (image_x_current, image_y_current)
                    
                    # If the polygon is closed (first and last points are the same object or have same coords)
                    # and the point being dragged is the first point (which is also the last), update both.
                    # This handles the case where a closed polygon's vertex is moved.
                    if len(self.polygons[self.drag_polygon_index]['points']) > 1 and \
                       self.polygons[self.drag_polygon_index]['points'][0] == self.polygons[self.drag_polygon_index]['points'][-1]:
                        if self.drag_point_index == 0:
                             self.polygons[self.drag_polygon_index]['points'][-1] = (image_x_current, image_y_current)
                        elif self.drag_point_index == len(self.polygons[self.drag_polygon_index]['points']) -1:
                             self.polygons[self.drag_polygon_index]['points'][0] = (image_x_current, image_y_current)

                    self.display_annotations() # Redraw all annotations including the moved point

    def on_pan_release(self, event):
        # Panning release (Middle mouse or B2)
        if self.panning:
            self.panning = False
            self.canvas.config(cursor="")
        # Bounding box drawing release (Left mouse / B1)
        elif self.annotation_mode == 'box' and self.current_bbox and self.rect:
            self.canvas.delete(self.rect)
            self.rect = None
            self.current_bbox_orig_start = None
            self.rect_start_canvas = None
            self.display_annotations()
            self.save_history()
        # Polygon point dragging release (Left mouse / B1)
        elif self.dragging_point and self.annotation_mode == 'polygon':
            print(f"DEBUG: Ending point drag for polygon {self.drag_polygon_index}, point {self.drag_point_index}")
            self.dragging_point = False
            self.drag_polygon_index = -1
            self.drag_point_index = -1
            # Clear hover state to allow deselection
            self.hover_polygon_index = -1
            self.hover_point_index = -1
            self.save_history()
            self.display_annotations() # Redraw to update final state and clear hover if any
            self.canvas.config(cursor="") # Reset cursor
        
        # General cursor reset if not panning and not dragging a point
        if not self.panning and not self.dragging_point:
             self.canvas.config(cursor="")

    def clear_current_polygon_drawing(self):
        """Clears any visual artifacts of an in-progress polygon drawing."""
        self.canvas.delete("polygon_drawing") 
        self.canvas.delete("polygon_hover_point") 
        # self.polygon_line_ids = [] # Not used with current drawing logic

    def cancel_current_polygon(self):
        """Cancels the current in-progress polygon drawing."""
        self.clear_current_polygon_drawing()
        self.current_polygon_points = []
        self.polygon_drawing_active = False
        self.display_annotations() 

    def clear_polygon_hover_state(self):
        """
        Helper method to clear polygon hover state and refresh display.
        Returns True if something was cleared, False if nothing was selected.
        """
        if self.hover_polygon_index != -1 or self.hover_point_index != -1:
            self.hover_polygon_index = -1
            self.hover_point_index = -1
            self.canvas.config(cursor="")
            self.display_annotations()
            return True
        return False

    def on_escape_key(self, event):
        if self.annotation_mode == 'polygon' and self.polygon_drawing_active:
            self.cancel_current_polygon()
        elif self.annotation_mode == 'polygon':
            # Clear polygon hover state if any polygon point is selected
            if not self.clear_polygon_hover_state():
                # If nothing was selected, clear class selection
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

    def load_dataset(self):
        if not self.folder_path: messagebox.showerror("Error", "Dataset folder not set."); return
        for item in self.image_tree.get_children(): self.image_tree.delete(item)
        self.image_files = []
        for root_dir, _, files in os.walk(self.folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    relative_path = os.path.relpath(os.path.join(root_dir, file), self.folder_path)
                    self.image_files.append(relative_path)
        self.image_files.sort()
        if not self.image_files: messagebox.showinfo("No Images", "No images found in the selected folder."); return
        self.load_statuses()
        for relative_image_path in self.image_files:
            status = self.image_status.get(relative_image_path, "not_viewed")
            self.image_tree.insert("", tk.END, iid=relative_image_path, values=(relative_image_path,), tags=(status,))
        self.save_statuses()
        self.update_status_labels()

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
        if len(self.history) > self.max_history_size: self.history.pop(0); self.history_index -= 1
        self.update_undo_redo_buttons()

    def undo(self):
        if self.history_index > 0: self.history_index -= 1; self.restore_from_history()

    def redo(self):
        if self.history_index < len(self.history) - 1: self.history_index += 1; self.restore_from_history()

    def restore_from_history(self):
        if 0 <= self.history_index < len(self.history):
            state = self.history[self.history_index]
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
            relative_image_path = selected[0]
            image_path = os.path.join(self.folder_path, relative_image_path)
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

        cv2_module = lazy_importer.get_cv2() 
        original_image_cv = cv2_module.imread(self.image_path)
        if original_image_cv is None:
            messagebox.showerror("Error", f"Failed to load image: {self.image_path}\nFile might be missing, corrupted, or in an unsupported format.")
            self.image = None; self.original_image = None
            self.image_name_label.config(text=f"Error loading: {os.path.basename(self.image_path)}")
            self.bboxes = []; self.polygons = []
            self.display_image(); self.display_annotations()
            return

        original_image_cv = cv2_module.cvtColor(original_image_cv, cv2_module.COLOR_BGR2RGB)
        self.original_image = Image.fromarray(original_image_cv)
        
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
            print("Error: Project name not found, cannot save project config."); return
        project_name = self.project['project_name']
        safe_project_filename = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in project_name).rstrip()
        if not safe_project_filename: safe_project_filename = "Untitled_Project"
        project_file_path = os.path.join(PROJECTS_DIR, f"{safe_project_filename}.json")
        try:
            with open(project_file_path, "w") as f: json.dump(self.project, f, indent=4)
        except Exception as e: print(f"Error saving project configuration to {project_file_path}: {e}")

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
                for point_idx, (px_orig, py_orig) in enumerate(points_orig):
                    canvas_px, canvas_py = self.image_to_canvas_coords(px_orig, py_orig)
                    if canvas_px is not None and canvas_py is not None:
                        is_hovered = (i == self.hover_polygon_index and point_idx == self.hover_point_index)
                        if is_hovered:
                            self.canvas.create_oval(canvas_px-5, canvas_py-5, canvas_px+5, canvas_py+5, fill="yellow", outline="orange", width=2, tags="polygon")
                        else:
                            self.canvas.create_oval(canvas_px-3, canvas_py-3, canvas_px+3, canvas_py+3, fill=color, outline="white", width=1, tags="polygon")
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
                    # Check if clicking on an existing vertex to start dragging
                    if self.hover_polygon_index != -1 and self.hover_point_index != -1:
                        # Ensure the hover indices are valid for the polygons list
                        if 0 <= self.hover_polygon_index < len(self.polygons) and \
                           0 <= self.hover_point_index < len(self.polygons[self.hover_polygon_index]['points']):
                            print(f"DEBUG: Starting point drag for polygon {self.hover_polygon_index}, point {self.hover_point_index}")
                            self.dragging_point = True
                            self.drag_polygon_index = self.hover_polygon_index
                            self.drag_point_index = self.hover_point_index
                            self.canvas.config(cursor="fleur")
                            # Do not start a new polygon, just set flags for dragging
                            return
                        else:
                            # Reset hover indices if they became invalid (e.g., polygon deleted)
                            self.hover_polygon_index = -1
                            self.hover_point_index = -1

                    # If not dragging a point, check if clicking on empty area to deselect
                    if self.hover_polygon_index == -1 and self.hover_point_index == -1:
                        # Clicking on empty area - this can be used to deselect or start new polygon
                        # But first check if we just completed a polygon to avoid immediate new polygon creation
                        if self.polygon_just_completed:
                            print("DEBUG: Ignoring click immediately after polygon completion")
                            return
                        
                        # Clear any existing hover state (deselection)
                        self.clear_polygon_hover_state()
                        
                        # If nothing was selected, proceed to start a new polygon
                        print(f"DEBUG: Starting new polygon at ({image_x}, {image_y})")
                        current_selection_tuple = self.class_listbox.curselection()
                        if not current_selection_tuple:
                            messagebox.showwarning("No Class Selected", "Please select a class before drawing a polygon.", parent=self.root)
                            return
                        self.selected_class_index = current_selection_tuple[0]
                        
                        self.current_polygon_points = [(image_x, image_y)]
                        self.polygon_drawing_active = True
                        self.draw_current_polygon_drawing() # Draw the first point
                else: # Polygon drawing is active, add a new point
                    print(f"DEBUG: Adding point to existing polygon at ({image_x}, {image_y})")
                    self.current_polygon_points.append((image_x, image_y))
                    self.draw_current_polygon_drawing()
        # self.display_annotations() # Called by draw_current_polygon_drawing if needed, or when finalizing

    def on_motion(self, event):
        """
        Handles mouse motion events.
        - For active polygon drawing: provides live visual feedback by calling draw_current_polygon_drawing with current mouse coords.
        - For completed polygons: handles hover detection for vertex/polygon dragging.
        """
        if self.annotation_mode == 'polygon' and self.polygon_drawing_active and self.current_polygon_points:
            # Provide live feedback for the next segment and closing line to current mouse cursor
            self.draw_current_polygon_drawing(live_canvas_x=event.x, live_canvas_y=event.y)
        elif self.annotation_mode == 'polygon' and not self.dragging_point and not self.dragging_whole_polygon:
            # Hover detection for completed polygons (when not actively drawing or dragging)
            prev_hover_polygon = self.hover_polygon_index
            prev_hover_point = self.hover_point_index
            self.hover_polygon_index = -1
            self.hover_point_index = -1
            
            for poly_idx, poly_data in enumerate(self.polygons):
                points_orig = poly_data['points']
                for point_idx, (px_orig, py_orig) in enumerate(points_orig):
                    canvas_px, canvas_py = self.image_to_canvas_coords(px_orig, py_orig)
                    if canvas_px is not None and canvas_py is not None:
                        distance = ((event.x - canvas_px) ** 2 + (event.y - canvas_py) ** 2) ** 0.5
                        if distance <= 8: # Hover radius
                            self.hover_polygon_index = poly_idx
                            self.hover_point_index = point_idx
                            if prev_hover_polygon != poly_idx or prev_hover_point != point_idx:
                                print(f"DEBUG: Hovering over polygon {poly_idx}, point {point_idx}")
                            break
                if self.hover_polygon_index != -1:
                    break
            
            if self.hover_polygon_index != -1:
                self.canvas.config(cursor="hand2")
            else:
                self.canvas.config(cursor="")
            
            if (prev_hover_polygon != self.hover_polygon_index or 
                prev_hover_point != self.hover_point_index):
                self.display_annotations() # Redraw completed polygons with hover highlights
        # else:
            # Other cases (e.g., box mode, or already dragging a polygon/point)
            # pass # No specific action needed for these cases in on_motion by default

    def draw_current_polygon_drawing(self, live_canvas_x=None, live_canvas_y=None):
        """
        Draws the current in-progress polygon: committed points, lines between them,
        and live lines to the current mouse cursor if provided.
        Matches v9 style: red points, red lines, dashed closing line.
        """
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
                # The points in self.current_polygon_points are already original image coordinates.
                # Close the polygon by appending the first point.
                self.current_polygon_points.append(self.current_polygon_points[0]) 
                
                if self.selected_class_index is None:
                    messagebox.showwarning("No Class Selected", "Please select a class before completing the polygon.", parent=self.root)
                    self.cancel_current_polygon() # Clear the drawing
                    return

                # Store the polygon (points are already original image coords)
                self.polygons.append({
                    "class_id": self.selected_class_index, 
                    "points": self.current_polygon_points[:] # Use a copy
                })
                self.current_polygon_points = [] 
                self.polygon_drawing_active = False
                self.clear_current_polygon_drawing() # Clear temporary drawing
                
                # Reset hover state to prevent conflicts with subsequent clicks
                self.hover_polygon_index = -1
                self.hover_point_index = -1
                
                # Set flag to prevent immediate polygon creation after completion
                self.polygon_just_completed = True
                self.root.after(100, self._reset_polygon_completion_flag)  # Reset after 100ms
                
                self.display_annotations() # Redraw all finalized annotations
                self.save_history() 
            else: # Not enough points to form a polygon
                self.cancel_current_polygon()
        # If not in polygon drawing mode, or not enough points, display_annotations might still be needed
        # if other actions (like clearing selection) should refresh the main annotation display.
        # However, if double-click is only for polygon completion, this call might be too broad.
        # For now, keeping it to ensure display is up-to-date after any double-click attempt.
        # self.display_annotations() # Re-evaluate if this is needed here or only after successful polygon completion.
                                   # display_annotations is called within cancel_current_polygon if that path is taken.

    def _reset_polygon_completion_flag(self):
        """Reset the polygon completion flag to allow new polygon creation."""
        self.polygon_just_completed = False

    def on_pan_release(self, event):
        if self.panning:
            self.panning = False; self.canvas.config(cursor="")
        elif self.annotation_mode == 'box' and self.current_bbox and self.rect:
            self.canvas.delete(self.rect); self.rect = None
            self.current_bbox_orig_start = None; self.rect_start_canvas = None
            self.display_annotations(); self.save_history()
        if not self.panning: self.canvas.config(cursor="")

    def on_mouse_wheel(self, event):
        if not self.image_files: return
        if hasattr(event, 'delta'): delta = event.delta
        elif event.num == 4: delta = 120
        elif event.num == 5: delta = -120
        else: return
        if delta > 0: self.navigate_image(-1)
        elif delta < 0: self.navigate_image(1)
        self.display_image()

    def navigate_image(self, direction):
        if not self.image_files: return
        self.save_history()
        self.current_image_index += direction
        if self.current_image_index < 0: self.current_image_index = 0
        elif self.current_image_index >= len(self.image_files): self.current_image_index = len(self.image_files) - 1
        self.load_image(os.path.join(self.folder_path, self.image_files[self.current_image_index]))

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
            print(f"Warning: Original image not available for saving labels for {self.image_path}. Annotations might be incorrect.")
            fallback_shape = (480, 640) # Default fallback
            if hasattr(self, 'image') and self.image is not None and hasattr(self.image, 'shape'): # Check if self.image (numpy array) exists
                 pil_image_from_numpy = Image.fromarray(self.image)
                 fallback_shape = (pil_image_from_numpy.height, pil_image_from_numpy.width)
            write_annotations_to_file(label_path, self.bboxes, self.polygons, fallback_shape)
        print(f"Saved labels for {self.image_path}")
        new_status = "edited" if (self.bboxes or self.polygons) else "viewed"
        self.image_status[relative_image_path] = new_status
        self.image_tree.item(relative_image_path, tags=(new_status,))
        self.save_statuses(); self.update_status_labels()

    def delete_image(self):
        if self.current_image_index == -1: messagebox.showwarning("Warning", "No image selected to delete."); return
        if self.current_image_index == -1: messagebox.showwarning("Warning", "Cannot delete manually loaded image."); return
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
            self.load_image(os.path.join(self.folder_path, self.image_files[self.current_image_index]))
        else: self.current_image_index = -1
        self.update_status_labels(); self.save_history()

    # --------------------------------------------------
    # Class/Editor Utilities
    # --------------------------------------------------

    def delete_annotation(self, index, annotation_type):
        """Deletes a specific annotation (bbox or polygon) by index."""
        if annotation_type == 'bbox':
            if 0 <= index < len(self.bboxes):
                del self.bboxes[index]
        elif annotation_type == 'polygon':
            if 0 <= index < len(self.polygons):
                del self.polygons[index]
        
        self.display_annotations()
        self.save_history()
        # Update status as image might become "viewed" if all annotations are deleted
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
    # YOLO Model Loading / Auto-Annotation
    # --------------------------------------------------

    def load_model(self):
        model_path = filedialog.askopenfilename(title="Select YOLO Model", filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")])
        if model_path:
            YOLO = lazy_importer.get_yolo()
            try: self.model = YOLO(model_path); messagebox.showinfo("Success", f"Model loaded successfully from {model_path}")
            except Exception as e: messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def auto_annotate_dataset_threaded(self):
        if self.model is None: messagebox.showerror("Model Not Loaded", "Please load a YOLO model first."); return
        self.auto_annotate_button.config(state=tk.DISABLED)
        self.progress_win = tk.Toplevel(self.root); self.progress_win.title("Auto Annotation Progress")
        self.progress_win.transient(self.root); self.progress_win.grab_set()
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_win, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(padx=20, pady=10)
        self.progress_label = tk.Label(self.progress_win, text="0/0 images processed"); self.progress_label.pack(pady=5)
        self.cancel_button = tk.Button(self.progress_win, text="Cancel", command=self.cancel_annotation); self.cancel_button.pack(pady=5)
        self.progress_win.update_idletasks()
        main_width = self.root.winfo_width(); main_height = self.root.winfo_height()
        main_x = self.root.winfo_x(); main_y = self.root.winfo_y()
        progress_width = 300; progress_height = 100
        x = main_x + (main_width - progress_width) // 2; y = main_y + (main_height - progress_height) // 2
        self.progress_win.geometry(f"{progress_width}x{progress_height}+{x}+{y}")
        self.cancel_event = threading.Event()
        threading.Thread(target=self.auto_annotate_dataset, daemon=True).start()

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
        conf_threshold = self.confidence_threshold.get(); flagged_images = []; total_images = len(self.image_files); processed_count = 0
        try:
            for idx, image_file in enumerate(self.image_files):
                processed_count = idx + 1
                if self.cancel_event and self.cancel_event.is_set(): break
                image_path = os.path.join(self.folder_path, image_file)
                label_filename = os.path.splitext(image_file)[0] + '.txt'
                label_path = os.path.join(self.label_folder, label_filename)
                results = self.model(image_path, conf=conf_threshold, verbose=False)
                detections = results[0].boxes; bboxes = []; uncertain = False; img_h, img_w = None, None
                relative_image_path = image_file
                if not detections: self.image_status[relative_image_path] = "viewed"
                else:
                    for box in detections:
                        if self.cancel_event and self.cancel_event.is_set(): break
                        conf_score = box.conf[0].item(); class_id = int(box.cls[0])
                        if img_h is None or img_w is None: img_h, img_w = results[0].orig_shape[:2]
                        if class_id >= len(self.class_names): continue
                        np_module = lazy_importer.get_numpy() 
                        x_center, y_center, width, height = box.xywhn[0].cpu().numpy()
                        x_center_abs = x_center * img_w; y_center_abs = y_center * img_h
                        width_abs = width * img_w; height_abs = height * img_h
                        x_min = int(x_center_abs - width_abs / 2); y_min = int(y_center_abs - height_abs / 2)
                        bboxes.append((x_min, y_min, int(width_abs), int(height_abs), class_id, conf_score))
                        if conf_score < conf_threshold: uncertain = True
                    if bboxes:
                        if uncertain: flagged_images.append(relative_image_path); self.image_status[relative_image_path] = "review_needed"
                        else:
                            existing_bboxes, existing_polygons = read_annotations_from_file(label_path, (img_h, img_w))
                            new_bboxes_for_file = [(x,y,w,h,cid) for (x,y,w,h,cid,_) in bboxes]
                            write_annotations_to_file(label_path, new_bboxes_for_file, existing_polygons, (img_h, img_w))
                            self.image_status[relative_image_path] = "edited"
                    else: self.image_status[relative_image_path] = "viewed"
                progress_percent = (processed_count / total_images) * 100
                self.root.after(0, self.update_progress, progress_percent, processed_count, total_images)
                if self.cancel_event.is_set(): break
        except Exception as e: self.root.after(0, lambda: messagebox.showerror("Error", f"Annotation failed: {str(e)}"))
        finally:
            self.save_statuses(); self.root.after(0, self.update_status_labels)
            if hasattr(self, 'progress_win') and self.progress_win.winfo_exists(): self.root.after(0, self.progress_win.destroy)
            self.root.after(0, lambda: self.auto_annotate_button.config(state=tk.NORMAL))
            for relative_image_path in self.image_files: self.image_tree.item(relative_image_path, tags=(self.image_status.get(relative_image_path, "not_viewed"),))
            if self.cancel_event.is_set(): self.root.after(0, lambda: messagebox.showinfo("Cancelled", f"Annotation cancelled. Processed {processed_count}/{total_images} images."))
            elif flagged_images: self.root.after(0, lambda: messagebox.showwarning("Review Needed", f"{len(flagged_images)} images have low-confidence detections requiring review."))
            else: self.root.after(0, lambda: messagebox.showinfo("Complete", "Auto-annotation finished successfully!"))

    # --------------------------------------------------
    # YAML Export Logic (Refactored)
    # --------------------------------------------------
    def _export_yaml_logic(self, base_export_folder, split_option, test_data, include_val):
        try:
            with open(self.yaml_path, "r") as f: data = yaml.safe_load(f)
        except Exception as e: messagebox.showerror("Error", f"Could not load YAML file:\\n{e}"); return
        if not include_val: data["val"] = ""
        export_folder = os.path.join(base_export_folder, "exported_yaml_dataset"); os.makedirs(export_folder, exist_ok=True)
        if split_option == "in_sample":
            train_folder = os.path.join(export_folder, "train")
            os.makedirs(os.path.join(train_folder, "images"), exist_ok=True); os.makedirs(os.path.join(train_folder, "labels"), exist_ok=True)
            val_folder = train_folder; data["train"] = os.path.relpath(train_folder, export_folder); data["val"] = os.path.relpath(val_folder, export_folder)
            test_files = []
        else: 
            annotated = []
            for relative_image_path in self.image_files:
                label_relative_path = os.path.splitext(relative_image_path)[0] + '.txt'
                src_label_path = os.path.join(self.label_folder, label_relative_path)
                if os.path.exists(src_label_path) and os.path.getsize(src_label_path) > 0:
                    with open(src_label_path, 'r') as f:
                        if any(line.strip() for line in f): annotated.append(relative_image_path)
            num_annotated = len(annotated)
            if num_annotated ==  0: messagebox.showwarning("Warning", "No annotated images found for YAML export."); return
            if test_data:
                num_train = int(num_annotated * 0.6); num_val = int(num_annotated * 0.2)
                train_files = annotated[:num_train]; val_files = annotated[num_train : num_train + num_val]; test_files = annotated[num_train + num_val :]
            else:
                num_train = int(num_annotated * 0.8); train_files = annotated[:num_train]; val_files = annotated[num_train:]; test_files = []
            train_folder = os.path.join(export_folder, "train")
            os.makedirs(os.path.join(train_folder, "images"), exist_ok=True); os.makedirs(os.path.join(train_folder, "labels"), exist_ok=True)
            data["train"] = os.path.relpath(train_folder, export_folder)
            val_folder = os.path.join(export_folder, "val")
            os.makedirs(os.path.join(val_folder, "images"), exist_ok=True); os.makedirs(os.path.join(val_folder, "labels"), exist_ok=True)
            data["val"] = os.path.relpath(val_folder, export_folder)
            if test_files:
                test_folder_path = os.path.join(export_folder, "test")
                os.makedirs(os.path.join(test_folder_path, "images"), exist_ok=True); os.makedirs(os.path.join(test_folder_path, "labels"), exist_ok=True)
                data["test"] = os.path.relpath(test_folder_path, export_folder)
            else: data["test"] = ""
        data["path"] = "." 
        export_yaml_path = os.path.join(export_folder, "dataset.yaml")
        try:
            with open(export_yaml_path, "w") as f: yaml.dump(data, f, sort_keys=False)
        except Exception as e: messagebox.showerror("Error", f"Could not export YAML:\\n{e}"); return
        if split_option == "in_sample":
            all_labeled = [rf_path for rf_path in self.image_files if os.path.exists(os.path.join(self.label_folder, os.path.splitext(rf_path)[0] + '.txt'))]
            copy_files_recursive(all_labeled, self.folder_path, os.path.join(train_folder, "images"), self.label_folder, os.path.join(train_folder, "labels"))
        else: 
            copy_files_recursive(train_files, self.folder_path, os.path.join(train_folder, "images"), self.label_folder, os.path.join(train_folder, "labels"))
            copy_files_recursive(val_files, self.folder_path, os.path.join(val_folder, "images"), self.label_folder, os.path.join(val_folder, "labels"))
            if test_files: 
                test_folder_path = os.path.join(export_folder, "test")
                copy_files_recursive(test_files, self.folder_path, os.path.join(test_folder_path, "images"), self.label_folder, os.path.join(test_folder_path, "labels"))
        messagebox.showinfo("Success", f"YAML Dataset export complete!\\nExported to:\\n{export_folder}")

    # --------------------------------------------------
    # Export Format Conversion Functions
    # --------------------------------------------------
    
    @staticmethod
    def convert_to_coco_format(image_files, all_bboxes, all_polygons, class_names, base_folder):
        coco_data = {"info": {"description": "Dataset exported from BBox & Polygon Annotator", "version": "1.0", "year": datetime.now().year, "contributor": "BBox & Polygon Annotator v9", "date_created": datetime.now().isoformat()},
                     "licenses": [{"id": 1, "name": "Unknown", "url": ""}], "images": [], "annotations": [], "categories": []}
        for i, class_name in enumerate(class_names): coco_data["categories"].append({"id": i, "name": class_name, "supercategory": "object"})
        annotation_id = 1
        for img_idx, image_path in enumerate(image_files):
            full_image_path = os.path.join(base_folder, image_path)
            if os.path.exists(full_image_path):
                cv2_module = lazy_importer.get_cv2(); img = cv2_module.imread(full_image_path) 
                height, width = img.shape[:2]
            else: width, height = 640, 480
            coco_data["images"].append({"id": img_idx, "width": width, "height": height, "file_name": os.path.basename(image_path)})
            if image_path in all_bboxes:
                for bbox in all_bboxes[image_path]:
                    x, y, w, h, class_id = bbox
                    coco_data["annotations"].append({"id": annotation_id, "image_id": img_idx, "category_id": class_id, "bbox": [x,y,w,h], "area": w*h, "iscrowd": 0}); annotation_id += 1
            if image_path in all_polygons:
                for polygon in all_polygons[image_path]:
                    class_id = polygon['class_id']; points = polygon['points']; segmentation = []
                    for x,y in points: segmentation.extend([float(x), float(y)])
                    xs = [p[0] for p in points]; ys = [p[1] for p in points]
                    x_min, x_max = min(xs), max(xs); y_min, y_max = min(ys), max(ys)
                    bbox_w, bbox_h = x_max - x_min, y_max - y_min; area = bbox_w * bbox_h
                    coco_data["annotations"].append({"id": annotation_id, "image_id": img_idx, "category_id": class_id, "segmentation": [segmentation], "bbox": [x_min,y_min,bbox_w,bbox_h], "area": area, "iscrowd": 0}); annotation_id += 1
        return coco_data
    
    @staticmethod
    def convert_to_pascal_voc_format(image_path, bboxes, polygons, class_names, image_shape):
        height, width = image_shape[:2]; annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = os.path.dirname(image_path) or "images"
        ET.SubElement(annotation, "filename").text = os.path.basename(image_path)
        source = ET.SubElement(annotation, "source"); ET.SubElement(source, "database").text = "BBox & Polygon Annotator"
        size = ET.SubElement(annotation, "size"); ET.SubElement(size, "width").text = str(width); ET.SubElement(size, "height").text = str(height); ET.SubElement(size, "depth").text = "3"
        ET.SubElement(annotation, "segmented").text = "1" if polygons else "0"
        for x,y,w,h,class_id in bboxes:
            obj = ET.SubElement(annotation, "object"); ET.SubElement(obj, "name").text = class_names[class_id]
            ET.SubElement(obj, "pose").text = "Unspecified"; ET.SubElement(obj, "truncated").text = "0"; ET.SubElement(obj, "difficult").text = "0"
            bndbox = ET.SubElement(obj, "bndbox"); ET.SubElement(bndbox, "xmin").text = str(int(x)); ET.SubElement(bndbox, "ymin").text = str(int(y)); ET.SubElement(bndbox, "xmax").text = str(int(x+w)); ET.SubElement(bndbox, "ymax").text = str(int(y+h))
        for polygon in polygons:
            class_id = polygon['class_id']; points = polygon['points']
            obj = ET.SubElement(annotation, "object"); ET.SubElement(obj, "name").text = class_names[class_id]
            ET.SubElement(obj, "pose").text = "Unspecified"; ET.SubElement(obj, "truncated").text = "0"; ET.SubElement(obj, "difficult").text = "0"
            xs = [p[0] for p in points]; ys = [p[1] for p in points]; x_min, x_max = min(xs), max(xs); y_min, y_max = min(ys), max(ys)
            bndbox = ET.SubElement(obj, "bndbox"); ET.SubElement(bndbox, "xmin").text = str(int(x_min)); ET.SubElement(bndbox, "ymin").text = str(int(y_min)); ET.SubElement(bndbox, "xmax").text = str(int(x_max)); ET.SubElement(bndbox, "ymax").text = str(int(y_max))
            polygon_elem = ET.SubElement(obj, "polygon")
            for i, (px,py) in enumerate(points): point = ET.SubElement(polygon_elem, f"point{i+1}"); point.set("x", str(int(px))); point.set("y", str(int(py)))
        return ET.tostring(annotation, encoding='unicode')
    
    @staticmethod
    def convert_to_csv_format(image_files, all_bboxes, all_polygons, class_names):
        rows = []; headers = ["image_name", "annotation_type", "class_name", "class_id", "coordinates", "area"]; rows.append(headers)
        for image_path in image_files:
            image_name = os.path.basename(image_path)
            if image_path in all_bboxes:
                for x,y,w,h,class_id in all_bboxes[image_path]:
                    coordinates = f"x={x},y={y},w={w},h={h}"; area = w*h
                    rows.append([image_name, "bbox", class_names[class_id], class_id, coordinates, area])
            if image_path in all_polygons:
                for polygon in all_polygons[image_path]:
                    class_id = polygon['class_id']; points = polygon['points']; coordinates = ";".join([f"{x},{y}" for x,y in points])
                    area = 0.5 * abs(sum(points[i][0] * (points[(i+1)%len(points)][1] - points[i-1][1]) for i in range(len(points)))) if len(points) >= 3 else 0
                    rows.append([image_name, "polygon", class_names[class_id], class_id, coordinates, area])
        return rows

    def export_format_selection_window(self):
        export_win = tk.Toplevel(self.root); export_win.title("Export Annotations"); export_win.transient(self.root); export_win.grab_set(); center_window(export_win, 500, 450)
        format_frame = tk.LabelFrame(export_win, text="Export Format"); format_frame.pack(fill=tk.X, padx=10, pady=5)
        format_var = tk.StringVar(value="yaml"); formats = [("YAML (YOLO Training)","yaml"),("COCO JSON","coco"),("Pascal VOC XML","voc"),("CSV Spreadsheet","csv"),("JSON (Generic)","json")]
        for text, value in formats: tk.Radiobutton(format_frame, text=text, variable=format_var, value=value).pack(anchor=tk.W, padx=5, pady=2)
        location_frame = tk.LabelFrame(export_win, text="Export Location"); location_frame.pack(fill=tk.X, padx=10, pady=5)
        export_to_current_var = tk.BooleanVar(value=True); tk.Checkbutton(location_frame, text="Export to Current Dataset Location", variable=export_to_current_var).pack(anchor=tk.W, padx=5, pady=2)
        custom_frame = tk.Frame(location_frame); custom_frame.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(custom_frame, text="Custom Location:").pack(side=tk.LEFT); custom_export_entry = tk.Entry(custom_frame, width=40); custom_export_entry.pack(side=tk.LEFT, padx=5)
        custom_export_button = tk.Button(custom_frame, text="Browse", command=lambda: custom_export_entry.insert(0, filedialog.askdirectory(title="Select Export Folder") or "")); custom_export_button.pack(side=tk.LEFT)
        def toggle_custom(*args):
            state = "disabled" if export_to_current_var.get() else "normal"
            custom_export_entry.config(state=state); custom_export_button.config(state=state)
        export_to_current_var.trace("w", toggle_custom); toggle_custom()
        yaml_frame = tk.LabelFrame(export_win, text="YAML Options (YOLO Training)"); yaml_frame.pack(fill=tk.X, padx=10, pady=5)
        split_option = tk.StringVar(value="in_sample")
        tk.Radiobutton(yaml_frame, text="In-Sample Validation", variable=split_option, value="in_sample").pack(anchor=tk.W, padx=5, pady=2)
        tk.Radiobutton(yaml_frame, text="Split Data (Train/Val/Test)", variable=split_option, value="split").pack(anchor=tk.W, padx=5, pady=2)
        test_data_var = tk.BooleanVar(value=False); test_data_check = tk.Checkbutton(yaml_frame, text="Include Test Data (60:20:20 split)", variable=test_data_var); test_data_check.pack(anchor=tk.W, padx=20, pady=2)
        include_val_var = tk.BooleanVar(value=self.validation); tk.Checkbutton(yaml_frame, text="Include Validation in YAML", variable=include_val_var).pack(anchor=tk.W, padx=5, pady=2)
        def toggle_yaml_options(*args):
            if format_var.get() == "yaml": yaml_frame.pack(fill=tk.X, padx=10, pady=5); test_data_check.config(state="normal" if split_option.get() == "split" else "disabled")
            else: yaml_frame.pack_forget()
        def toggle_test_data_check(*args): test_data_check.config(state="normal" if split_option.get() == "split" and format_var.get() == "yaml" else "disabled")
        format_var.trace("w", toggle_yaml_options); split_option.trace("w", toggle_test_data_check); toggle_yaml_options()
        options_frame = tk.LabelFrame(export_win, text="Export Options"); options_frame.pack(fill=tk.X, padx=10, pady=5)
        include_images_var = tk.BooleanVar(value=True); tk.Checkbutton(options_frame, text="Copy Images to Export Folder", variable=include_images_var).pack(anchor=tk.W, padx=5, pady=2)
        include_unannotated_var = tk.BooleanVar(value=False); tk.Checkbutton(options_frame, text="Include Unannotated Images", variable=include_unannotated_var).pack(anchor=tk.W, padx=5, pady=2)
        button_frame = tk.Frame(export_win); button_frame.pack(fill=tk.X, padx=10, pady=10)
        def perform_export():
            selected_format = format_var.get()
            base_export_folder = self.folder_path if export_to_current_var.get() else custom_export_entry.get().strip()
            if not base_export_folder and not export_to_current_var.get(): messagebox.showerror("Error", "Please select a custom export location."); return
            if selected_format == "yaml": self._export_yaml_logic(base_export_folder, split_option.get(), test_data_var.get(), include_val_var.get())
            else: self.export_to_other_formats(selected_format, base_export_folder, include_images_var.get(), include_unannotated_var.get())
            export_win.destroy()
        tk.Button(button_frame, text="Export", command=perform_export).pack(side=tk.RIGHT, padx=5)
        tk.Button(button_frame, text="Cancel", command=export_win.destroy).pack(side=tk.RIGHT, padx=5)

    def export_to_other_formats(self, format_type, base_folder, copy_images, include_unannotated):
        try:
            export_format_folder = os.path.join(base_folder, f"exported_{format_type}_dataset"); os.makedirs(export_format_folder, exist_ok=True)
            all_bboxes = {}; all_polygons = {}; annotated_images = []
            for image_path in self.image_files:
                label_relative_path = os.path.splitext(image_path)[0] + '.txt'
                label_full_path = os.path.join(self.label_folder, label_relative_path)
                if os.path.exists(label_full_path) and os.path.getsize(label_full_path) > 0:
                    full_image_path = os.path.join(self.folder_path, image_path)
                    if os.path.exists(full_image_path):
                        cv2_module = lazy_importer.get_cv2(); img = cv2_module.imread(full_image_path) 
                        if img is not None:
                            height, width = img.shape[:2]
                            bboxes, polygons = read_annotations_from_file(label_full_path, (height,width))
                            if bboxes or polygons: all_bboxes[image_path] = bboxes; all_polygons[image_path] = polygons; annotated_images.append(image_path)
            if format_type == "coco":
                coco_data = BoundingBoxEditor.convert_to_coco_format(self.image_files, all_bboxes, all_polygons, self.class_names, self.folder_path)
                with open(os.path.join(export_format_folder, "annotations.json"), "w") as f: json.dump(coco_data, f, indent=2)
                messagebox.showinfo("Export Complete", f"COCO format export successful to:\n{export_format_folder}")
            elif format_type == "voc":
                voc_output_dir = os.path.join(export_format_folder, "Annotations"); os.makedirs(voc_output_dir, exist_ok=True)
                for image_path in annotated_images:
                    bboxes = all_bboxes.get(image_path, []); polygons = all_polygons.get(image_path, [])
                    full_image_path = os.path.join(self.folder_path, image_path)
                    if os.path.exists(full_image_path):
                        cv2_module = lazy_importer.get_cv2(); img = cv2_module.imread(full_image_path) 
                        if img is not None:
                            height, width = img.shape[:2]
                            xml_str = BoundingBoxEditor.convert_to_pascal_voc_format(image_path, bboxes, polygons, self.class_names, (height,width))
                            image_name = os.path.splitext(os.path.basename(image_path))[0]
                            with open(os.path.join(voc_output_dir, f"{image_name}.xml"), "w") as xml_file: xml_file.write(xml_str)
                messagebox.showinfo("Export Complete", f"Pascal VOC format export successful to:\n{voc_output_dir}")
            elif format_type == "csv":
                csv_rows = BoundingBoxEditor.convert_to_csv_format(self.image_files, all_bboxes, all_polygons, self.class_names)
                with open(os.path.join(export_format_folder, "annotations.csv"), "w", newline="") as csv_file: csv.writer(csv_file).writerows(csv_rows)
                messagebox.showinfo("Export Complete", f"CSV format export successful to:\n{export_format_folder}")
            elif format_type == "json":
                json_data = {"images": [], "annotations": [], "categories": [{"id": i, "name": name} for i,name in enumerate(self.class_names)]}
                annotation_id = 1
                for img_idx, image_path in enumerate(self.image_files):
                    full_image_path = os.path.join(self.folder_path, image_path); width, height = 640,480
                    if os.path.exists(full_image_path):
                        cv2_module = lazy_importer.get_cv2(); img = cv2_module.imread(full_image_path) 
                        if img is not None: height, width = img.shape[:2]
                    json_data["images"].append({"id":img_idx, "file_name":os.path.basename(image_path), "width":width, "height":height})
                    if image_path in all_bboxes:
                        for x,y,w,h,class_id in all_bboxes[image_path]:
                            json_data["annotations"].append({"id":annotation_id, "image_id":img_idx, "category_id":class_id, "bbox":[x,y,w,h], "area":w*h, "type":"bbox"}); annotation_id+=1
                    if image_path in all_polygons:
                        for polygon in all_polygons[image_path]:
                            class_id = polygon['class_id']; points = polygon['points']
                            json_data["annotations"].append({"id":annotation_id, "image_id":img_idx, "category_id":class_id, "polygon":points, "type":"polygon"}); annotation_id+=1
                with open(os.path.join(export_format_folder, "annotations.json"), "w") as json_file: json.dump(json_data, json_file, indent=2)
                messagebox.showinfo("Export Complete", f"JSON format export successful to:\n{export_format_folder}")
            if copy_images:
                images_output_dir = os.path.join(export_format_folder, "images"); os.makedirs(images_output_dir, exist_ok=True)
                images_to_copy = annotated_images if not include_unannotated else self.image_files
                for relative_image_path in images_to_copy:
                    src_image_path = os.path.join(self.folder_path, relative_image_path)
                    dst_image_path = os.path.join(images_output_dir, relative_image_path)
                    os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
                    if os.path.exists(src_image_path): shutil.copy(src_image_path, dst_image_path)
        except Exception as e: logging.exception("Error during export_to_other_formats"); messagebox.showerror("Export Error", f"An error occurred during export:\n{e}")

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
