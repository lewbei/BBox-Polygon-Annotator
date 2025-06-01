import os
import json
import logging
from datetime import datetime

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from image_labelling.helpers import center_window
from image_labelling.constants import PROJECTS_DIR
from image_labelling.startup_optimizer import SplashScreen
from image_labelling.editor import BoundingBoxEditor
class ProjectManager:
    """
    Manages creating, opening, and deleting projects.
    Displays projects in an integrated panel.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Project Manager")
        center_window(self.root, 750, 500)  # Adjusted size

        self.main_container = ttk.Frame(root, padding="10")
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # PanedWindow for resizable layout
        self.paned_window = ttk.PanedWindow(self.main_container, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Left Pane: Action Buttons
        self.actions_frame = ttk.Labelframe(self.paned_window, text="Actions", padding="10")
        self.paned_window.add(self.actions_frame, weight=1) # Smaller weight for actions panel

        self.new_project_button = ttk.Button(self.actions_frame, text="New Project", command=self.new_project)
        self.new_project_button.pack(pady=5, fill=tk.X)

        self.open_selected_button = ttk.Button(self.actions_frame, text="Open Selected", command=self._open_selected_project_action, state=tk.DISABLED)
        self.open_selected_button.pack(pady=5, fill=tk.X)
        
        self.refresh_button = ttk.Button(self.actions_frame, text="Refresh List", command=self._populate_project_list)
        self.refresh_button.pack(pady=5, fill=tk.X)

        self.delete_selected_button = ttk.Button(self.actions_frame, text="Delete Selected", command=self._delete_selected_project_action, state=tk.DISABLED)
        self.delete_selected_button.pack(pady=5, fill=tk.X)
        
        ttk.Button(self.actions_frame, text="Quit", command=self.root.quit).pack(pady=20, fill=tk.X, side=tk.BOTTOM)

        # Right Pane: Project List
        self.projects_list_frame = ttk.Labelframe(self.paned_window, text="Existing Projects", padding="10")
        self.paned_window.add(self.projects_list_frame, weight=3) # Larger weight for project list

        columns = ("project_name", "dataset_path", "last_modified_date")
        self.project_tree = ttk.Treeview(
            self.projects_list_frame,
            columns=columns,
            show="headings",
            selectmode="browse"
        )
        self.project_tree.heading("project_name", text="Project Name")
        self.project_tree.column("project_name", anchor=tk.W, width=200, stretch=tk.NO)
        self.project_tree.heading("dataset_path", text="Dataset Path")
        self.project_tree.column("dataset_path", anchor=tk.W, width=300) # Allow dataset_path to expand
        self.project_tree.heading("last_modified_date", text="Last Modified Date")
        self.project_tree.column("last_modified_date", anchor=tk.W, width=150, stretch=tk.NO)
        
        self.project_tree_scrollbar = ttk.Scrollbar(self.projects_list_frame, orient="vertical", command=self.project_tree.yview)
        self.project_tree.configure(yscrollcommand=self.project_tree_scrollbar.set)
        
        self.project_tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.project_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.project_tree.bind("<<TreeviewSelect>>", self._on_project_select)
        self.project_tree.bind("<Double-1>", self._open_selected_project_action)

        self._populate_project_list()

    def _populate_project_list(self):
        """Populates the project treeview with projects from PROJECTS_DIR."""
        for item in self.project_tree.get_children():
            self.project_tree.delete(item)

        project_files = [f for f in os.listdir(PROJECTS_DIR) if f.endswith(".json")]
        if not project_files:
            self.project_tree.insert("", tk.END, iid="no_projects_placeholder", values=("No projects found.", "", ""), tags=("placeholder",))
            self._on_project_select() # Ensure buttons are disabled
            return

        for f_name in sorted(project_files):
            project_name_display = os.path.splitext(f_name)[0]
            dataset_path_display = "N/A"
            full_path = os.path.join(PROJECTS_DIR, f_name)
            try:
                last_modified_display = datetime.fromtimestamp(os.path.getmtime(full_path)).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                logging.error(f"Error getting last modified time for project file {f_name}", exc_info=True)
                last_modified_display = ""
            try:
                with open(full_path, "r") as f:
                    project_data = json.load(f)
                    dataset_path_display = project_data.get("dataset_path", "N/A")
            except Exception as e:
                logging.error(f"Error reading project file {f_name}", exc_info=True)
                messagebox.showerror(
                    "Error Reading Project",
                    f"Error reading project file {f_name}:\n{e}"
                )

            self.project_tree.insert("", tk.END, iid=f_name, values=(project_name_display, dataset_path_display, last_modified_display))
        self._on_project_select()

    def _on_project_select(self, event=None):
        """Handles selection changes in the project treeview."""
        selected_item_ids = self.project_tree.selection()
        if selected_item_ids:
            # Check if the selected item is not the placeholder
            first_selected_iid = selected_item_ids[0]
            if first_selected_iid != "no_projects_placeholder":
                self.open_selected_button.config(state=tk.NORMAL)
                self.delete_selected_button.config(state=tk.NORMAL)
                return
        
        self.open_selected_button.config(state=tk.DISABLED)
        self.delete_selected_button.config(state=tk.DISABLED)

    def _open_selected_project_action(self, event=None):
        """Loads the selected project and opens the editor."""
        selected_item_ids = self.project_tree.selection()
        if not selected_item_ids:
            messagebox.showwarning("No Project Selected", "Please select a project from the list to open.")
            return
        
        project_file_iid = selected_item_ids[0]
        if project_file_iid == "no_projects_placeholder":
             messagebox.showwarning("No Project Selected", "Please create or select a valid project.")
             return

        full_path = os.path.join(PROJECTS_DIR, project_file_iid)
        try:
            with open(full_path, "r") as f:
                project = json.load(f)
            
            # Destroy ProjectManager window first
            self.root.destroy() 
            # Then open the editor, which will start its own mainloop
            self.open_editor(project)
        except Exception as e:
            logging.exception(f"Error opening project {project_file_iid}")
            # Since self.root is destroyed, we can't use it as a parent for messagebox.
            # This error might not be visible if it happens after self.root.destroy().
            # Consider logging to a file or a more robust error display if this becomes an issue.
            # For now, we'll try to show it without a parent, or it will go to console/log.
            try:
                messagebox.showerror("Error Opening Project", f"Could not load project '{project_file_iid}':\n{e}")
            except tk.TclError: # If Tkinter is in a bad state
                 print(f"ERROR: Could not load project '{project_file_iid}':\n{e}")
            # Only refresh project list if window still exists
            try:
                self._populate_project_list()
            except tk.TclError:
                # Window was destroyed, ignore
                pass

    def _delete_selected_project_action(self):
        """Deletes the selected project's .json file."""
        selected_item_ids = self.project_tree.selection()
        if not selected_item_ids:
            messagebox.showwarning("No Project Selected", "Please select a project to delete.")
            return

        project_file_iid = selected_item_ids[0]
        if project_file_iid == "no_projects_placeholder":
             messagebox.showwarning("No Project Selected", "Cannot delete placeholder item.")
             return
             
        project_name_display = self.project_tree.item(project_file_iid)['values'][0]

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete project '{project_name_display}'?\nThis will delete the project file ({project_file_iid}) but NOT the dataset itself."):
            full_path = os.path.join(PROJECTS_DIR, project_file_iid)
            try:
                os.remove(full_path)
                messagebox.showinfo("Project Deleted", f"Project '{project_name_display}' deleted successfully.")
            except Exception as e:
                messagebox.showerror("Error Deleting Project", f"Could not delete project file '{project_file_iid}':\n{e}")
            finally:
                self._populate_project_list() 

    def new_project(self):
        """
        Opens a dialog to create a new project: user specifies project name + dataset path.
        """
        new_win = tk.Toplevel(self.root)
        new_win.transient(self.root)
        new_win.grab_set()
        new_win.title("New Project")

        # Use ttk style for consistency
        form_frame = ttk.Frame(new_win, padding="10")
        form_frame.pack(expand=True, fill=tk.BOTH)

        ttk.Label(form_frame, text="Project Name:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        name_entry = ttk.Entry(form_frame, width=40)
        name_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(form_frame, text="Dataset Path:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        dataset_entry = ttk.Entry(form_frame, width=40)
        dataset_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

        def browse_dataset():
            folder = filedialog.askdirectory(title="Select Dataset Folder")
            if folder:
                dataset_entry.delete(0, tk.END)
                dataset_entry.insert(0, folder)

        ttk.Button(form_frame, text="Browse", command=browse_dataset).grid(row=1, column=2, padx=5, pady=5)

        buttons_frame = ttk.Frame(form_frame)
        buttons_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        def create_project_action():
            project_name = name_entry.get().strip()
            dataset_path = dataset_entry.get().strip()
            if not project_name or not dataset_path:
                messagebox.showerror("Error", "Project name and dataset path are required.", parent=new_win)
                return
            
            if not os.path.isdir(dataset_path):
                messagebox.showerror("Error", "Dataset path must be a valid directory.", parent=new_win)
                return

            project = {
                "project_name": project_name,
                "dataset_path": dataset_path,
                "label_path": os.path.join(dataset_path, "labels") # Consistent with BoundingBoxEditor
            }
            # Sanitize project_name for use as a filename
            safe_project_filename = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in project_name).rstrip()
            if not safe_project_filename:
                safe_project_filename = "Untitled_Project"
            project_file = os.path.join(PROJECTS_DIR, f"{safe_project_filename}.json")

            if os.path.exists(project_file):
                if not messagebox.askyesno("Overwrite Project?", f"Project '{safe_project_filename}.json' already exists. Overwrite?", parent=new_win):
                    return

            try:
                with open(project_file, "w") as f:
                    json.dump(project, f, indent=4)
                messagebox.showinfo("Project Created", f"Project '{project_name}' created successfully as '{safe_project_filename}.json'.", parent=new_win)
                new_win.destroy()
                self._populate_project_list() # Refresh the list in ProjectManager
                # Do not automatically open the editor, let user open from the list
                # self.root.destroy() 
                # self.open_editor(project)
            except Exception as e:
                messagebox.showerror("Error Creating Project", f"Could not save project file:\n{e}", parent=new_win)


        ttk.Button(buttons_frame, text="Create Project", command=create_project_action).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Cancel", command=new_win.destroy).pack(side=tk.LEFT, padx=5)
        
        form_frame.columnconfigure(1, weight=1) # Make entry expand        center_window(new_win, 500, 150) # Adjusted size for new project dialog
        name_entry.focus_set()
    
    def open_editor(self, project):
        """
        Launches the BoundingBoxEditor in a new Tk window with the chosen project loaded.
        Returns True if editor starts, False otherwise (though return value isn't used by caller anymore).
        """
        editor_root = None # Initialize to None
        splash = None      # Initialize to None
        try:
            editor_root = tk.Tk() # This is now the main Tk root for the editor
            editor_root.withdraw() # Hide main window initially
            
            splash = SplashScreen(editor_root, subtitle="Loading dataset...")
            editor_root.update_idletasks() # Process pending events for splash screen
            
            # Try to create the editor instance
            # This is where BoundingBoxEditor.__init__ will run
            editor = BoundingBoxEditor(editor_root, project)
            
            # If BoundingBoxEditor initialization is successful
            if splash:
                splash.destroy()
                splash = None # Ensure it's not destroyed again in finally
            
            editor_root.deiconify() # Show main window
            editor_root.mainloop() # Start the editor's main event loop
            # Mainloop blocks, so code here won't run until editor closes.
            
        except Exception as e:
            logging.exception("Failed to create or run BoundingBoxEditor")
            if splash:
                try:
                    splash.destroy()
                except tk.TclError:
                    pass
            if editor_root:
                try:
                    editor_root.destroy()
                except tk.TclError:
                    pass
            # Show error without a parent if the original root is gone.
            try:
                messagebox.showerror("Editor Error", f"Failed to open editor:\n{e}")
            except tk.TclError:
                 print(f"EDITOR ERROR: Failed to open editor:\n{e}")
            # No return True/False needed as the caller doesn't use it anymore.
