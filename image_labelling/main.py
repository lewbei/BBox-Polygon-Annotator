import os
import sys
import logging
import logging.handlers
import tkinter as tk
from tkinter import messagebox, ttk

if __name__ == "__main__" and __package__ is None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(script_dir))
    __package__ = "image_labelling"

# Critical fix: Set multiprocessing start method to prevent segmentation faults
import multiprocessing
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

from .project_manager import ProjectManager

def main():
    """Main entry point for the application."""
    # Configure signal handling for graceful shutdown
    import signal
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(0)
    
    # Handle common signals that might cause segmentation faults
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    # Configure logging
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'error.log')
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.handlers.RotatingFileHandler(log_file_path, maxBytes=1024*1024, backupCount=5), # 1MB per file, 5 backups
            logging.StreamHandler() # Also log to console
        ]
    )

    root = tk.Tk()
    def report_callback_exception(self, exc, val, tb):
        logging.error("Exception in Tkinter callback", exc_info=(exc, val, tb))
        messagebox.showerror("Error", f"An unexpected error occurred:\n{val}")
    tk.Tk.report_callback_exception = report_callback_exception

    style = ttk.Style(root)
    available_themes = style.theme_names()
    if "clam" in available_themes:
        style.theme_use("clam")
    elif "vista" in available_themes and os.name == 'nt':
        style.theme_use("vista")

    try:
        pm = ProjectManager(root)
        root.mainloop()
    except Exception as e:
        logging.exception("Fatal error during application initialization")
        messagebox.showerror("Fatal Error", f"A fatal error occurred:\n{e}")

if __name__ == "__main__":
    main()
