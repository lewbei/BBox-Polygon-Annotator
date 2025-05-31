"""
Startup performance optimization utilities for the BBox & Polygon Annotator.
Provides lazy loading, progress indicators, and startup timing.
"""

import time
import threading
import tkinter as tk
from tkinter import ttk
import logging

from typing import Any, Dict

class StartupTimer:
    """Measures and logs startup performance metrics."""
    
    def __init__(self) -> None:
        self.start_time = time.time()
        self.checkpoints = {}
    
    def checkpoint(self, name: str) -> None:
        """Record a checkpoint during startup."""
        elapsed = time.time() - self.start_time
        self.checkpoints[name] = elapsed
        logging.info(f"Startup checkpoint '{name}': {elapsed:.3f}s")
    
    def total_time(self) -> float:
        """Get total startup time."""
        return time.time() - self.start_time
    
    def log_summary(self) -> None:
        """Log a summary of all startup timings."""
        total = self.total_time()
        logging.info(f"=== Startup Performance Summary ===")
        logging.info(f"Total startup time: {total:.3f}s")
        for name, elapsed in self.checkpoints.items():
            percentage = (elapsed / total) * 100
            logging.info(f"  {name}: {elapsed:.3f}s ({percentage:.1f}%)")

class LazyImporter:
    """Lazy loading utility for heavy modules."""
    
    def __init__(self) -> None:
        self._modules = {}
    
    def get_cv2(self) -> Any:
        """Lazy load OpenCV."""
        if 'cv2' not in self._modules:
            logging.info("Lazy loading cv2...")
            import cv2
            self._modules['cv2'] = cv2
        return self._modules['cv2']
    
    def get_yolo(self) -> Any:
        """Lazy load YOLO from ultralytics."""
        if 'YOLO' not in self._modules:
            logging.info("Lazy loading YOLO...")
            from ultralytics import YOLO
            self._modules['YOLO'] = YOLO
        return self._modules['YOLO']
    
    def get_pil(self) -> Any:
        """Lazy load PIL components."""
        if 'PIL' not in self._modules:
            logging.info("Lazy loading PIL...")
            from PIL import Image, ImageTk
            self._modules['PIL'] = {'Image': Image, 'ImageTk': ImageTk}
        return self._modules['PIL']
    
    def get_numpy(self) -> Any:
        """Lazy load numpy."""
        if 'numpy' not in self._modules:
            logging.info("Lazy loading numpy...")
            import numpy as np
            self._modules['numpy'] = np
        return self._modules['numpy']

class SplashScreen:
    """Simple splash screen with progress indication."""
    
    def __init__(self, parent: tk.Tk, title: str = "BBox & Polygon Annotator", subtitle: str = "Loading...") -> None:
        self.splash = tk.Toplevel(parent)
        self.splash.title(title)
        self.splash.geometry("400x200")
        self.splash.resizable(False, False)
        
        # Center the splash screen relative to the parent or screen
        parent.update_idletasks() # Ensure parent dimensions are up-to-date
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()

        if parent_width > 1 and parent_height > 1: # Check if parent is visible/sized
             x = parent_x + (parent_width // 2) - 200
             y = parent_y + (parent_height // 2) - 100
        else: # Fallback to screen centering if parent is not yet sized (e.g. withdrawn)
            screen_width = self.splash.winfo_screenwidth()
            screen_height = self.splash.winfo_screenheight()
            x = (screen_width // 2) - 200
            y = (screen_height // 2) - 100
        
        self.splash.geometry(f"+{x}+{y}")
        
        # Remove window decorations
        self.splash.overrideredirect(True)
        
        # Make it modal-like
        self.splash.transient(parent)
        self.splash.grab_set()
        
        # Main frame
        main_frame = tk.Frame(self.splash, bg='white', relief='raised', bd=2)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(main_frame, text=title, font=('Arial', 16, 'bold'), bg='white')
        title_label.pack(pady=(20, 10))
        
        # Subtitle
        self.subtitle_label = tk.Label(main_frame, text=subtitle, font=('Arial', 10), bg='white')
        self.subtitle_label.pack(pady=(0, 10))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', length=300)
        self.progress.pack(pady=(10, 20))
        self.progress.start(10)
        
        # Status label
        self.status_label = tk.Label(main_frame, text="Initializing...", font=('Arial', 9), bg='white', fg='gray')
        self.status_label.pack()
        
        self.splash.update()
    
    def update_status(self, status_text: str) -> None:
        """Update the status text."""
        self.status_label.config(text=status_text)
        self.splash.update()
    
    def destroy(self) -> None:
        """Close the splash screen."""
        self.progress.stop()
        self.splash.destroy()

# Global instances
startup_timer = StartupTimer()
lazy_importer = LazyImporter()
