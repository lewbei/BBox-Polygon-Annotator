import os
import logging

# Setup logging for global error handling
LOG_FILE = os.path.join(os.getcwd(), "error.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Icon glyphs for toolbar buttons (using simple Unicode emojis)
ICON_UNICODE = {
    "auto_annotate": "⚡",
    "save": "💾",
    "load_model": "📂",
    "export": "📤",
    "mode_box": "⬜",
    "mode_polygon": "🔷",
    "undo": "↶",
    "redo": "↷",
    "zoom_in": "🔍+",
    "zoom_out": "🔍-",
    "shortcuts": "⌨️",
}

PROJECTS_DIR = os.path.join(os.getcwd(), "projects")
os.makedirs(PROJECTS_DIR, exist_ok=True)