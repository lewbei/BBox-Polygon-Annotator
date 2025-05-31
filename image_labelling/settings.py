"""
Settings persistence utilities for window geometry and theme.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict

_SCHEMA_VERSION = 1
_DEFAULT_SETTINGS: Dict[str, Any] = {
    "schema_version": _SCHEMA_VERSION,
    "geometry": None,
    "theme": None,
}

def get_settings_path() -> Path:
    return Path.home() / ".bbox_annotator_settings.json"

def load_settings() -> Dict[str, Any]:
    path = get_settings_path()
    if not path.exists():
        return _DEFAULT_SETTINGS.copy()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("schema_version") != _SCHEMA_VERSION:
            logging.warning("Settings schema mismatch or invalid, loading defaults")
            return _DEFAULT_SETTINGS.copy()
        settings = _DEFAULT_SETTINGS.copy()
        settings.update(data)
        return settings
    except Exception as e:
        logging.error("Failed to load settings from %s: %s", path, e, exc_info=True)
        return _DEFAULT_SETTINGS.copy()

def save_settings(settings: Dict[str, Any]) -> None:
    path = get_settings_path()
    to_save = dict(settings)
    to_save["schema_version"] = _SCHEMA_VERSION
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=4)
    except Exception as e:
        logging.error("Failed to save settings to %s: %s", path, e, exc_info=True)