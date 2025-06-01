"""
Model Analysis Utility for Auto-Annotation Feature
This module provides functionality to analyze YOLO models and determine their capabilities.
"""

import os
import re
from typing import Dict, List, Tuple, Optional

class ModelAnalyzer:
    """Analyzes YOLO models to determine their capabilities and annotation types."""
    
    def __init__(self):
        self.model_info = {}
    
    def analyze_model(self, model_path: str, model_instance=None) -> Dict:
        """
        Analyze a YOLO model to determine its capabilities.
        
        Args:
            model_path: Path to the model file
            model_instance: Optional loaded model instance
            
        Returns:
            Dictionary containing model analysis results
        """
        if not model_path or not os.path.exists(model_path):
            return self._create_error_result("Model file not found")
        
        analysis = {
            "model_path": model_path,
            "model_name": os.path.basename(model_path),
            "model_type": "Unknown",
            "task_type": "detect",
            "supports_detection": False,
            "supports_segmentation": False,
            "recommended_annotation": "bounding_boxes",
            "confidence": 0.0,
            "classes": [],
            "input_size": (640, 640),
            "error": None
        }
        
        try:
            # File name analysis
            analysis.update(self._analyze_filename(model_path))
            
            # Model instance analysis (if model is loaded)
            if model_instance:
                analysis.update(self._analyze_model_instance(model_instance))
            
            # Set final recommendations
            analysis.update(self._set_recommendations(analysis))
            
        except Exception as e:
            analysis["error"] = str(e)
            return analysis
        
        return analysis
    
    def _analyze_filename(self, model_path: str) -> Dict:
        """Analyze model filename to determine type."""
        filename = os.path.basename(model_path).lower()
        analysis = {}
        
        # Check for segmentation models
        if "-seg" in filename or "seg" in filename:
            analysis["model_type"] = "Segmentation"
            analysis["task_type"] = "segment"
            analysis["supports_detection"] = True
            analysis["supports_segmentation"] = True
            analysis["confidence"] = 0.9
        
        # Check for detection models
        elif any(pattern in filename for pattern in ["yolo", "detect", ".pt"]):
            analysis["model_type"] = "Detection"
            analysis["task_type"] = "detect"
            analysis["supports_detection"] = True
            analysis["supports_segmentation"] = False
            analysis["confidence"] = 0.8
        
        # Extract model size/variant
        size_patterns = ["n", "s", "m", "l", "x"]
        for size in size_patterns:
            if f"yolov8{size}" in filename:
                analysis["model_variant"] = f"YOLOv8{size.upper()}"
                break
        
        return analysis
    
    def _analyze_model_instance(self, model_instance) -> Dict:
        """Analyze loaded model instance for detailed information."""
        analysis = {}
        
        try:
            # Get model task if available
            if hasattr(model_instance, 'task'):
                analysis["task_type"] = model_instance.task
                analysis["confidence"] = 0.95
                
                if model_instance.task == "segment":
                    analysis["model_type"] = "Segmentation"
                    analysis["supports_detection"] = True
                    analysis["supports_segmentation"] = True
                elif model_instance.task == "detect":
                    analysis["model_type"] = "Detection"
                    analysis["supports_detection"] = True
                    analysis["supports_segmentation"] = False
            
            # Get class names if available
            if hasattr(model_instance, 'names'):
                analysis["classes"] = list(model_instance.names.values())
            elif hasattr(model_instance, 'model') and hasattr(model_instance.model, 'names'):
                analysis["classes"] = list(model_instance.model.names.values())
            
            # Get input size if available
            if hasattr(model_instance, 'imgsz'):
                size = model_instance.imgsz
                if isinstance(size, (list, tuple)) and len(size) >= 2:
                    analysis["input_size"] = (size[0], size[1])
                elif isinstance(size, int):
                    analysis["input_size"] = (size, size)
        
        except Exception as e:
            # Model instance analysis failed, but that's okay
            pass
        
        return analysis
    
    def _set_recommendations(self, analysis: Dict) -> Dict:
        """Set annotation recommendations based on analysis."""
        recommendations = {}
        
        if analysis.get("supports_segmentation") and analysis.get("supports_detection"):
            # Segmentation models can do both
            recommendations["recommended_annotation"] = "segmentation"
            recommendations["available_options"] = [
                ("segmentation", "Segmentation Masks (Recommended)", True),
                ("bounding_boxes", "Bounding Boxes Only", False),
                ("both", "Both Masks + Boxes", False)
            ]
        elif analysis.get("supports_detection"):
            # Detection only models
            recommendations["recommended_annotation"] = "bounding_boxes"
            recommendations["available_options"] = [
                ("bounding_boxes", "Bounding Boxes (Only Option)", True)
            ]
        else:
            # Unknown or unsupported
            recommendations["recommended_annotation"] = "bounding_boxes"
            recommendations["available_options"] = [
                ("bounding_boxes", "Bounding Boxes (Default)", True),
                ("segmentation", "Segmentation (May not work)", False)
            ]
        
        return recommendations
    
    def _create_error_result(self, error_msg: str) -> Dict:
        """Create error result dictionary."""
        return {
            "model_path": "",
            "model_name": "Unknown",
            "model_type": "Error",
            "task_type": "detect",
            "supports_detection": False,
            "supports_segmentation": False,
            "recommended_annotation": "bounding_boxes",
            "confidence": 0.0,
            "classes": [],
            "input_size": (640, 640),
            "error": error_msg,
            "available_options": [("bounding_boxes", "Bounding Boxes (Default)", True)]
        }
    
    def get_model_description(self, analysis: Dict) -> str:
        """Generate human-readable model description."""
        if analysis.get("error"):
            return f"Error: {analysis['error']}"
        
        model_name = analysis.get("model_name", "Unknown")
        model_type = analysis.get("model_type", "Unknown")
        
        if analysis.get("supports_segmentation") and analysis.get("supports_detection"):
            capabilities = "Detection + Segmentation"
        elif analysis.get("supports_detection"):
            capabilities = "Detection Only"
        elif analysis.get("supports_segmentation"):
            capabilities = "Segmentation Only"
        else:
            capabilities = "Unknown Capabilities"
        
        class_count = len(analysis.get("classes", []))
        if class_count > 0:
            class_info = f" • {class_count} classes"
        else:
            class_info = ""
        
        return f"{model_name}\nType: {model_type}\nCapabilities: {capabilities}{class_info}"
