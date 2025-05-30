import os
import json
import csv
import xml.etree.ElementTree as ET
from datetime import datetime

def convert_to_coco_format(image_files, all_bboxes, all_polygons, class_names, base_folder):
    """
    Converts annotations to COCO format.
    Returns a COCO-formatted dictionary.
    """
    coco_data = {
        "info": {
            "description": "Dataset exported from BBox & Polygon Annotator",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "BBox & Polygon Annotator v9",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }

    for i, class_name in enumerate(class_names):
        coco_data["categories"].append({
            "id": i,
            "name": class_name,
            "supercategory": "object"
        })

    annotation_id = 1
    for img_idx, image_path in enumerate(image_files):
        full_image_path = os.path.join(base_folder, image_path)
        if os.path.exists(full_image_path):
            import cv2
            img = cv2.imread(full_image_path)
            height, width = img.shape[:2] if img is not None else (480, 640)
        else:
            width, height = 640, 480

        coco_data["images"].append({
            "id": img_idx,
            "width": width,
            "height": height,
            "file_name": os.path.basename(image_path)
        })

        if image_path in all_bboxes:
            for bbox in all_bboxes[image_path]:
                x, y, w, h, class_id = bbox
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_idx,
                    "category_id": class_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                annotation_id += 1

        if image_path in all_polygons:
            for polygon in all_polygons[image_path]:
                class_id = polygon['class_id']
                points = polygon['points']

                segmentation = []
                for x, y in points:
                    segmentation.extend([float(x), float(y)])

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bbox_w, bbox_h = x_max - x_min, y_max - y_min
                area = bbox_w * bbox_h

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_idx,
                    "category_id": class_id,
                    "segmentation": [segmentation],
                    "bbox": [x_min, y_min, bbox_w, bbox_h],
                    "area": area,
                    "iscrowd": 0
                })
                annotation_id += 1

    return coco_data

def convert_to_pascal_voc_format(image_path, bboxes, polygons, class_names, image_shape):
    """
    Converts annotations for a single image to Pascal VOC XML format.
    Returns XML string.
    """
    height, width = image_shape[:2]

    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = os.path.dirname(image_path) or "images"

    filename = ET.SubElement(annotation, "filename")
    filename.text = os.path.basename(image_path)

    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "BBox & Polygon Annotator"

    size = ET.SubElement(annotation, "size")
    width_elem = ET.SubElement(size, "width")
    width_elem.text = str(width)
    height_elem = ET.SubElement(size, "height")
    height_elem.text = str(height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "1" if polygons else "0"

    for bbox in bboxes:
        x, y, w, h, class_id = bbox

        obj = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj, "name")
        name.text = class_names[class_id]

        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"

        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"

        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(x))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(y))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(x + w))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(y + h))

    for polygon in polygons:
        class_id = polygon['class_id']
        points = polygon['points']

        obj = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj, "name")
        name.text = class_names[class_id]

        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"

        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"

        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(x_min))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(y_min))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(x_max))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(y_max))

        polygon_elem = ET.SubElement(obj, "polygon")
        for i, (px, py) in enumerate(points):
            point = ET.SubElement(polygon_elem, f"point{i+1}")
            point.set("x", str(int(px)))
            point.set("y", str(int(py)))

    return ET.tostring(annotation, encoding='unicode')

def convert_to_csv_format(image_files, all_bboxes, all_polygons, class_names):
    """
    Converts annotations to CSV format.
    Returns list of rows for CSV writing.
    """
    rows = []
    headers = ["image_name", "annotation_type", "class_name", "class_id", "coordinates", "area"]
    rows.append(headers)

    for image_path in image_files:
        image_name = os.path.basename(image_path)

        if image_path in all_bboxes:
            for bbox in all_bboxes[image_path]:
                x, y, w, h, class_id = bbox
                coordinates = f"x={x},y={y},w={w},h={h}"
                area = w * h
                rows.append([
                    image_name, "bbox", class_names[class_id], class_id,
                    coordinates, area
                ])

        if image_path in all_polygons:
            for polygon in all_polygons[image_path]:
                class_id = polygon['class_id']
                points = polygon['points']
                coordinates = ";".join([f"{x},{y}" for x, y in points])

                if len(points) >= 3:
                    area = 0.5 * abs(sum(points[i][0] * (points[(i+1) % len(points)][1] - points[i-1][1])
                                       for i in range(len(points))))
                else:
                    area = 0

                rows.append([
                    image_name, "polygon", class_names[class_id], class_id,
                    coordinates, area
                ])

    return rows