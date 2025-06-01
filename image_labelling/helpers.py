import os
import shutil
from tkinter import messagebox

def center_window(win, width, height):
    """
    Centers a Tkinter window 'win' given the desired 'width' and 'height'.
    """
    win.update_idletasks()
    screen_width = win.winfo_screenwidth()
    screen_height = win.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    win.geometry(f"{width}x{height}+{x}+{y}")

def write_annotations_to_file(label_path, bboxes, polygons, image_shape):
    """
    Writes bounding boxes (YOLO format) and polygons (normalized points) to a label file.

    :param label_path: Path to the .txt label file.
    :param bboxes: List of bounding boxes [ (x, y, w, h, class_id), ... ] in pixel coords.
    :param polygons: List of polygons [ {'class_id': int, 'points': [(x1, y1), ...]}, ... ] in pixel coords.
    :param image_shape: (height, width) of the image used for normalization.
    """
    img_h, img_w = image_shape[:2]
    
    with open(label_path, 'w') as label_file:
        # Write bounding boxes
        for x, y, w, h, class_id in bboxes:
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            width_norm = w / img_w
            height_norm = h / img_h
            bbox_line = f"{class_id} {x_center} {y_center} {width_norm} {height_norm}\n"
            label_file.write(bbox_line)

        # Write polygons in YOLO segmentation format (normalized points)
        for poly_data in polygons:
            class_id = poly_data['class_id']
            points = poly_data['points']
            normalized_points = []
            for px, py in points:
                normalized_points.append(px / img_w)
                normalized_points.append(py / img_h)
            polygon_line = f"{class_id} {' '.join(map(str, normalized_points))}\n"
            label_file.write(polygon_line)

def read_annotations_from_file(label_path, image_shape):
    """
    Reads YOLO-format bounding boxes and normalized polygons from a label file
    and converts them to pixel coordinates.

    :param label_path: Path to the .txt label file.
    :param image_shape: (height, width) of the image used for denormalization.
    :return: Tuple (list of bboxes, list of polygons)
             bboxes: [ (x, y, w, h, class_id), ... ] in pixel coords.
             polygons: [ {'class_id': int, 'points': [(x1, y1), ...]}, ... ] in pixel coords.
    """
    import os
    bboxes = []
    polygons = []
    
    img_h, img_w = image_shape[:2]
    if not os.path.exists(label_path):
        return bboxes, polygons
        
    with open(label_path, 'r') as label_file:
        for line in label_file:
            line_content = line.strip()
            if not line_content:
                continue
                
            parts = list(map(float, line_content.split()))
            class_id = int(parts[0])
            coords = parts[1:]
            
            if len(coords) == 4:
                # Bounding box
                x_center, y_center, width, height = coords
                x_center_abs = x_center * img_w
                y_center_abs = y_center * img_h
                width_abs = width * img_w
                height_abs = height * img_h
                x_min = int(x_center_abs - width_abs / 2)
                y_min = int(y_center_abs - height_abs / 2)
                bbox = (x_min, y_min, int(width_abs), int(height_abs), class_id)
                bboxes.append(bbox)
            elif len(coords) % 2 == 0 and len(coords) >= 6:
                # Polygon
                points = []
                for i in range(0, len(coords), 2):
                    px_norm = coords[i]
                    py_norm = coords[i+1]
                    point = (int(px_norm * img_w), int(py_norm * img_h))
                    points.append(point)
                polygon = {'class_id': class_id, 'points': points}
                polygons.append(polygon)
                
    return bboxes, polygons

def copy_files_recursive(file_list_relative_paths, base_images_src_dir, images_dst_base,
                         base_labels_src_dir, labels_dst_base):
    """
    Copies images and their corresponding label files, preserving subdirectory structure.

    :param file_list_relative_paths: List of image file paths relative to base_images_src_dir.
    :param base_images_src_dir: The root directory where source images are located.
    :param images_dst_base: The base destination directory for images.
    :param base_labels_src_dir: The root directory where source labels are located.
    :param labels_dst_base: The base destination directory for labels.
    """
    for relative_path in file_list_relative_paths:
        src_image_path = os.path.join(base_images_src_dir, relative_path)

        label_relative_path = os.path.splitext(relative_path)[0] + '.txt'
        src_label_path = os.path.join(base_labels_src_dir, label_relative_path)

        dst_image_path = os.path.join(images_dst_base, relative_path)
        dst_label_path = os.path.join(labels_dst_base, label_relative_path)

        os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(dst_label_path), exist_ok=True)

        try:
            shutil.copy(src_image_path, dst_image_path)
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dst_label_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export {relative_path}:\n{e}")
