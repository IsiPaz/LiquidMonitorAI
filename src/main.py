import os
import cv2
import time
import json
import psutil
import random
import logging
import datetime
import torch
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from ultralytics import YOLO

# Custom utility imports (you must provide these)
from utils.colors import drink_color_risk, color_reference_rgb


# Global result storage
id_colors = {}
fill_level_data = {}
severities_data = {}

# --- Utility Functions ---

def get_color_for_id(track_id):
    """Assign a consistent color to each track ID."""
    if track_id not in id_colors:
        id_colors[track_id] = [random.randint(0, 255) for _ in range(3)]
    return id_colors[track_id]

def get_average_color(frame, mask):
    """Compute average color in masked region."""
    masked_pixels = frame[mask > 0]
    return masked_pixels.mean(axis=0) if len(masked_pixels) else None

def find_closest_color_name(avg_color):
    """Find the nearest predefined color name."""
    color_names = list(color_reference_rgb.keys())
    color_values = np.array(list(color_reference_rgb.values()))
    closest_idx = pairwise_distances_argmin([avg_color], color_values)
    return color_names[closest_idx[0]]

def calculate_fill_percentage(container_box, liquid_box):
    """Estimate liquid fill percentage inside the container."""
    container_top = container_box[1]
    liquid_top = max(liquid_box[1], container_top)
    liquid_bottom = liquid_box[3]
    max_fill_height = max(liquid_bottom - container_top, 1e-5)
    fill_height = max(0, min(liquid_bottom - liquid_top, max_fill_height))
    return (fill_height / max_fill_height) * 100

# --- YOLO Model Loader ---

class YOLOLoader:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_path = model_path
        self.device = device
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.set_device(0)
            logging.info("Using CUDA device")
        else:
            self.device = 'cpu'
            logging.warning("CUDA unavailable. Switching to CPU.")
        self.model = self._load_model()

    def _load_model(self):
        try:
            model = YOLO(self.model_path)
            logging.info(f"Model loaded from: {self.model_path}")
            return model
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return None

    def track(self, frame, conf=0.25):
        if not self.model:
            logging.warning("Model not initialized.")
            return None
        try:
            return self.model.track(source=frame, conf=conf, device=self.device, persist=True, verbose=False)
        except Exception as e:
            logging.error(f"Tracking failed: {e}")
            return None

# --- Frame Processing ---

def process_frame(frame, results, image_name=None, force_id=False):
    has_container = False
    original_frame = frame.copy()

    if results:
        for r in results:
            if r.masks is None or r.boxes is None:
                continue

            masks = r.masks.data.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else (np.arange(len(boxes)) if force_id else [None] * len(boxes))
            classes = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            containers = []
            liquids = []

            for i, (mask, box, cls_id, conf_score, track_id) in enumerate(zip(masks, boxes, classes, confs, ids)):
                x1, y1, x2, y2 = map(int, box)
                class_name = r.names[int(cls_id)].lower()
                label = f"{class_name} {conf_score:.2f}"
                if track_id is not None:
                    label += f" ID:{int(track_id)}"

                if class_name in ['bottle', 'glass', 'wine-glass']:
                    has_container = True
                    containers.append((box, class_name, conf_score, track_id))
                elif class_name == 'liquid':
                    liquids.append(box)

                # Prepare binary mask
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_bin = (mask_resized > 0.5).astype(np.uint8)

                # Analyze color
                if class_name == 'liquid':
                    avg_color = get_average_color(original_frame, mask_bin)
                    if avg_color is not None:
                        color_name = find_closest_color_name(avg_color)
                        risk_level = drink_color_risk.get(color_name, "unknown")
                        info_text = f"{color_name} ({risk_level})"
                        cv2.putText(frame, info_text, (x1, y2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        # Save severity with x1 for sorting
                        if image_name and track_id is not None:
                            severities_data.setdefault(image_name, []).append({
                                'container_id': int(track_id),
                                'color': color_name,
                                'severity': risk_level,
                                'x1': x1
                            })
                # Visualization
                color = get_color_for_id(int(track_id) if track_id is not None else i)
                color_mask = np.zeros_like(frame)
                for c in range(3):
                    color_mask[:, :, c] = mask_bin * color[c]
                frame = cv2.addWeighted(frame, 1.0, color_mask, 0.5, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Compute fill levels
            for container_box, container_name, conf_score, track_id in containers:
                matched = False
                cx1, cy1, cx2, cy2 = map(int, container_box)
                container_area = (cx2 - cx1) * (cy2 - cy1)

                for liquid_box in liquids:
                    lx1, ly1, lx2, ly2 = map(int, liquid_box)
                    inter_area = max(0, min(cx2, lx2) - max(cx1, lx1)) * max(0, min(cy2, ly2) - max(cy1, ly1))
                    if container_area > 0 and (inter_area / container_area) > 0.1:
                        fill = calculate_fill_percentage(container_box, liquid_box)
                        fill_text = f"Fill: {fill:.1f}%"
                        cv2.putText(frame, fill_text, (cx1, cy2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        matched = True
                        if image_name and track_id is not None:
                            fill_level_data.setdefault(image_name, []).append({
                                'container_id': int(track_id),
                                'class': container_name,
                                'conf': float(conf_score),
                                'liquid_percentage': float(fill),
                                'x1': cx1
                            })
                        break

                if not matched and image_name and track_id is not None:
                    fill_level_data.setdefault(image_name, []).append({
                        'container_id': int(track_id),
                        'class': container_name,
                        'conf': float(conf_score),
                        'liquid_percentage': 0.0,
                        'x1': cx1
                    })

            # Ensure containers with no liquids still create entries
            if image_name and has_container and image_name not in severities_data: #tiene contenedor pero no liquido
                severities_data[image_name] = [{
                    "container_id": 0,
                    "color": None,
                    "severity": None
                }]

            # Sort all detections left to right by x1
            if image_name in fill_level_data:
                fill_level_data[image_name].sort(key=lambda x: x.get('x1', 0))
            if image_name in severities_data:
                severities_data[image_name].sort(key=lambda x: x.get('x1', 0))

    return frame, has_container

# --- Output Handling ---

def save_json_outputs(output_dir):
    """Save detection results to JSON files, removing x1 field for clean output."""
    os.makedirs(output_dir, exist_ok=True)

    def clean_entries(entries):
        for entry in entries:
            entry.pop('x1', None)
        return entries

    sorted_fill = {k: clean_entries(v) for k, v in fill_level_data.items()}
    sorted_severities = {k: clean_entries(v) for k, v in severities_data.items()}

    with open(os.path.join(output_dir, "fill_level.json"), "w") as f:
        json.dump(sorted_fill, f, indent=2)

    with open(os.path.join(output_dir, "severities.json"), "w") as f:
        json.dump(sorted_severities, f, indent=2)

def save_performance_metrics(output_dir, source, frame_count, total_time, total_infer_time, image_name):
    """Save FPS, inference time, RAM, VRAM usage to a text file."""
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_infer_time = total_infer_time / frame_count if frame_count > 0 else 0
    memory_usage_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    vram_usage_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0

    os.makedirs(output_dir, exist_ok=True)
    stats_file = os.path.join(output_dir, f"{image_name}_performance_metrics.txt")
    with open(stats_file, "w") as f:
        f.write(f"Processed source: {source}\n")
        f.write(f"Total frames: {frame_count}\n")
        f.write(f"Total time: {total_time:.2f} sec\n")
        f.write(f"Average FPS: {avg_fps:.2f}\n")
        f.write(f"Average inference time: {avg_infer_time:.4f} sec\n")
        f.write(f"RAM usage: {memory_usage_mb:.2f} MB\n")
        f.write(f"VRAM usage: {vram_usage_mb:.2f} MB\n")
        f.write(f"Completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# --- Main Detection Function ---

def run_detection(source, model_loader: YOLOLoader, conf=0.25, save_output=False, output_path=None, json_output_dir="output/json"):
    image_name = os.path.basename(source) if isinstance(source, str) else "webcam_frame"
    process_start = time.time()
    total_infer_time = 0
    frame_count = 0

    # Webcam or video
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        cap = cv2.VideoCapture(int(source))
        is_stream = True
    elif source.lower().endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(source)
        is_stream = False
    else:
        # Image input
        frame = cv2.imread(source)
        results = model_loader.track(frame, conf)
        processed, has_container = process_frame(frame, results, image_name=image_name, force_id=True)
        if not has_container and image_name not in severities_data: #imagen sin contenedor y sin liquido
            fill_level_data[image_name] = [{
                 "container_id": 0, "class": None, "conf": 0, "liquid_percentage": 0
            }]
            severities_data[image_name] = [{
                "container_id": 0,
                "color": None,
                "severity": None
            }]
        elif not has_container and image_name in severities_data: #imagen sin contenedor pero con liquido
            fill_level_data[image_name] = [{
                 "container_id": 0, "class": None, "conf": 0, "liquid_percentage": 0
            }]

        if save_output and output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, processed)
            save_json_outputs(json_output_dir)

        cv2.imshow("Processed Image", processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save performance metrics for image
        total_time = time.time() - process_start
        total_infer_time = total_time  # For one image, total time = infer time
        frame_count = 1

        save_performance_metrics(json_output_dir, source, frame_count, total_time, total_infer_time, image_name)
        return

    if not cap.isOpened():
        logging.error("Cannot open video/camera source.")
        return

    print(save_output, is_stream, output_path)
    if save_output and not is_stream and output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width, height = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
    else:
        out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_infer = time.time()
        results = model_loader.track(frame, conf)
        infer_time = time.time() - start_infer

        total_infer_time += infer_time
        frame_count += 1

        processed, _ = process_frame(frame, results)

        if save_output and out:
            out.write(processed)

        cv2.imshow("Processed Video", processed)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if out:
        out.release()

    cv2.destroyAllWindows()
    save_json_outputs(json_output_dir)

    process_total = time.time() - process_start
    save_performance_metrics(json_output_dir, source, frame_count, process_total, total_infer_time, image_name)

# --- Entry Point ---

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    #=== USER CONFIGURATION ===
    input_source = "C:/Users/Isidora/Downloads/test_image.jpg"  # Path to image/video or "0" for webcam
    save_predictions = True

    # Output paths
    base_output_path = "test/"
    output_path_pred = os.path.join(base_output_path, "test6_processed.mp4")
    output_folder_json = os.path.join(base_output_path, "json")

    # Model path
    model_path = "weights/lmai-11n-seg.pt"

    # === Ensure output directories exist ===
    os.makedirs(base_output_path, exist_ok=True)
    os.makedirs(output_folder_json, exist_ok=True)
    #===========================

    loader = YOLOLoader(model_path, device='cuda')
    run_detection(
        source=input_source,
        model_loader=loader,
        conf=0.5,
        save_output=save_predictions,
        output_path=output_path_pred if save_predictions else None,
        json_output_dir=output_folder_json
    )


""" if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # === USER CONFIGURATION ===
    input_folder = "C:/Users/Isidora/Downloads/test/test"  # Carpeta con imágenes
    output_folder = "C:/Users/Isidora/Desktop/proyectos-isi/liquid/output2/medium"
    model_path = "./weight/yolo11m/best.pt"
    # ===========================

    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Cargar modelo
    loader = YOLOLoader(model_path, device='cuda')

    # Obtener lista de imágenes
    image_paths = glob.glob(os.path.join(input_folder, "*.*"))  # Lee todos los archivos (puedes filtrar por .jpg, .png, etc.)

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        output_image_path = os.path.join(output_folder, filename)

        run_detection(
            source=image_path,
            model_loader=loader,
            conf=0.5,
            save_output=True,
            output_path=output_image_path,
            json_output_dir=output_folder  # JSONs se guardarán en la misma carpeta que las imágenes
        ) """