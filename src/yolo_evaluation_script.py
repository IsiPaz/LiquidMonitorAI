#!/usr/bin/env python
# YOLOv11 Evaluation Script with COCO Metrics
# This script evaluates both base and fine-tuned YOLOv11 models against test images using COCO metrics

import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import argparse
import glob
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import cv2
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def create_coco_predictions(model, test_images_dir, annotations_file, output_dir, is_base_model=False):
    """
    Creates COCO format predictions for a model on test images
    
    Args:
        model: YOLO model object
        test_images_dir: Directory containing test images
        annotations_file: Path to COCO format annotations
        output_dir: Directory to save results
        is_base_model: Whether this is a base model or fine-tuned model
    
    Returns:
        List of prediction dictionaries in COCO format
    """
    # Load COCO annotations
    coco_gt = COCO(annotations_file)
    
    # Get image IDs from annotations
    image_ids = list(coco_gt.imgs.keys())
    image_info = coco_gt.loadImgs(image_ids)
    
    # Get categories from annotations
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    coco_category_names = {cat['id']: cat['name'] for cat in categories}
    
    # Define category mapping for base models
    # This maps from the base model's class names to the COCO category IDs in your dataset
    base_model_mapping = {
        'wine glass': 4,  # wine-glass in your COCO dataset
        'bottle': 1,      # bottle in your COCO dataset
        'cup': 2,         # glass in your COCO dataset
        0: 1,             # Fallback mappings by index 
        39: 1,            # bottle in COCO
        41: 2,            # cup/glass in COCO
        44: 4,            # wine glass in COCO
        'liquid': 3       # liquid - may not be in base model
    }
    
    # Print model's class names for debugging
    print(f"Model class names: {model.names}")
    print(f"COCO categories: {coco_category_names}")
    
    # Prepare predictions list
    predictions = []
    
    # Process each image
    for img_info in tqdm(image_info, desc=f"Generating predictions"):
        # Get image path
        img_path = os.path.join(test_images_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping")
            continue
        
        # Run inference
        results = model(img_path)
        
        # Extract detections
        boxes = results[0].boxes
        for i in range(len(boxes)):
            # Get box coordinates (xmin, ymin, xmax, ymax)
            box = boxes.xyxy[i].cpu().numpy()
            
            # Convert to COCO format (x, y, width, height)
            x, y, x2, y2 = box
            width = x2 - x
            height = y2 - y
            
            # Get class and confidence
            class_id = int(boxes.cls[i].item())
            class_name = model.names[class_id]
            score = float(boxes.conf[i].item())
            
            # Map the class ID to the proper category ID in your COCO dataset
            if is_base_model:
                # For base models, map the class name or ID to your dataset's category ID
                if class_name in base_model_mapping:
                    coco_cat_id = base_model_mapping[class_name]
                elif class_id in base_model_mapping:
                    coco_cat_id = base_model_mapping[class_id]
                else:
                    print(f"Warning: Unrecognized class {class_name} (id: {class_id}), skipping")
                    continue
            else:
                # For fine-tuned models, add 1 because COCO categories start from 1
                coco_cat_id = class_id + 1
            
            # Create prediction dictionary
            pred = {
                'image_id': img_info['id'],
                'category_id': coco_cat_id,
                'bbox': [float(x), float(y), float(width), float(height)],
                'score': score
            }
            
            predictions.append(pred)
    
    return predictions

def evaluate_model_coco(model_path, test_images_dir, annotations_file, output_dir):
    """
    Evaluates a YOLO model using COCO metrics
    
    Args:
        model_path: Path to the model file
        test_images_dir: Directory containing test images
        annotations_file: Path to COCO format annotations
        output_dir: Directory to save results
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Load model
    try:
        model = YOLO(model_path)
        print(f"Successfully loaded {model_path}")
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None
    
    # Get model name
    model_name = os.path.basename(model_path).split('.')[0]
    is_base = not model_path.startswith("runs/")
    model_display_name = f"{model_name} {('(base)' if is_base else '')}".strip()
    
    # Create predictions
    predictions = create_coco_predictions(model, test_images_dir, annotations_file, output_dir, is_base_model=is_base)
    
    # If no predictions were generated, return empty metrics
    if not predictions:
        print(f"No predictions generated for {model_path}")
        return {
            'model': model_display_name,
            'mAP@0.5': 0.0,
            'mAP@0.5:0.95': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'avg_inference_time_ms': 0.0
        }
    
    # Save predictions to file
    pred_file = os.path.join(output_dir, f"{model_name}_predictions.json")
    with open(pred_file, 'w') as f:
        json.dump(predictions, f)
    
    print(f"Saved {len(predictions)} predictions to {pred_file}")
    
    # Load COCO ground truth
    coco_gt = COCO(annotations_file)
    
    # Create COCO format for predictions
    try:
        coco_dt = coco_gt.loadRes(pred_file)
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return {
            'model': model_display_name,
            'mAP@0.5': 0.0,
            'mAP@0.5:0.95': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'avg_inference_time_ms': 0.0
        }
    
    # Create evaluator
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Set evaluation parameters
    coco_eval.params.maxDets = [100, 300, 1000]  # Adjust as needed
    
    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract metrics
    metrics = {
        'model': model_display_name,
        'mAP@0.5': coco_eval.stats[1],  # AP at IoU=0.50
        'mAP@0.5:0.95': coco_eval.stats[0],  # AP at IoU=0.50:0.95
        'precision': coco_eval.stats[8],  # Precision
        'recall': coco_eval.stats[10],  # Recall
    }
    
    # Also measure inference time
    image_files = glob.glob(os.path.join(test_images_dir, '*.jpg'))[:10]  # Use 10 images for timing
    inference_times = []
    
    for img_path in image_files:
        start_time = time.time()
        model(img_path)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        inference_times.append(inference_time)
    
    metrics['avg_inference_time_ms'] = np.mean(inference_times)
    
    return metrics

def evaluate_models(test_images_dir, annotations_file, output_dir):
    """
    Evaluates both base and fine-tuned YOLOv11 models
    
    Args:
        test_images_dir: Directory containing test images
        annotations_file: Path to COCO format annotations
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Base models
    base_models = [
        'yolov11n-seg.pt',  # Nano
        'yolov11s-seg.pt',  # Small
        'yolov11m-seg.pt',  # Medium
    ]
    
    # Fine-tuned models - update these paths to match your fine-tuned model locations
    fine_tuned_models = [
        'runs/segment/train/weights/best.pt',  # Nano fine-tuned
        'runs/segment/train2/weights/best.pt',  # Small fine-tuned
        'runs/segment/train3/weights/best.pt',  # Medium fine-tuned
    ]
    
    # Parse the COCO annotations to get class information
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    print("\nCOCO Categories in the dataset:")
    for cat in coco_data['categories']:
        print(f"ID: {cat['id']}, Name: {cat['name']}")
    
    # Check annotation counts
    annotations_count = {}
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        if cat_id not in annotations_count:
            annotations_count[cat_id] = 0
        annotations_count[cat_id] += 1
    
    print("\nAnnotation counts by category:")
    for cat in coco_data['categories']:
        count = annotations_count.get(cat['id'], 0)
        print(f"Category {cat['name']} (ID: {cat['id']}): {count} annotations")
    
    # Evaluate all models
    all_models = base_models + fine_tuned_models
    results = []
    
    for model_path in all_models:
        if os.path.exists(model_path):
            print(f"\n{'='*50}\nEvaluating {model_path}\n{'='*50}")
            metrics = evaluate_model_coco(model_path, test_images_dir, annotations_file, output_dir)
            if metrics:
                results.append(metrics)
        else:
            print(f"Model file not found: {model_path}")
    
    # Create results DataFrame
    if results:
        results_df = pd.DataFrame(results)
        
        # Save results to CSV
        results_csv = os.path.join(output_dir, 'model_evaluation_results.csv')
        results_df.to_csv(results_csv, index=False)
        print(f"\nResults saved to {results_csv}")
        
        # Print results table
        print("\nModel Performance:")
        print(results_df.to_string(index=False))
        
        # Generate LaTeX table
        latex_table = results_df.to_latex(index=False, float_format="%.3f")
        latex_file = os.path.join(output_dir, 'model_evaluation_table.tex')
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to {latex_file}")
        
        # Return for further processing if needed
        return results_df
    else:
        print("No results obtained from any model.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate YOLOv11 models using COCO metrics')
    parser.add_argument('--images', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--annotations', type=str, required=True,
                        help='Path to COCO format annotations JSON file')
    parser.add_argument('--output', type=str, default='./output/evaluation',
                        help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_models(args.images, args.annotations, args.output)