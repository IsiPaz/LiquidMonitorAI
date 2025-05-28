import cv2
import numpy as np
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from dataclasses import dataclass

@dataclass
class Detection:
    """Represents a detected container and its liquid level"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    liquid_level: float  # percentage (0-100)
    confidence: float
    method: str

class LiquidLevelEstimator:
    """Base class for liquid level estimation methods"""
    
    def __init__(self, method_name: str):
        self.method_name = method_name
    
    def detect_containers_and_levels(self, image: np.ndarray) -> List[Detection]:
        """Detect containers and estimate liquid levels"""
        raise NotImplementedError
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Common preprocessing steps"""
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

class CannyContourEstimator(LiquidLevelEstimator):
    """Liquid level estimation using Canny edge detection + contour analysis"""
    
    def __init__(self, 
                 canny_low: int = 50, 
                 canny_high: int = 150,
                 min_contour_area: int = 1000,
                 aspect_ratio_range: Tuple[float, float] = (0.3, 3.0)):
        super().__init__("Canny_Contour")
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_contour_area = min_contour_area
        self.aspect_ratio_range = aspect_ratio_range
    
    def detect_containers_and_levels(self, image: np.ndarray) -> List[Detection]:
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Filter by aspect ratio (containers are typically taller than wide)
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                continue
            
            # Estimate liquid level using color analysis within bounding box
            roi = image[y:y+h, x:x+w]
            liquid_level = self._estimate_liquid_level_in_roi(roi, h)
            
            # Calculate confidence based on contour properties
            confidence = min(1.0, area / 10000.0)  # Simple confidence metric
            
            detections.append(Detection(
                bbox=(x, y, x + w, y + h),
                liquid_level=liquid_level,
                confidence=confidence,
                method=self.method_name
            ))
        
        return detections
    
    def _estimate_liquid_level_in_roi(self, roi: np.ndarray, container_height: int) -> float:
        """Estimate liquid level within a region of interest"""
        if roi.size == 0:
            return 0.0
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Create mask for non-background pixels (assuming background is light)
        # This is a simple heuristic - adjust based on your data
        mask = cv2.inRange(hsv, (0, 30, 30), (180, 255, 255))
        
        # Find the topmost and bottommost non-zero pixels
        nonzero_y = np.where(mask.any(axis=1))[0]
        
        if len(nonzero_y) == 0:
            return 0.0
        
        liquid_top = nonzero_y[0]
        liquid_bottom = nonzero_y[-1]
        liquid_height = liquid_bottom - liquid_top
        
        # Calculate percentage
        liquid_percentage = (liquid_height / container_height) * 100
        return min(100.0, max(0.0, liquid_percentage))

class HSVColorSegmentationEstimator(LiquidLevelEstimator):
    """Liquid level estimation using HSV color segmentation"""
    
    def __init__(self, 
                 min_contour_area: int = 1000,
                 morphology_kernel_size: int = 5):
        super().__init__("HSV_Segmentation")
        self.min_contour_area = min_contour_area
        self.morphology_kernel_size = morphology_kernel_size
    
    def detect_containers_and_levels(self, image: np.ndarray) -> List[Detection]:
        detections = []
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define multiple color ranges for different types of liquids
        color_ranges = [
            # Clear/transparent liquids (low saturation)
            ((0, 0, 50), (180, 50, 255)),
            # Colored liquids (higher saturation)
            ((0, 50, 50), (180, 255, 255)),
        ]
        
        all_masks = []
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            all_masks.append(mask)
        
        # Combine all masks
        combined_mask = np.zeros_like(all_masks[0])
        for mask in all_masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((self.morphology_kernel_size, self.morphology_kernel_size), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Estimate liquid level using the segmented region
            roi_mask = combined_mask[y:y+h, x:x+w]
            liquid_level = self._estimate_liquid_level_from_mask(roi_mask)
            
            confidence = min(1.0, area / 5000.0)
            
            detections.append(Detection(
                bbox=(x, y, x + w, y + h),
                liquid_level=liquid_level,
                confidence=confidence,
                method=self.method_name
            ))
        
        return detections
    
    def _estimate_liquid_level_from_mask(self, mask: np.ndarray) -> float:
        """Estimate liquid level from binary mask"""
        if mask.size == 0:
            return 0.0
        
        # Find rows with liquid pixels
        row_sums = np.sum(mask, axis=1)
        liquid_rows = np.where(row_sums > 0)[0]
        
        if len(liquid_rows) == 0:
            return 0.0
        
        # Calculate the percentage of rows with liquid
        total_rows = mask.shape[0]
        liquid_row_count = len(liquid_rows)
        
        liquid_percentage = (liquid_row_count / total_rows) * 100
        return min(100.0, max(0.0, liquid_percentage))

class WatershedSegmentationEstimator(LiquidLevelEstimator):
    """Liquid level estimation using Watershed segmentation"""
    
    def __init__(self, min_contour_area: int = 1000):
        super().__init__("Watershed_Segmentation")
        self.min_contour_area = min_contour_area
    
    def detect_containers_and_levels(self, image: np.ndarray) -> List[Detection]:
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        
        # Find contours from watershed result
        watershed_mask = np.uint8(markers > 1) * 255
        contours, _ = cv2.findContours(watershed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Estimate liquid level
            roi = image[y:y+h, x:x+w]
            liquid_level = self._estimate_liquid_level_watershed(roi, markers[y:y+h, x:x+w])
            
            confidence = min(1.0, area / 8000.0)
            
            detections.append(Detection(
                bbox=(x, y, x + w, y + h),
                liquid_level=liquid_level,
                confidence=confidence,
                method=self.method_name
            ))
        
        return detections
    
    def _estimate_liquid_level_watershed(self, roi: np.ndarray, roi_markers: np.ndarray) -> float:
        """Estimate liquid level using watershed markers"""
        if roi.size == 0:
            return 0.0
        
        # Find different segments in the watershed
        unique_markers = np.unique(roi_markers)
        if len(unique_markers) <= 2:  # Background + 1 segment
            return 50.0  # Default estimate
        
        # Analyze the distribution of segments vertically
        h = roi_markers.shape[0]
        segment_distribution = []
        
        for i in range(h):
            row = roi_markers[i, :]
            unique_in_row = len(np.unique(row[row > 0]))
            segment_distribution.append(unique_in_row)
        
        # Find the transition point (where segments change significantly)
        if len(segment_distribution) == 0:
            return 0.0
        
        # Simple heuristic: assume liquid occupies lower portion
        liquid_rows = sum(1 for x in segment_distribution if x > 0)
        liquid_percentage = (liquid_rows / h) * 100
        
        return min(100.0, max(0.0, liquid_percentage))

def process_image_folder(folder_path: str, 
                        estimator: LiquidLevelEstimator,
                        output_path: str = None,
                        visualize: bool = False) -> Dict:
    """Process all images in a folder and return results"""
    
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise ValueError(f"Folder {folder_path} does not exist")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in folder_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        raise ValueError(f"No image files found in {folder_path}")
    
    results = {
        'method': estimator.method_name,
        'total_images': len(image_files),
        'detections': {},
        'summary': {
            'total_detections': 0,
            'avg_liquid_level': 0.0,
            'avg_confidence': 0.0
        }
    }
    
    all_levels = []
    all_confidences = []
    total_detections = 0
    
    for image_file in image_files:
        print(f"Processing {image_file.name}...")
        
        # Load image
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"Warning: Could not load {image_file}")
            continue
        
        # Detect containers and liquid levels
        detections = estimator.detect_containers_and_levels(image)
        
        # Store results
        results['detections'][image_file.name] = [
            {
                'bbox': det.bbox,
                'liquid_level': det.liquid_level,
                'confidence': det.confidence,
                'method': det.method
            }
            for det in detections
        ]
        
        # Update statistics
        for det in detections:
            all_levels.append(det.liquid_level)
            all_confidences.append(det.confidence)
            total_detections += 1
        
        # Visualize if requested
        if visualize and detections:
            vis_image = image.copy()
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_image, f"{det.liquid_level:.1f}%", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(f"{estimator.method_name} - {image_file.name}", vis_image)
            cv2.waitKey(1000)  # Show for 1 second
    
    if visualize:
        cv2.destroyAllWindows()
    
    # Calculate summary statistics
    if all_levels:
        results['summary']['total_detections'] = total_detections
        results['summary']['avg_liquid_level'] = sum(all_levels) / len(all_levels)
        results['summary']['avg_confidence'] = sum(all_confidences) / len(all_confidences)
    
    # Save results if output path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Traditional CV Baselines for Liquid Level Detection')
    parser.add_argument('--input_folder', type=str, required=True, 
                        help='Path to folder containing test images')
    parser.add_argument('--method', type=str, choices=['canny', 'hsv', 'watershed', 'all'], 
                        default='all', help='Detection method to use')
    parser.add_argument('--output_dir', type=str, default='baseline_results',
                        help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                        help='Show detection results visually')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define estimators
    estimators = {}
    if args.method in ['canny', 'all']:
        estimators['canny'] = CannyContourEstimator()
    if args.method in ['hsv', 'all']:
        estimators['hsv'] = HSVColorSegmentationEstimator()
    if args.method in ['watershed', 'all']:
        estimators['watershed'] = WatershedSegmentationEstimator()
    
    # Process with each estimator
    all_results = {}
    for method_name, estimator in estimators.items():
        print(f"\n{'='*50}")
        print(f"Running {estimator.method_name}")
        print(f"{'='*50}")
        
        output_file = output_dir / f"{method_name}_results.json"
        
        try:
            results = process_image_folder(
                args.input_folder, 
                estimator, 
                output_file,
                args.visualize
            )
            all_results[method_name] = results
            
            # Print summary
            summary = results['summary']
            print(f"\nResults for {estimator.method_name}:")
            print(f"  Total images: {results['total_images']}")
            print(f"  Total detections: {summary['total_detections']}")
            print(f"  Avg liquid level: {summary['avg_liquid_level']:.1f}%")
            print(f"  Avg confidence: {summary['avg_confidence']:.3f}")
            
        except Exception as e:
            print(f"Error processing with {method_name}: {e}")
    
    # Save combined results
    combined_output = output_dir / "combined_results.json"
    with open(combined_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to {output_dir}")

if __name__ == "__main__":
    main()
