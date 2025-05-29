"""
Model Evaluation Framework

This framework provides a flexible and extensible system for evaluating computer vision models
in liquid level estimation and anomaly detection. It supports multiple metrics,
visualization generation, and customizable evaluation strategies.

Features:
- Multiple evaluation types
- Configurable label mappings and risk categorizations
- Confusion matrix generation with PDF export
- Extensible metric computation
- Robust error handling and logging
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_recall_fscore_support
)


class DataLoader:
    """Handles loading and parsing of JSON data files."""
    
    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """
        Load JSON data from file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary containing the loaded JSON data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {e}")


class LabelProcessor:
    """Handles label transformations, filtering, and mapping operations."""
    
    def __init__(self, label_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize the label processor.
        
        Args:
            label_mapping: Dictionary mapping original labels to transformed labels
        """
        self.label_mapping = label_mapping or {}
    
    def transform_labels(self, labels: List[str]) -> List[str]:
        """
        Transform labels using the provided mapping.
        
        Args:
            labels: List of original labels
            
        Returns:
            List of transformed labels
        """
        return [self.label_mapping.get(label, label) for label in labels]
    
    def find_unknown_labels(self, labels: List[str], known_labels: set) -> set:
        """
        Find labels that are not in the known labels set.
        
        Args:
            labels: List of labels to check
            known_labels: Set of known/valid labels
            
        Returns:
            Set of unknown labels
        """
        return set(label for label in labels if label not in known_labels)
    
    def filter_labels(self, y_true: List[str], y_pred: List[str], 
                     exclude_labels: set) -> Tuple[List[str], List[str]]:
        """
        Filter out specified labels from true and predicted label lists.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            exclude_labels: Set of labels to exclude
            
        Returns:
            Tuple of filtered (y_true, y_pred) lists
        """
        filtered_true = []
        filtered_pred = []
        
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label not in exclude_labels and pred_label not in exclude_labels:
                filtered_true.append(true_label)
                filtered_pred.append(pred_label)
                
        return filtered_true, filtered_pred


class VisualizationManager:
    """Handles the creation and saving of evaluation visualizations."""
    
    def __init__(self, output_directory: str = "evaluation_results"):
        """
        Initialize the visualization manager.
        
        Args:
            output_directory: Directory to save visualization files
        """
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)
    
    def save_confusion_matrix(self, y_true: List[str], y_pred: List[str], 
                            model_name: str, label_type: str = "classification",
                            figsize: Tuple[int, int] = (10, 8)) -> str:
        """
        Generate and save confusion matrix as PDF.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model being evaluated
            label_type: Type of labels (for filename and title)
            figsize: Figure size for the plot
            
        Returns:
            Path to the saved PDF file
        """
        # Get unique labels and create confusion matrix
        labels = sorted(set(y_true + y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Calculate percentages with zero-division handling
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm_percentage = cm.astype('float') / row_sums * 100
        
        # Create DataFrame for better visualization
        cm_df = pd.DataFrame(cm_percentage, index=labels, columns=labels)
        
        # Create and customize the plot
        plt.figure(figsize=figsize)
        sns.heatmap(cm_df, annot=True, fmt='.1f', cmap='Blues', 
                   cbar=True, square=True)
        plt.title(f"Confusion Matrix ({label_type.title()}) - {model_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Save the plot
        filename = f"{model_name}_confusion_matrix_{label_type}.pdf"
        filepath = os.path.join(self.output_directory, filename)
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        return filepath


class BaseEvaluator(ABC):
    """Abstract base class for model evaluators."""
    
    def __init__(self, data_loader: DataLoader, label_processor: LabelProcessor,
                 visualization_manager: VisualizationManager):
        """
        Initialize the base evaluator.
        
        Args:
            data_loader: Instance for loading data files
            label_processor: Instance for processing labels
            visualization_manager: Instance for creating visualizations
        """
        self.data_loader = data_loader
        self.label_processor = label_processor
        self.visualization_manager = visualization_manager
    
    @abstractmethod
    def compute_predictions_and_labels(self, model_file: str, 
                                     ground_truth_file: str) -> Tuple[List, List]:
        """
        Extract predictions and ground truth labels from files.
        
        Args:
            model_file: Path to model predictions file
            ground_truth_file: Path to ground truth annotations file
            
        Returns:
            Tuple of (y_true, y_pred) lists
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, y_true: List, y_pred: List) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values
            
        Returns:
            Dictionary of metric names and values
        """
        pass
    
    def evaluate_single_model(self, model_file: str, ground_truth_file: str,
                            model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a single model.
        
        Args:
            model_file: Path to model predictions file
            ground_truth_file: Path to ground truth file
            model_name: Optional model name (defaults to filename)
            
        Returns:
            Dictionary containing evaluation results
        """
        if model_name is None:
            model_name = os.path.splitext(os.path.basename(model_file))[0]
        
        try:
            # Extract predictions and labels
            y_true, y_pred = self.compute_predictions_and_labels(
                model_file, ground_truth_file
            )
            
            # Compute metrics
            metrics = self.compute_metrics(y_true, y_pred)
            
            return {
                'model_name': model_name,
                'metrics': metrics,
                'y_true': y_true,
                'y_pred': y_pred,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'model_name': model_name,
                'metrics': {},
                'y_true': [],
                'y_pred': [],
                'success': False,
                'error': str(e)
            }


class ClassificationEvaluator(BaseEvaluator):
    """Evaluator for classification tasks."""
    
    def __init__(self, data_loader: DataLoader, label_processor: LabelProcessor,
                 visualization_manager: VisualizationManager,
                 exclude_labels: Optional[set] = None):
        """
        Initialize the classification evaluator.
        
        Args:
            data_loader: Instance for loading data files
            label_processor: Instance for processing labels
            visualization_manager: Instance for creating visualizations
            exclude_labels: Set of labels to exclude from evaluation
        """
        super().__init__(data_loader, label_processor, visualization_manager)
        self.exclude_labels = exclude_labels or set()
    
    def compute_predictions_and_labels(self, model_file: str, 
                                     ground_truth_file: str) -> Tuple[List[str], List[str]]:
        """
        Extract classification predictions and labels.
        
        Args:
            model_file: Path to model predictions JSON file
            ground_truth_file: Path to ground truth JSON file
            
        Returns:
            Tuple of (y_true, y_pred) label lists
        """
        model_data = self.data_loader.load_json(model_file)
        gt_data = self.data_loader.load_json(ground_truth_file)
        
        # Extract detections and annotations
        model_detections = model_data
        gt_annotations = gt_data.get("annotations", {})
        
        y_true = []
        y_pred = []
        
        # Process each image
        for img_name, gt_labels in gt_annotations.items():
            # Clean ground truth labels
            gt_labels_clean = [label for label in gt_labels if label]
            
            # Get model predictions for this image
            model_objects = model_detections.get(img_name, [])
            model_labels = self._extract_labels_from_objects(model_objects)
            
            # Align predictions with ground truth
            for idx, gt_label in enumerate(gt_labels_clean):
                if idx < len(model_labels):
                    pred_label = model_labels[idx]
                else:
                    pred_label = "no_detection"
                
                y_true.append(gt_label.strip().lower())
                y_pred.append(pred_label.strip().lower())
        
        return y_true, y_pred
    
    def _extract_labels_from_objects(self, model_objects: List[Dict]) -> List[str]:
        """
        Extract labels from model detection objects.
        
        Args:
            model_objects: List of detection objects
            
        Returns:
            List of extracted labels
        """
        labels = []
        for obj in model_objects:
            # Try different possible label keys
            label = obj.get("color") or obj.get("label") or obj.get("class_name")
            if label and isinstance(label, str) and label.strip():
                labels.append(label.strip().lower())
        return labels
    
    def compute_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of classification metrics
        """
        # Filter out excluded labels
        y_true_filtered, y_pred_filtered = self.label_processor.filter_labels(
            y_true, y_pred, self.exclude_labels
        )
        
        if not y_true_filtered:
            return {'error': 'No valid samples after filtering'}
        
        # Compute metrics
        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_filtered, y_pred_filtered, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'num_samples': len(y_true_filtered)
        }


class RegressionEvaluator(BaseEvaluator):
    """Evaluator for regression tasks."""
    
    def compute_predictions_and_labels(self, model_file: str, 
                                     ground_truth_file: str) -> Tuple[List[float], List[float]]:
        """
        Extract regression predictions and values.
        
        Args:
            model_file: Path to model predictions JSON file
            ground_truth_file: Path to ground truth JSON file
            
        Returns:
            Tuple of (y_true, y_pred) value lists
        """
        model_data = self.data_loader.load_json(model_file)
        gt_data = self.data_loader.load_json(ground_truth_file)
        
        model_detections = model_data.get("detections", {})
        gt_annotations = gt_data.get("annotations", {})
        
        y_true = []
        y_pred = []
        
        # Use only images present in both datasets
        common_images = set(gt_annotations.keys()) & set(model_detections.keys())
        
        for img_name in common_images:
            gt_values = gt_annotations[img_name]
            model_objects = model_detections[img_name]
            
            for idx, gt_value in enumerate(gt_values):
                if idx < len(model_objects):
                    # Extract numeric value from model object
                    pred_value = self._extract_numeric_value(model_objects[idx])
                else:
                    pred_value = 0.0  # Penalty for missing detections
                
                y_true.append(float(gt_value))
                y_pred.append(float(pred_value))
        
        return y_true, y_pred
    
    def _extract_numeric_value(self, model_object: Dict) -> float:
        """
        Extract numeric value from model detection object.
        
        Args:
            model_object: Detection object dictionary
            
        Returns:
            Extracted numeric value
        """
        # Try different possible keys for numeric values
        value = (model_object.get("liquid_level") or 
                model_object.get("value") or 
                model_object.get("score") or 0.0)
        return float(value)
    
    def compute_metrics(self, y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """
        Compute regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of regression metrics
        """
        if not y_true:
            return {'error': 'No valid samples'}
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'num_samples': len(y_true)
        }


class EvaluationFramework:
    """Main framework class that orchestrates the evaluation process."""
    
    def __init__(self, output_directory: str = "evaluation_results"):
        """
        Initialize the evaluation framework.
        
        Args:
            output_directory: Directory to save evaluation results
        """
        self.data_loader = DataLoader()
        self.label_processor = LabelProcessor()
        self.visualization_manager = VisualizationManager(output_directory)
        self.results = []
    
    def evaluate_classification_models(self, model_files: List[str], 
                                     ground_truth_file: str,
                                     label_mapping: Optional[Dict[str, str]] = None,
                                     exclude_labels: Optional[set] = None,
                                     generate_visualizations: bool = True) -> List[Dict]:
        """
        Evaluate multiple classification models.
        
        Args:
            model_files: List of model prediction file paths
            ground_truth_file: Path to ground truth file
            label_mapping: Optional mapping for label transformation
            exclude_labels: Set of labels to exclude from evaluation
            generate_visualizations: Whether to generate confusion matrices
            
        Returns:
            List of evaluation results for each model
        """
        # Update label processor with mapping
        if label_mapping:
            self.label_processor.label_mapping = label_mapping
        
        # Create evaluator
        evaluator = ClassificationEvaluator(
            self.data_loader, self.label_processor, self.visualization_manager,
            exclude_labels=exclude_labels
        )
        
        results = []
        print("=== Classification Model Evaluation ===")
        
        for model_file in model_files:
            print(f"\nEvaluating: {os.path.basename(model_file)}")
            
            result = evaluator.evaluate_single_model(model_file, ground_truth_file)
            
            if result['success']:
                # Print metrics
                metrics = result['metrics']
                print(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
                print(f"F1-Score: {metrics.get('f1_weighted', 0):.3f}")
                print(f"Samples: {metrics.get('num_samples', 0)}")
                
                # Generate visualization if requested
                if generate_visualizations and result['y_true']:
                    filepath = self.visualization_manager.save_confusion_matrix(
                        result['y_true'], result['y_pred'], result['model_name']
                    )
                    print(f"Confusion matrix saved: {filepath}")
            else:
                print(f"Error: {result['error']}")
            
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def evaluate_regression_models(self, model_files: List[str], 
                                 ground_truth_file: str) -> List[Dict]:
        """
        Evaluate multiple regression models.
        
        Args:
            model_files: List of model prediction file paths
            ground_truth_file: Path to ground truth file
            
        Returns:
            List of evaluation results for each model
        """
        evaluator = RegressionEvaluator(
            self.data_loader, self.label_processor, self.visualization_manager
        )
        
        results = []
        print("=== Regression Model Evaluation ===")
        
        for model_file in model_files:
            print(f"\nEvaluating: {os.path.basename(model_file)}")
            
            result = evaluator.evaluate_single_model(model_file, ground_truth_file)
            
            if result['success']:
                metrics = result['metrics']
                print(f"MAE: {metrics.get('mae', 0):.3f}")
                print(f"RMSE: {metrics.get('rmse', 0):.3f}")
                print(f"Samples: {metrics.get('num_samples', 0)}")
            else:
                print(f"Error: {result['error']}")
            
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def generate_summary_report(self, output_file: str = "evaluation_summary.json"):
        """
        Generate a summary report of all evaluations.
        
        Args:
            output_file: Path to save the summary report
        """
        summary = {
            'total_models_evaluated': len(self.results),
            'successful_evaluations': sum(1 for r in self.results if r['success']),
            'failed_evaluations': sum(1 for r in self.results if not r['success']),
            'results': self.results
        }
        
        filepath = os.path.join(self.visualization_manager.output_directory, output_file)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nSummary report saved: {filepath}")
        return summary


# Example usage demonstrating the framework
def example_usage():
    """Example demonstrating how to use the evaluation framework."""
    
    # Initialize the framework
    framework = EvaluationFramework(output_directory="my_evaluation_results")
    
    # Example 1: Color-based anomaly detection
    risk_mapping = {
        "bright green": "high",
        "light green": "high",
        "green": "medium",
        "bright yellow": "medium",
        "black": "medium",
        "crimson": "medium",
        "blue": "medium",
        # ... other mappings
    }
    
    classification_models = [
        "path/to/model1.json",
        "path/to/model2.json",
    ]
    
    classification_results = framework.evaluate_classification_models(
        model_files=classification_models,
        ground_truth_file="path/to/ground_truth.json",
        label_mapping=risk_mapping,
        exclude_labels={"unknown", "no_detection"},
        generate_visualizations=True
    )
    
    # Example 2: Liquid level estimation
    regression_models = [
        "path/to/model1.json",
        "path/to/model2.json",
    ]
    
    regression_results = framework.evaluate_regression_models(
        model_files=regression_models,
        ground_truth_file="path/to/ground_truth.json"
    )
    
    # Generate summary report
    framework.generate_summary_report()


if __name__ == "__main__":
    example_usage()
