# A Real-Time Computer Vision System for Liquid Monitoring in Transparent Containers

This repository contains the official implementation of our technical paper: **"A Real-Time Computer Vision System for Liquid Monitoring in Transparent Containers"**. The system leverages advanced computer vision techniques to detect and monitor liquid levels in transparent containers in real-time.

---

## Repository Structure

- `src/` – Contains the source code for the liquid level monitoring system:
  - `main.py` – Main script for real-time liquid level monitoring.
  - `baselines.py` – Implementation of classical computer vision methods for baseline comparisons.
  - `utils/` – General-purpose utility functions. 
- `weights/` – Pre-trained YOLOv11 model weights:
  - `lmai-11n-seg.pt` – Nano variant (optimized for edge devices).
  - `lmai-11s-seg.pt` – Small variant (balanced for speed and accuracy).
  - `lmai-11m-seg.pt` – Medium variant (higher accuracy model).
- `annotations/` – Annotation files used for evaluation:
  - `liquid_level/` – Annotations indicating liquid levels in containers.
  - `color_annotations/` – Annotations based on color features for liquid detection.
- `README.md` – Project overview and usage instructions.

---

## Dataset

The dataset used for training, validation, and testing is publicly available on Roboflow:

🔗 [LiquiContain Dataset on Roboflow](https://universe.roboflow.com/liquidfy/liquicontain)

This dataset is designed for semantic segmentation tasks and includes manually annotated objects related to liquids and transparent containers. The labeled classes are:

- `glass`
- `wine`
- `bottle`
- `liquid`


---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.10+
- **CUDA-compatible NVIDIA GPU** (strongly recommended for training and inference acceleration)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/IsiPaz/LiquidMonitorAI
   cd liquid-monitoring
   ```

2. Create a Virtual Environment

    #### On Windows
    
      ```bash
      python -m venv venv
      ```
    
    #### On Linux/macOS
    
      ```bash
      python3 -m venv venv
      ```
      

    > Note: On Debian/Ubuntu-based systems, you may need to install the venv module first

3. Activate the Virtual Environment

    #### On Windows
    
      ```bash
      venv\Scripts\activate
      ```
    #### On Linux/macOS
      ```bash
      source venv/bin/activate
      ```
    > Once activated, your terminal prompt will show the virtual environment name, indicating you are working inside it.

4. Install required packages:
   With the virtual environment activated, install the required packages using:
      ```bash
      pip install -r requirements.txt

5. Run Inference

    To perform inference, navigate to the `src/` directory and run `main.py` with the appropriate arguments.
    
      ```bash
      cd src
      python main.py --input <path_to_input> [--model <path_to_model>] [--conf <threshold>] [--output-dir <dir>] [--save]
      ```
    
    Example (with image input and saving results):
    
      ```bash
      python main.py --input ../examples/sample.jpg --model ../weights/lmai-11m-seg.pt --conf 0.5 --output-dir ../output --save
      ```
  
    Example (webcam input):
    
      ```bash 
      python main.py --input 0 --model ../weights/lmai-11m-seg.pt --conf 0.5
      ```
  
    > **Note:** Use a CUDA-enabled GPU for optimal performance. The system automatically runs on cuda if available.

    #### Inference Arguments

    | Argument        | Type    | Description                                                                 |
    |----------------|---------|-----------------------------------------------------------------------------|
    | `--input`       | `str`   | **Required.** Path to an image/video file or use `0` for webcam input.     |
    | `--model`       | `str`   | Path to the YOLOv11 model weights. Default: `../weights/lmai-11m-seg.pt`   |
    | `--conf`        | `float` | Confidence threshold for detections. Default is `0.5`.                     |
    | `--output-dir`  | `str`   | Directory to save output image/video and JSON files. Default: `output/`    |
    | `--save`        | `flag`  | If included, the system will save both visual output and JSON predictions. |
    

