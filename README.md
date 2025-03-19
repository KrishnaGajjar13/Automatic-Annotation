# Automatic-Annotation

## Overview
Automatic-Annotation is a Python-based tool for capturing multi-view images of objects and generating dataset annotations in the **COCO format**. This project automates the annotation process, making it easier to collect and prepare datasets for machine learning and computer vision tasks.

## Features
- Captures **6 views** of an object: **Front, Back, Left, Right, Top, Bottom**
- Automatically determines **camera distances** based on object size (up to **35 cm**)
- Stores object details in a **CSV file**
- Generates COCO format annotations in **JSON**
- Augments images with **blurs, flips, and transformations**
- Organized storage:
  - **label_videos/** → Captured videos
  - **LabelImages/** → Original images
  - **AugmentedImages/** → Augmented images

## Installation
```
# Clone the repository
git clone https://github.com/KrishnaGajjar13/Automatic-Annotation.git
cd Automatic-Annotation

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Step 1: Capture Object Images & Metadata
Run the **mode.py** script to capture images:
```
python mode.py
```
#### What Happens?
- The script asks for:
  - **Item Name**
  - **Shape** (Cuboid, Cylindrical, Sphere)
  - **Dimension Size**
- Opens the **camera** (you may need to modify the port index)
- Captures **6 views** at **two distances**
- Saves the details to a **CSV file**

### Step 2: Generate Annotations
Run the **ann.py** script to create annotations in **COCO format**:
```
python ann.py
```
#### Output:
- A **JSON annotation file** is generated for your dataset.

### Step 3: Data Augmentation (Optional)
Run the **aug.py** script to expand your dataset:
```
python aug.py
```
#### Transformation Options:
- **Blur** (various types)
- **Vertical & Horizontal Flips**
- **Recommended Upscale Ratio**: `1:1` (default is `2:1`, adjust as needed)

## Notes
- **Captured images** are stored in `LabelImages/`
- **Videos** are saved in `label_videos/`
- **Augmented images** are stored in `AugmentedImages/`
- Ensure your **camera is properly configured** and modify the port index if needed.

## License
This project is licensed under the **MIT License**.

## Contributions
Contributions are welcome! Feel free to submit **issues** or **pull requests**.

---

🚀 Happy Annotating!

