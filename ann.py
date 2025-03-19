import os,time, re, json, pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Constants
focal_length_pixel = 1600
mode = {1: 'Front View', 2: 'Top View', 3: 'Back View', 4: 'Bottom View', 5: 'Left-Hand Side View', 6: 'Right-Hand Side View'}
base = {
    20: 35,
    40: 65,
    50: 80,  # temp
    60: 85,
    75: 100,
    85: 108
}

def CreateNewFile(filename):
    """Create initial JSON file with empty structures"""
    data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    with open(f"{filename}.json", "w") as json_file:
        json.dump(data, json_file, indent=4)
    print("Initial JSON file created successfully!")

def BatchProcessImage(base_dir, filename, object_df):
    """Process all images in batches and write to JSON once at the end"""
    
    # Load the initial file structure
    with open(f"{filename}.json", "r") as json_file:
        data = json.load(json_file)
    
    # Create all categories up front
    category_map = {}  # Maps category name to ID
    for index, row in object_df.iterrows():
        object_name = row['label_name']
        if object_name not in category_map:
            category_id = len(data["categories"])
            category_map[object_name] = category_id
            data["categories"].append({
                "id": category_id,
                "name": object_name,
                "supercategory": "CommonObject"
            })
    
    # Keep track of next IDs
    next_image_id = 1
    next_annotation_id = 1
    
    # Process each object class
    for index, row in object_df.iterrows():
        object_name = row['label_name']
        shape = row['shape']
        
        # Load all images for this class
        try:
            images = os.listdir(f'{base_dir}/{object_name}')
        except FileNotFoundError:
            print(f"Directory not found: {base_dir}/{object_name}")
            continue
        
        # Process based on shape
        for img_name in images:
            image_path = f'{base_dir}/{object_name}/{img_name}'
            
            # Extract distance from filename
            match = re.search(r'_(\d+)_', img_name)
            if not match:
                print(f"Warning: Could not extract distance from filename {img_name}")
                continue
                
            distance = int(match.group(1))
            temp_mode = int(img_name[len(object_name)+4]) if len(img_name) > len(object_name)+4 else 1
            mode_base = base.get(distance, 80)  # Default to 80 if not found
            
            # Calculate bounding box based on shape
            xmin, ymin, box_width, box_height, area = CalculateBBox(
                shape, row, distance, temp_mode, mode_base, focal_length_pixel
            )
            
            if box_width <= 0 or box_height <= 0:
                print(f"Warning: Invalid box dimensions for {img_name}")
                continue
            
            # Add image entry
            new_image = {
                "id": next_image_id,
                "file_name": image_path,
                "width": 1920,  # Assuming standard resolution
                "height": 1080  # Assuming standard resolution
            }
            
            # Add annotation entry
            new_annotation = {
                "id": next_annotation_id,
                "image_id": next_image_id,
                "category_id": category_map[object_name],
                "bbox": [float(xmin), float(ymin), float(box_width), float(box_height)],
                "area": float(area),
                "iscrowd": 0
            }
            
            # Add to data dictionary
            data["images"].append(new_image)
            data["annotations"].append(new_annotation)
            
            # Increment IDs
            next_image_id += 1
            next_annotation_id += 1
    
    # Write all data at once to JSON file
    with open(f"{filename}.json", "w") as json_file:
        json.dump(data, json_file, indent=4)
    
    print(f"Processed {next_image_id-1} images with {next_annotation_id-1} annotations")

def CalculateBBox(shape, row, distance, temp_mode, mode_base, focal_length_pixel):
    """Calculate bounding box parameters based on object shape and properties"""
    # Image dimensions
    img_width = 1920
    img_height = 1080
    xmid = img_width // 2  # 960
    
    # Initialize default values
    xmin, ymin, width, height, area = 0, 0, 0, 0, 0
    
    if shape == 'Cuboid':
        height_obj = row['height']
        breadth = row['breadth']
        width_obj = row['width']
        
        pixel_height = (height_obj * focal_length_pixel) / distance
        pixel_breadth = (breadth * focal_length_pixel) / distance
        pixel_width = (width_obj * focal_length_pixel) / distance
        
        # Apply adjustment factor to prevent negative values
        adjustment_factor = 0.82
        
        if temp_mode in [1, 3]:  # Front View, Back View
            pixel_breadth = pixel_breadth / adjustment_factor
            xmin = xmid - (pixel_breadth / 2)
            ymin = img_height - mode_base - pixel_height
            width = pixel_breadth
            height = pixel_height
            
        elif temp_mode in [2, 4]:  # Top View, Bottom View
            pixel_breadth = pixel_breadth / adjustment_factor
            xmin = xmid - (pixel_breadth / 2)
            ymin = img_height - mode_base - pixel_width
            width = pixel_breadth
            height = pixel_width
            
        elif temp_mode in [5, 6]:  # Left-Hand Side View, Right-Hand Side View
            pixel_width = pixel_width / adjustment_factor
            xmin = xmid - (pixel_width / 2)
            ymin = img_height - mode_base - pixel_height
            width = pixel_width
            height = pixel_height
    
    elif shape == 'Spherical':
        radius = row['radius']
        pixel_radius = (radius * focal_length_pixel) / distance
        diameter = pixel_radius * 2
        pixel_radius = pixel_radius / 0.82
        
        xmin = xmid - pixel_radius
        ymin = img_height - mode_base - diameter
        width = 2 * pixel_radius
        height = diameter
        
    elif shape == 'Cylindrical':
        height_obj = row['height']
        radius = row['radius']
        
        pixel_height = (height_obj * focal_length_pixel) / distance
        pixel_radius = (radius * focal_length_pixel) / distance
        pixel_radius = pixel_radius / 0.825
        diameter = 2 * pixel_radius
        xmin = xmid - pixel_radius
        
        if temp_mode in [1, 3, 5, 6]:  # Front, Back, Side Views
            ymin = img_height - pixel_height - mode_base
            width = diameter
            height = pixel_height
            
        elif temp_mode in [2, 4]:  # Top, Bottom Views
            pixel_height = 2 * pixel_radius * 0.825
            ymin = img_height - pixel_height - mode_base
            width = pixel_radius * 2
            height = pixel_height
    
    # Ensure all coordinates are valid (apply safety margins)
    safety_margin = 1.0  # Add a small safety margin to avoid borderline cases
    
    # Constrain to image boundaries with safety margin
    xmin = max(safety_margin, min(img_width - width - safety_margin, xmin))
    ymin = max(safety_margin, min(img_height - height - safety_margin, ymin))
    
    # Ensure minimum dimensions
    width = max(1.0, width)
    height = max(1.0, height)
    
    # Recalculate area
    area = width * height
    
    return xmin, ymin, width, height, area

def main():
    base_dir = 'LabelImages'
    filename = 'ImageAnnotation'
    
    
    # Load metadata only once
    object_df = pd.read_csv("object_metadata.csv")
    
    # Create empty JSON file
    CreateNewFile(filename)
    
    # Process all images in one go
    BatchProcessImage(base_dir, filename, object_df)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Time Elapsed = " ,end - start)