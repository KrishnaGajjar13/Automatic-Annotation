import os,cv2,json, pandas as pd
import numpy as np
from PIL import Image as img

object_df = pd.read_csv("object_metadata.csv")
focal_length_pixel = 1600
mode = {1 : 'Front View',2 : 'Top View', 3 : 'Back View', 4 : 'Bottom View', 5 : 'Left-Hand Side View', 6 : 'Right-Hand Side View'}
base = {
    20 : 35,
    40 : 60,
    50 : 70,
    60 : 85,
    75 : 100,
    85 : 100
}

"""annotation{
"id": int, "image_id": int, "category_id": int, "segmentation": RLE or [polygon], "area": float, "bbox": [x,y,width,height], "iscrowd": 0 or 1,
}"""

def create_new_file(filename):
    with open(f"{filename}.json", "w") as json_file:
        data = {
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 0,  # Initially, you can set this to 0 or a placeholder
                    "name": "Undefined",  # Placeholder name
                    "supercategory": "CommonObject"  # Placeholder supercategory
                }
            ]
        }
        json.dump(data, json_file, indent=4)
    print("Initial JSON file created successfully!")

# Function to add new annotation
def add_image_and_annotation(mainfile, file_name, width, height, bbox, area, category_name):
    # Loading existing data from JSON file
    with open(f"{mainfile}.json", "r") as json_file:
        data = json.load(json_file)

    # Ensuring category exists if not then creating it
    category_id = None
    for category in data["categories"]:
        if category["name"] == category_name:
            category_id = category["id"]
            break
    if category_id is None:
        # If the category doesn't exist, create a new category
        category_id = len(data["categories"]) + 1
        new_category = {
            "id": category_id,
            "name": category_name,
            "supercategory": "CommonObject"
        }
        data["categories"].append(new_category)
        print(f"New category '{category_name}' added successfully!")

    # Get the next image id by adding 1 to the last image id
    new_image_id = len(data["images"]) + 1

    # Create the new image dictionary
    new_image = {
        "id": new_image_id,
        "file_name": file_name,
        "width": width,
        "height": height
    }

    # Get the next annotation id by adding 1 to the last annotation id
    new_annotation_id = len(data["annotations"]) + 1

    # Create the new annotation dictionary
    new_annotation = {
        "id": new_annotation_id,
        "image_id": new_image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": area,
    }

    # Add the new image and annotation to the lists
    data["images"].append(new_image)
    data["annotations"].append(new_annotation)

    # Save the updated JSON back to the file
    with open(f"{mainfile}.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"New image '{file_name}' with annotation added successfully!")


def add_new_class(name, mainfile):
    with open(f"{mainfile}.json", "r") as json_file:
        data = json.load(json_file)

    # Get the next category id by adding 1 to the last category id
    new_category_id = len(data["categories"])

    # Create the new category dictionary
    new_category = {
        "id": new_category_id,
        "name": name,
    }

    # Add the new category to the list
    data["categories"].append(new_category)

    # Save the updated JSON back to the file
    with open(f"{mainfile}.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"New category '{name}' added successfully!")

def load_images(base_dir,class_name):
    images = os.listdir(f'{base_dir}/{class_name}')
    return images    


def label_images(base_dir,filename):
    for index, row in object_df.iterrows():
        object_name = row['label_name']
        add_new_class(object_name,filename)
        shape = row['shape']
        images = load_images(base_dir,object_name)
        xmid = 960
        if shape == 'Cuboid':
            height = row['height']
            breadth = row['breadth']
            width = row['width']
            for img in images:
                distance = int(img[len(object_name)+1]+img[len(object_name)+2])
                temp_mode = int(img[len(object_name)+4])
                pixel_height = int((height*focal_length_pixel)/distance)
                pixel_breadth = int((breadth*focal_length_pixel)/distance)
                pixel_width = int((width*focal_length_pixel)/distance)
                mode_base = int(base.get(distance)) # type: ignore
                match temp_mode:
                    case 1 | 3:
                        pixel_breadth = int(pixel_breadth / 0.825)
                        xmin = xmid - (pixel_breadth / 2)
                        ymin = 1080 - pixel_height - mode_base
                        area = pixel_height * pixel_breadth
                        add_image_and_annotation(filename, f'{base_dir}/{object_name}/{img}', pixel_breadth, pixel_height, [xmin,ymin,pixel_breadth,pixel_height],area,object_name)
                    
                    case 2 | 4:
                        pixel_breadth = int(pixel_breadth / 0.825)
                        xmin = xmid - (pixel_breadth / 2) 
                        ymin = 1080 - pixel_width - mode_base
                        area = pixel_breadth * pixel_width
                        add_image_and_annotation(filename, f'{base_dir}/{object_name}/{img}', pixel_breadth, pixel_width, [xmin,ymin,pixel_breadth,pixel_width],area,object_name)

                    case 5 | 6:
                        pixel_width = int(pixel_width / 0.825)
                        xmin = xmid - (pixel_width / 2) 
                        ymin = 1080 - pixel_height - mode_base
                        area = pixel_height * pixel_width
                        add_image_and_annotation(filename, f'{base_dir}/{object_name}/{img}', pixel_width, pixel_height, [xmin,ymin,pixel_width,pixel_height],area,object_name)
                                        
        elif shape == 'Spherical':
            radius = row['radius']
            for img in images:
                distance = int(img[len(object_name)+1]+img[len(object_name)+2])
                temp_mode = int(img[len(object_name)+4])
                pixel_radius = int((radius*focal_length_pixel)/distance)
                diameter = 2 * pixel_radius
                pixel_radius = int(pixel_radius / 0.825)
                mode_base = int(base.get(distance))  # type: ignore
                ymin = 1080 - diameter - mode_base 
                xmin = xmid - pixel_radius
                area = diameter * diameter
                add_image_and_annotation(filename, f'{base_dir}/{object_name}/{img}', diameter, diameter, [xmin,ymin,2*pixel_radius,diameter],area,object_name)


        elif shape == 'Cylindrical':
            height = row['height']
            radius = row['radius']  
            for img in images:
                distance = int(img[len(object_name)+1]+img[len(object_name)+2])
                temp_mode = int(img[len(object_name)+4])
                pixel_height = int((height*focal_length_pixel)/distance)
                pixel_radius = int((radius*focal_length_pixel)/distance)
                diameter = 2 * pixel_radius
                pixel_radius = int(pixel_radius / 0.825)
                mode_base = int(base.get(distance))  # type: ignore
                ymin = 1080 - pixel_height - mode_base 
                match temp_mode:
                    case 1 | 3 | 5 | 6:
                        xmin = xmid - pixel_radius
                        area = pixel_height * diameter
                        add_image_and_annotation(filename, f'{base_dir}/{object_name}/{img}', diameter, pixel_height, [xmin,ymin,diameter,pixel_height],area,object_name)

                    case 2 | 4:
                        xmin = xmid - pixel_radius
                        area = diameter * diameter
                        add_image_and_annotation(filename, f'{base_dir}/{object_name}/{img}', diameter, diameter, [xmin,ymin,diameter,diameter],area,object_name)

        else:
            print("Shape not given or Invalid Shape!")

base_dir = 'label_images'
filename = 'image_annotation'
create_new_file(filename)
label_images(base_dir=base_dir,filename=filename)