import torch
import albumentations as A
import json, os, cv2, numpy as np
from torch.utils.data import Dataset, DataLoader,random_split
from albumentations.pytorch import ToTensorV2
from typing import List
from albumentations.core.transforms_interface import BasicTransform
  
"""def get_transforms():
    return A.Compose([
        A.Resize(300, 300),  # Resize all images to 300x300
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomRotate90(p=0.3),
        A.GaussianBlur(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize for SSD
        ToTensorV2()  # Convert to tensor
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))  # SSD uses Coco format"""

def get_transforms(
    input_size=300,
    min_area=0.0,
    min_visibility=0.0,
    prob_geometric=0.5,
    prob_pixel=0.5,
    use_heavy_augmentations=False
):
    """
    Create a comprehensive set of image transforms for object detection.
    
    Args:
        input_size (int): Target size for resizing images. Defaults to 300.
        min_area (float): Minimum area of bounding box after augmentation (relative to original image).
        min_visibility (float): Minimum visibility of bounding box after augmentation.
        prob_geometric (float): Probability of applying geometric augmentations.
        prob_pixel (float): Probability of applying pixel-level augmentations.
        use_heavy_augmentations (bool): Enable more aggressive augmentations.
    
    Returns:
        A.Compose: Albumentations composition of transforms
    """
    # Base transforms that are always applied
    base_transforms: List[BasicTransform] = [
        A.Resize(width=input_size, height=input_size, interpolation=1),
    ]
    
    # Geometric augmentations
    geometric_transforms: List[BasicTransform] = [
        A.HorizontalFlip(p=prob_geometric),
        A.VerticalFlip(p=prob_geometric * 0.3),  # Less likely than horizontal flip
        A.RandomRotate90(p=prob_geometric * 0.4),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            rotate=(-15, 15),
            shear=(-10, 10),
            p=prob_geometric
        ),
    ]
    
    # Pixel-level augmentations
    pixel_transforms: List[BasicTransform] = [
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=prob_pixel
        ),
        A.GaussNoise(p=prob_pixel * 0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=prob_pixel * 0.3),
        A.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.2, 
            p=prob_pixel
        ),
    ]
    
    # Heavy augmentations (optional)
    if use_heavy_augmentations:
        geometric_transforms.extend([
            # Replace ShiftScaleRotate with Affine (which was already included)
            A.ElasticTransform(
                alpha=1, 
                sigma=50,  
                p=prob_geometric * 0.3
            ),
        ])
        
        pixel_transforms.extend([
            A.RandomFog(p=prob_pixel * 0.2),
            A.RandomShadow(p=prob_pixel * 0.2),
            A.ChannelShuffle(p=prob_pixel * 0.1),
        ])
    
    # Normalization and conversion
    final_transforms: List[BasicTransform] = [
        A.Normalize(
            mean=(0.485, 0.456, 0.406),  # ImageNet mean
            std=(0.229, 0.224, 0.225),   # ImageNet std
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ]
    
    # Combine all transforms
    all_transforms: List[BasicTransform] = base_transforms + geometric_transforms + pixel_transforms + final_transforms
    
    return A.Compose(
        all_transforms,  # type: ignore
        bbox_params=A.BboxParams(
            format='coco', 
            label_fields=['class_labels'],
            min_area=min_area,
            min_visibility=min_visibility
        )
    )


def normalize_bboxes(bboxes: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Normalize bounding boxes from pixel coordinates to relative coordinates.
    
    Args:
        bboxes (np.ndarray): Bounding boxes in [x_min, y_min, x_max, y_max] format
        height (int): Image height
        width (int): Image width
    
    Returns:
        np.ndarray: Normalized bounding boxes in [x_min, y_min, x_max, y_max] format
    """
    normalized_bboxes = bboxes.copy().astype(np.float32)
    
    # Divide x coordinates by width
    normalized_bboxes[:, [0, 2]] /= width
    
    # Divide y coordinates by height 
    normalized_bboxes[:, [1, 3]] /= height
    
    return normalized_bboxes

def denormalize_bboxes(bboxes: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Denormalize bounding boxes from relative coordinates to pixel coordinates.
    
    Args:
        bboxes (np.ndarray): Normalized bounding boxes in [x_min, y_min, x_max, y_max] format
        height (int): Image height
        width (int): Image width
    
    Returns:
        np.ndarray: Denormalized bounding boxes in [x_min, y_min, x_max, y_max] format
    """
    denormalized_bboxes = bboxes.copy().astype(np.float32)
    
    # Multiply x coordinates by width
    denormalized_bboxes[:, [0, 2]] *= width
    
    # Multiply y coordinates by height
    denormalized_bboxes[:, [1, 3]] *= height
    
    return denormalized_bboxes

def convert_to_serializable(obj):
    """
    Convert numpy types to standard Python types for JSON serialization.
    
    Args:
        obj: Input object potentially containing numpy types
    
    Returns:
        Serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj

def augment_dataset(
    json_path, 
    transform_func=None,
    num_augmentations_per_image=1,
    verbose=True
):
    """
    Augment dataset by creating transformed copies of images.
    
    Args:
        json_path (str): Path to annotation JSON
        transform_func (callable, optional): Transform function to apply. 
        num_augmentations_per_image (int): Number of augmented versions to create per image
        verbose (bool): Print detailed augmentation information
    """
    # Load annotations
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Track the last used image and annotation IDs
    last_image_id = max(img['id'] for img in data['images'])
    last_annotation_id = max(ann['id'] for ann in data['annotations'])
    
    # Process each original image
    original_images = data['images'].copy()
    original_annotations = data['annotations'].copy()
    
    # Use default transforms if no transform function provided
    if transform_func is None:
        transform_func = get_transforms()
    
    # Counters for tracking
    augmented_image_count = 0
    augmented_annotation_count = 0
    
    for original_img_info in original_images:
        original_file_name = original_img_info['file_name']
        original_image_id = original_img_info['id']
        
        # Read the original image
        try:
            image = cv2.imread(original_file_name)
            if image is None:
                print(f"Warning: Could not read image {original_file_name}")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
        except Exception as e:
            print(f"Error reading {original_file_name}: {e}")
            continue
        
        # Find all annotations for this image
        image_annotations = [
            ann for ann in original_annotations 
            if ann['image_id'] == original_image_id
        ]
        
        # Create multiple augmented versions
        for aug_idx in range(num_augmentations_per_image):
            # Prepare bounding boxes and labels for transformation
            bboxes = []
            labels = []
            for ann in image_annotations:
                bbox = ann['bbox']
                # Convert COCO format (x, y, width, height) to [x_min, y_min, x_max, y_max]
                x_min, y_min, bbox_w, bbox_h = bbox
                x_max, y_max = x_min + bbox_w, y_min + bbox_h
                bboxes.append([x_min, y_min, x_max, y_max])
                labels.append(ann['category_id'])
            
            # Convert to numpy arrays and normalize
            bboxes = np.array(bboxes, dtype=np.float32)
            normalized_bboxes = normalize_bboxes(bboxes, height, width)
            labels = np.array(labels, dtype=np.int64)
            
            try:
                # Apply transformation
                transformed = transform_func(
                    image=image, 
                    bboxes=normalized_bboxes, 
                    class_labels=labels
                )
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_labels = transformed['class_labels']
            except Exception as e:
                print(f"Transformation error for {original_file_name}: {e}")
                continue
            
            # Denormalize bounding boxes back to original image dimensions
            denormalized_bboxes = denormalize_bboxes(transformed_bboxes, height, width)
            
            # Generate new file name
            file_name_base, file_ext = os.path.splitext(original_file_name)
            new_file_name = f"{file_name_base}_aug{aug_idx+1}{file_ext}"
            
            # Save transformed image
            try:
                transformed_image_bgr = cv2.cvtColor(
                    transformed_image.numpy().transpose(1, 2, 0) * 255, 
                    cv2.COLOR_RGB2BGR
                ).astype(np.uint8)
                cv2.imwrite(new_file_name, transformed_image_bgr)
            except Exception as e:
                print(f"Error saving augmented image {new_file_name}: {e}")
                continue
            
            # Increment IDs
            last_image_id += 1
            
            # Add new image info
            new_img_info = original_img_info.copy()
            new_img_info['id'] = last_image_id
            new_img_info['file_name'] = new_file_name
            data['images'].append(new_img_info)
            
            # Convert transformed bboxes back to COCO format and add annotations
            for bbox, label in zip(denormalized_bboxes, transformed_labels):
                last_annotation_id += 1
                x_min, y_min, x_max, y_max = bbox
                width = x_max - x_min
                height = y_max - y_min
                
                new_annotation = {
                    'id': last_annotation_id,
                    'image_id': last_image_id,
                    'category_id': int(label),
                    'bbox': [float(x_min), float(y_min), float(width), float(height)],
                    'area': float(width * height)
                }
                data['annotations'].append(new_annotation)
            
            # Increment counters
            augmented_image_count += 1
            augmented_annotation_count += len(transformed_labels)
    
    # Save the updated annotation file with serializable data
    with open(json_path, 'w') as f:
        json.dump(convert_to_serializable(data), f, indent=2)
    
    if verbose:
        print(f"Augmentation Complete:")
        print(f"Original Images: {len(original_images)}")
        print(f"Total Images After Augmentation: {len(data['images'])}")
        print(f"Augmented Images: {augmented_image_count}")
        print(f"Original Annotations: {len(original_annotations)}")
        print(f"Total Annotations After Augmentation: {len(data['annotations'])}")
        print(f"Augmented Annotations: {augmented_annotation_count}")

augment_dataset(
    json_path='image_annotation.json',
    transform_func=get_transforms(
        input_size=300,
        use_heavy_augmentations=False
    ),
    num_augmentations_per_image=1,
    verbose=True
)