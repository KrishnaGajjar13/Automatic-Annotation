import torch
import albumentations as A
import json, os, cv2, numpy as np
import time
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from albumentations.pytorch import ToTensorV2
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("augmentation.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_transforms(input_size=300, min_area=0.0, min_visibility=0.0):
    return A.Compose([
        A.Resize(height=input_size, width=input_size, p=1.0),
        A.HorizontalFlip(p=0.7),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.0,
            p=0.7
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(5, 12), p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0)
        ], p=0.6),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.15, 0.15),
            rotate=(-20, 20),
            shear=(-10, 10),
            p=0.5
        ),
        A.CoarseDropout(
            num_holes_range=(3, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0,
            p=0.5
        ),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['class_labels'],
        min_area=min_area,
        min_visibility=min_visibility
    ))

def convert_to_serializable(obj):
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

def process_image(
    img_info: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    transform_func: Any,
    new_image_id: int,
    augmentation_number: int,
    save_dir: Optional[str] = None,
    input_size: int = 300
) -> Optional[Dict[str, Any]]:
    """
    Process a single image and its annotations with augmentation.
    For each augmented image, a unique id is generated that is used in the file name,
    the image info, and all corresponding annotations.
    
    If there is only one bounding box, its annotation 'id' is set equal to the image id.
    
    In case no bounding boxes remain after the transformation (due to filtering by min_visibility/min_area),
    this function falls back to using the original bounding boxes so that every augmented image has at least one annotation.
    """
    original_file_name = img_info['file_name']
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Read and convert the original image
        image = cv2.imread(original_file_name)
        if image is None:
            logger.warning(f"Could not read image {original_file_name}")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        bboxes = []
        labels = []
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            bboxes.append(bbox)
            labels.append(ann['category_id'])
        
        # If no bboxes in original image, log and return None (or decide on a fallback)
        if len(bboxes) == 0:
            logger.info(f"Skipping augmentation for {original_file_name} as it has no bounding boxes.")
            return None
        
        # Apply augmentation transformation
        transformed = transform_func(image=image, bboxes=bboxes, class_labels=labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed.get('bboxes', [])
        transformed_labels = transformed.get('class_labels', [])
        
        # If no transformed bounding boxes remain, fall back to the original boxes
        if not transformed_bboxes:
            logger.warning(f"No bounding boxes remain after transformation for {original_file_name}. "
                           f"Falling back to original bounding boxes.")
            transformed_bboxes = bboxes
            transformed_labels = labels
        
        # Create new file name embedding the augmentation number
        file_name_base, file_ext = os.path.splitext(os.path.basename(original_file_name))
        new_file_name = f"{file_name_base}_aug{augmentation_number}{file_ext}"
        new_file_path = os.path.join(save_dir, new_file_name) if save_dir else os.path.join(os.path.dirname(original_file_name), new_file_name)
        
        # Convert tensor to numpy if necessary and adjust type
        if isinstance(transformed_image, torch.Tensor):
            transformed_image_np = transformed_image.permute(1, 2, 0).numpy()
        else:
            transformed_image_np = transformed_image
        if transformed_image_np.dtype != np.uint8:
            transformed_image_np = (transformed_image_np * 255).astype(np.uint8)
        if transformed_image_np.shape[2] == 3:
            transformed_image_bgr = cv2.cvtColor(transformed_image_np, cv2.COLOR_RGB2BGR)
        else:
            transformed_image_bgr = transformed_image_np
        
        # Save the augmented image
        cv2.imwrite(new_file_path, transformed_image_bgr)
        
        # Update image info based on the actual dimensions of the transformed image
        new_img_info = {
            'id': new_image_id,
            'file_name': new_file_path,
            'width': int(transformed_image_np.shape[1]),
            'height': int(transformed_image_np.shape[0])
        }
        
        new_annotations = []
        for i, (bbox, label) in enumerate(zip(transformed_bboxes, transformed_labels)):
            # For a single bounding box, set the annotation id equal to the image id
            if len(transformed_bboxes) == 1:
                new_annotation_id = new_image_id
            else:
                new_annotation_id = new_image_id * 1000 + i + 1
            
            x, y, width, height = bbox
            # Clamp coordinates to ensure they are within the image boundaries
            x = max(0, min(x, new_img_info['width'] - 1))
            y = max(0, min(y, new_img_info['height'] - 1))
            width = max(1, min(width, new_img_info['width'] - x))
            height = max(1, min(height, new_img_info['height'] - y))
            
            new_annotation = {
                'id': new_annotation_id,
                'image_id': new_image_id,
                'category_id': int(label),
                'bbox': [float(x), float(y), float(width), float(height)],
                'area': float(width * height)
                # You can add "iscrowd": 0 or "segmentation": [] here if needed.
            }
            new_annotations.append(new_annotation)
            logger.debug(f"Added annotation for image {new_image_id}: {new_annotation}")
        
        return {
            'image_info': new_img_info,
            'annotations': new_annotations
        }
        
    except Exception as e:
        logger.error(f"Error processing {original_file_name}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def augment_dataset(
    json_path, 
    transform_func=None,
    num_augmentations_per_image=1,
    verbose=True,
    save_dir=None,
    max_workers=4,
    save_frequency=100,
    checkpoint_path=None,
    input_size=300
):
    start_time = time.time()
    logger.info(f"Starting dataset augmentation: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if transform_func is None:
        transform_func = get_transforms(input_size=input_size)
        logger.info(f"Using default transform with input_size={input_size}")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Saving augmented images to: {save_dir}")
    
    max_image_id = max(img['id'] for img in data['images']) if data['images'] else 0
    logger.info(f"Original dataset contains {len(data['images'])} images and {len(data['annotations'])} annotations")
    logger.info(f"Creating {num_augmentations_per_image} augmented versions per image")
    
    # Group annotations by image id
    annotations_by_image = {}
    for ann in data['annotations']:
        annotations_by_image.setdefault(ann['image_id'], []).append(ann)
    
    if checkpoint_path is None:
        checkpoint_dir = os.path.dirname(json_path) or '.'
        json_name = os.path.basename(json_path)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{json_name}")
    
    new_images = []
    new_annotations = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
            completed_image_ids = set(checkpoint.get('completed_image_ids', []))
            new_images = checkpoint.get('new_images', [])
            new_annotations = checkpoint.get('new_annotations', [])
            logger.info(f"Resuming from checkpoint with {len(completed_image_ids)} completed images")
    else:
        completed_image_ids = set()
    
    original_images = data['images']
    total_to_process = len(original_images) * num_augmentations_per_image
    current_new_image_id = max_image_id + 1 + len(new_images)
    
    with tqdm(total=total_to_process, desc="Augmenting images") as pbar:
        images_to_process = [img for img in original_images if img['id'] not in completed_image_ids]
        batch_size = min(100, len(images_to_process))
        for batch_start in range(0, len(images_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(images_to_process))
            batch = images_to_process[batch_start:batch_end]
            batch_results = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for img_info in batch:
                    img_id = img_info['id']
                    img_annotations = annotations_by_image.get(img_id, [])
                    for aug_idx in range(num_augmentations_per_image):
                        new_image_id = current_new_image_id
                        current_new_image_id += 1
                        futures.append(
                            executor.submit(
                                process_image,
                                img_info,
                                img_annotations,
                                transform_func,
                                new_image_id,
                                aug_idx + 1,
                                save_dir,
                                input_size
                            )
                        )
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        batch_results.append(result)
                        pbar.update(1)
            
            for result in batch_results:
                new_images.append(result['image_info'])
                new_annotations.extend(result['annotations'])
            
            completed_image_ids.update(img['id'] for img in batch)
            if len(new_images) % save_frequency == 0 or batch_end == len(images_to_process):
                with open(checkpoint_path, 'w') as f:
                    checkpoint = {
                        'completed_image_ids': list(completed_image_ids),
                        'new_images': new_images,
                        'new_annotations': new_annotations
                    }
                    json.dump(convert_to_serializable(checkpoint), f)
                logger.info(f"Progress: {len(completed_image_ids)}/{len(original_images)} images processed; {len(new_images)} augmentations added.")
    
    # Append augmented images and annotations to the original dataset
    data['images'].extend(new_images)
    data['annotations'].extend(new_annotations)
    
    output_path = json_path.replace('.json', '_augmented.json')
    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(data), f, indent=2)
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    elapsed_time = time.time() - start_time
    if verbose:
        logger.info(f"Augmentation Complete in {elapsed_time:.2f} seconds:")
        logger.info(f"Original Images: {len(original_images)}")
        logger.info(f"Total Images After Augmentation: {len(data['images'])}")
        logger.info(f"Augmented Images Added: {len(new_images)}")
        logger.info(f"Original Annotations: {len(data['annotations']) - len(new_annotations)}")
        logger.info(f"Total Annotations After Augmentation: {len(data['annotations'])}")
        logger.info(f"Augmented Annotations Added: {len(new_annotations)}")
        logger.info(f"Output saved to: {output_path}")

if __name__ == "__main__":
    # Adjust min_area and min_visibility as needed; here we use low thresholds to avoid filtering out bboxes.
    augment_dataset(
        json_path='ImageAnnotation_augmented_cleaned.json',
        transform_func=get_transforms(input_size=300, min_area=0.0, min_visibility=0.0),
        num_augmentations_per_image=2,
        verbose=True,
        save_dir='AugmentedImages',
        max_workers=4,
        save_frequency=500,
        input_size=300
    )
