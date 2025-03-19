import torch
import torchvision
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torch.utils.data import Dataset, DataLoader, random_split
import json, os, cv2, numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
from torch.nn.parallel import DataParallel
from functools import partial
import gc
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler


# Configure PyTorch for maximum performance
torch.backends.cudnn.benchmark = True  # Use cuDNN auto-tuner
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False, path='checkpoint.pt'):
        """
        Args:
            patience (int): How many epochs to wait after the last time validation loss improved.
            min_delta (float): Minimum change in the monitored loss to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
            path (str): Path to save the model when validation loss improves.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        # Check if there is improvement beyond the minimum delta threshold
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if improvement happens
            if isinstance(model, DataParallel):
                torch.save(model.module.state_dict(), self.path)
            else:
                torch.save(model.state_dict(), self.path)
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.4f}. Saving model to {self.path}.")
            return True
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation loss. Counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def calculate_iou(box1, box2):
    """
    Calculate intersection over union (IoU) between two bounding boxes.
    
    Args:
        box1 (tensor): First bounding box [x1, y1, x2, y2]
        box2 (tensor): Second bounding box [x1, y1, x2, y2]
        
    Returns:
        float: IoU value
    """
    # Get the coordinates of the intersection rectangle
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])
    
    # Calculate area of intersection rectangle
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
    
    return iou

class PrefetchLoader:
    """
    Prefetch data loader to GPU to minimize data transfer latency.
    """
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
    
    @property
    def dataset(self):
        # Expose the dataset from the underlying loader
        return self.loader.dataset

    @property
    def batch_size(self):
        # Expose batch_size from the underlying loader (if available)
        return self.loader.batch_size

    def __iter__(self):
        loader_iter = iter(self.loader)
        self.preload(loader_iter)
        while self.batch is not None:
            yield self.batch
            self.preload(loader_iter)
    
    def __len__(self):
        return len(self.loader)
    
    def preload(self, loader_iter):
        try:
            self.batch = next(loader_iter)
        except StopIteration:
            self.batch = None
            return
        
        with torch.cuda.stream(self.stream):
            # If the batch is a tuple of (images, targets)
            if isinstance(self.batch[0], list):
                # For object detection: list of images and corresponding targets
                self.batch = (
                    [img.to(self.device, non_blocking=True) for img in self.batch[0]],
                    [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in self.batch[1]]
                )
            else:
                # Generic case: if batch[0] is a Tensor
                if isinstance(self.batch[0], torch.Tensor):
                    self.batch = (self.batch[0].to(self.device, non_blocking=True), self.batch[1])

def create_prefetch_loader(loader, device):
    """Helper function to wrap a DataLoader in PrefetchLoader."""
    return PrefetchLoader(loader, device)

# -------------------------
# mAP Calculation Functions
# -------------------------
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two boxes.
    Boxes are in [xmin, ymin, xmax, ymax] format.
    """
    # Intersection
    inter_xmin = max(box1[0], box2[0])
    inter_ymin = max(box1[1], box2[1])
    inter_xmax = min(box1[2], box2[2])
    inter_ymax = min(box1[3], box2[3])
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    
    # Areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area + 1e-6
    
    return inter_area / union_area

def calculate_ap_single_class(class_preds, class_targets, iou_threshold=0.5):
    """
    Calculate Average Precision for a single class.
    """
    n_gt = sum(target['boxes'].size(0) for target in class_targets)
    if n_gt == 0:
        return 0.0
    
    all_boxes = []
    all_scores = []
    all_img_indices = []
    for img_idx, pred in enumerate(class_preds):
        if pred['boxes'].size(0) > 0:
            all_boxes.append(pred['boxes'])
            all_scores.append(pred['scores'])
            all_img_indices.extend([img_idx] * pred['boxes'].size(0))
    
    if not all_boxes:
        return 0.0
    
    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_img_indices = torch.tensor(all_img_indices)
    
    _, sort_indices = torch.sort(all_scores, descending=True)
    all_boxes = all_boxes[sort_indices]
    all_scores = all_scores[sort_indices]
    all_img_indices = all_img_indices[sort_indices]
    
    tp = torch.zeros_like(all_scores)
    fp = torch.zeros_like(all_scores)
    detected = [torch.zeros(target['boxes'].size(0), dtype=torch.bool) for target in class_targets]
    
    for i in range(all_boxes.size(0)):
        img_idx = all_img_indices[i].item()
        det_box = all_boxes[i]
        if class_targets[img_idx]['boxes'].size(0) == 0:
            fp[i] = 1
            continue
        gt_boxes = class_targets[img_idx]['boxes']
        max_iou = 0.0
        max_gt_idx = -1
        for gt_idx in range(gt_boxes.size(0)):
            if detected[img_idx][gt_idx]:
                continue
            iou = calculate_iou(det_box, gt_boxes[gt_idx])
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx
        if max_iou >= iou_threshold:
            tp[i] = 1
            detected[img_idx][max_gt_idx] = True
        else:
            fp[i] = 1
    
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    recalls = tp_cumsum / n_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    recalls = torch.cat((torch.tensor([0.0]), recalls, torch.tensor([1.0])))
    precisions = torch.cat((torch.tensor([1.0]), precisions, torch.tensor([0.0])))
    
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = torch.max(precisions[i], precisions[i + 1])
    
    indices = torch.where(recalls[1:] != recalls[:-1])[0]
    ap = torch.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap.item()

def calculate_map(predictions, targets, iou_threshold=0.5):
    """
    Calculate mean Average Precision
    
    Args:
        predictions: List of prediction dicts with 'boxes', 'labels', 'scores'
        targets: List of target dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for considering a detection correct
        
    Returns:
        mAP: Mean Average Precision value
    """
    # Get all unique class labels
    all_labels = set()
    for target in targets:
        # Move target labels to CPU before converting to numpy
        all_labels.update(target['labels'].cpu().numpy())
    
    aps = []  # To store Average Precision for each class
    
    # Calculate AP for each class
    for cls in all_labels:
        # Get all detections for this class
        class_preds = []
        class_targets = []
        
        for pred, target in zip(predictions, targets):
            # Predictions for this class
            cls_indices = (pred['labels'] == cls).nonzero(as_tuple=True)[0]
            cls_boxes = pred['boxes'][cls_indices]
            cls_scores = pred['scores'][cls_indices]
            
            # Sort by confidence
            sorted_indices = torch.argsort(cls_scores, descending=True)
            cls_boxes = cls_boxes[sorted_indices]
            cls_scores = cls_scores[sorted_indices]
            
            class_preds.append({
                'boxes': cls_boxes,
                'scores': cls_scores
            })
            
            # Ground truth for this class
            target_indices = (target['labels'] == cls).nonzero(as_tuple=True)[0]
            target_boxes = target['boxes'][target_indices]
            
            class_targets.append({
                'boxes': target_boxes
            })
        
        # Calculate AP for this class
        ap = calculate_ap_single_class(class_preds, class_targets, iou_threshold)
        aps.append(ap)
    
    # Calculate mAP
    mAP = sum(aps) / len(aps) if aps else 0.0
    return mAP


# -------------------------
# Evaluation Function
# -------------------------
def evaluate_model(model, val_loader, device, max_eval_samples=None):
    """
    Evaluate model on validation set with optimized GPU performance.
    
    Args:
        model: The model to evaluate.
        val_loader: Validation DataLoader.
        device: Device for evaluation.
        max_eval_samples: Maximum number of samples to evaluate.
    
    Returns:
        mAP: Mean Average Precision.
    """
    model.eval()
    torch.cuda.empty_cache()
    
    # Wrap the validation loader in a prefetcher if not already
    if not isinstance(val_loader, PrefetchLoader):
        val_loader = create_prefetch_loader(val_loader, device)
    
    total_samples = len(val_loader.dataset) if max_eval_samples is None else min(max_eval_samples, len(val_loader.dataset))
    batches_to_evaluate = (total_samples + val_loader.batch_size - 1) // val_loader.batch_size
    
    print(f"Evaluating on {total_samples} samples ({batches_to_evaluate} batches)...")
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if batch_idx >= batches_to_evaluate:
                break
            
            # If images are not already on device, move them (prefetcher should do this)
            if isinstance(images, list) and isinstance(images[0], torch.Tensor):
                images = [img.to(device, non_blocking=True) for img in images]
            
            predictions = model(images)
            
            for pred, target in zip(predictions, targets):
                cpu_pred = {
                    'boxes': pred['boxes'].cpu(),
                    'labels': pred['labels'].cpu(),
                    'scores': pred['scores'].cpu()
                }
                cpu_target = {
                    'boxes': target['boxes'],
                    'labels': target['labels']
                }
                all_predictions.append(cpu_pred)
                all_targets.append(cpu_target)
    
    mAP = calculate_map(all_predictions, all_targets)
    return mAP

class CustomSSD_Dataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None, use_minimal_aug=False, target_size=(320,320), cache_size=0):
        """
        Args:
            json_file (str): Path to the annotation JSON file.
            img_dir (str): Base path to the directory containing images.
            transform (callable, optional): Optional transform to be applied.
            use_minimal_aug (bool): Whether to use minimal augmentation.
            target_size (tuple): Target size for images.
            cache_size (int): Number of images to cache in memory (0 for no caching).
        """
        self.json_file = json_file
        self.img_dir = img_dir
        self.target_size = target_size
        self.cache_size = cache_size
        
        # Image cache
        self.img_cache = {}  # Dictionary to store cached images

        print(f"Initializing CustomSSD_Dataset with JSON: {json_file}")
        print(f"Image directory: {img_dir}")
        
        # Load JSON file (COCO format)
        try:
            with open(json_file, 'r') as f:
                self.data = json.load(f)
                print(f"Successfully loaded JSON file with {len(self.data.get('images', []))} images")
        except Exception as e:
            print(f"ERROR loading JSON file: {str(e)}")
            self.data = {"images": [], "annotations": [], "categories": []}

        # For simplicity, use img_dir as the base directory without any adjustments.
        self.base_dir = img_dir
            
        # Create category mapping
        self.categories = self.data.get("categories", [])
        self.category_mapping = {cat["id"]: cat["name"] for cat in self.categories}
        print(f"Found {len(self.category_mapping)} categories in annotation file")
        for cat_id, cat_name in self.category_mapping.items():
            print(f"  Category ID {cat_id}: {cat_name}")
        
        # Create mappings for images, skipping duplicates
        self.image_info = {}
        skipped_images = 0
        for img in self.data.get("images", []):
            img_id = img["id"]
            if img_id in self.image_info:
                skipped_images += 1
                continue
            self.image_info[img_id] = img
        print(f"Processed {len(self.image_info)} images, skipped {skipped_images} duplicates")
        
        # Organize annotations by image_id
        self.annotations = {}
        total_annotations = 0
        skipped_annotations = 0
        for ann in self.data.get("annotations", []):
            image_id = ann.get("image_id")
            if image_id is None or image_id not in self.image_info or "bbox" not in ann:
                skipped_annotations += 1
                continue
            self.annotations.setdefault(image_id, []).append(ann)
            total_annotations += 1
        print(f"Total valid annotations: {total_annotations} (skipped {skipped_annotations})")
        
        # Use all available image IDs for the dataset
        self.image_ids = list(self.image_info.keys())
        print(f"Including all {len(self.image_ids)} images in dataset")
        
        # Optionally, verify a few image paths (simple verification)
        self._verify_paths(num_samples=5)
        
        # Setup transformation: if use_minimal_aug is True, apply minimal augmentation; otherwise, basic resizing
        if transform is not None:
            self.transform = transform
        elif use_minimal_aug:
            self.transform = A.Compose([
                A.Resize(target_size[0], target_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(target_size[0], target_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['class_labels']))
    
    def _verify_paths(self, num_samples=5):
        """Verify a sample of image paths to ensure files exist"""
        import random
        if not self.image_ids:
            print("No image IDs available for path verification")
            return
            
        sample_indices = random.sample(range(len(self.image_ids)), min(num_samples, len(self.image_ids)))
        print("\nVerifying sample image paths:")
        for idx in sample_indices:
            image_id = self.image_ids[idx]
            img_metadata = self.image_info[image_id]
            file_name = img_metadata["file_name"]
            img_path = os.path.join(self.base_dir, file_name)
            exists = os.path.exists(img_path)
            print(f"Sample image {idx} (ID: {image_id}): Path: {img_path} - {'EXISTS' if exists else 'MISSING'}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def _validate_and_clip_bbox(self, bbox, img_width, img_height):
        """
        Validate and clip bbox coordinates to be within image boundaries.
        Args:
            bbox: List [xmin, ymin, xmax, ymax] 
            img_width: Width of the image
            img_height: Height of the image
        Returns:
            Validated bbox or None if invalid.
        """
        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, min(x_min, img_width - 1))
        y_min = max(0, min(y_min, img_height - 1))
        x_max = max(x_min + 1, min(x_max, img_width))
        y_max = max(y_min + 1, min(y_max, img_height))
        if x_max - x_min < 2 or y_max - y_min < 2:
            return None
        return [x_min, y_min, x_max, y_max]

    def _get_image_path(self, file_name):
        """Return the image path by joining the base directory and file name."""
        return os.path.join(self.base_dir, file_name)
        
    def _cache_image(self, img_id, image):
        """Cache image if cache is enabled and not full"""
        if self.cache_size > 0 and len(self.img_cache) < self.cache_size:
            self.img_cache[img_id] = image.copy()

    def _load_image(self, img_id, img_path):
        """Load image from cache or disk"""
        # Check if image is in cache
        if img_id in self.img_cache:
            return self.img_cache[img_id].copy()
            
        # Load from disk
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at {img_path}")
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Cache the image if caching is enabled
            self._cache_image(img_id, image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a small blank image as fallback
            blank = np.zeros((32, 32, 3), dtype=np.uint8)
            return blank

    def __getitem__(self, index):
        # Retrieve image metadata
        image_id = self.image_ids[index]
        img_metadata = self.image_info[image_id]
        file_name = img_metadata["file_name"]
        img_path = self._get_image_path(file_name)
        
        # Fallback path if primary path doesn't exist
        if not os.path.exists(img_path):
            img_path = os.path.join('/kaggle/input/smartcart-dataset', file_name)
        
        # Load image
        image = self._load_image(image_id, img_path)
        
        # Ensure 3 channels (RGB)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        img_height, img_width = image.shape[:2]
        bboxes = []
        labels = []
        
        # Process annotations for this image
        for ann in self.annotations.get(image_id, []):
            bbox = ann.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            x_min, y_min, width, height = bbox
            x_max, y_max = x_min + width, y_min + height
            valid_bbox = self._validate_and_clip_bbox([x_min, y_min, x_max, y_max], img_width, img_height)
            if valid_bbox is not None:
                bboxes.append(valid_bbox)
                labels.append(ann["category_id"])
        
        # Handle images with no bounding boxes
        if not bboxes:
            if self.transform:
                transformed = self.transform(image=image, bboxes=[], class_labels=[])
                image = transformed["image"]
            else:
                image = cv2.resize(image, self.target_size)
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            return image, {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64)
            }
        
        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transformations using albumentations
        try:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=labels)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["class_labels"]
            
            # If no bboxes remain after transformation, return empty targets
            if len(bboxes) == 0:
                return image, {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64)
                }
                
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            
            # Convert bbox from PASCAL VOC format (x1, y1, x2, y2) to normalized coordinates
            norm_bboxes = bboxes.clone()
            norm_bboxes[:, [0, 2]] /= self.target_size[1]  # width normalization
            norm_bboxes[:, [1, 3]] /= self.target_size[0]  # height normalization
            
            # Ensure all boxes are valid (0-1 range)
            valid_indices = ((norm_bboxes[:, 0] < norm_bboxes[:, 2]) & 
                            (norm_bboxes[:, 1] < norm_bboxes[:, 3]) &
                            (norm_bboxes[:, 0] >= 0) & (norm_bboxes[:, 1] >= 0) &
                            (norm_bboxes[:, 2] <= 1) & (norm_bboxes[:, 3] <= 1))
            
            if not valid_indices.any():
                return image, {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64)
                }
                
            norm_bboxes = norm_bboxes[valid_indices]
            labels = labels[valid_indices]
            
        except Exception as e:
            print(f"Transformation error for image {img_path}: {str(e)}")
            # Fallback transformation if an error occurs
            image = cv2.resize(image, self.target_size)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            return image, {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64)
            }
        
        target = {
            "boxes": norm_bboxes,
            "labels": labels
        }
        
        return image, target

def collate_fn(batch):
    """
    Custom collate function for SSDLite dataloader that handles invalid boxes.
    """
    images = []
    targets = []
    
    for image, target in batch:
        # Skip completely invalid samples
        if image is None or target is None:
            continue
            
        # Check for valid boxes (all coordinates between 0 and 1)
        if len(target["boxes"]) > 0:
            valid_boxes = []
            valid_labels = []
            
            for i, box in enumerate(target["boxes"]):
                # Check if box coordinates are valid
                if (0.0 <= box[0] <= 1.0 and 
                    0.0 <= box[1] <= 1.0 and 
                    0.0 <= box[2] <= 1.0 and 
                    0.0 <= box[3] <= 1.0 and
                    box[0] < box[2] and 
                    box[1] < box[3]):
                    valid_boxes.append(box)
                    valid_labels.append(target["labels"][i])
            
            if valid_boxes:
                # If we have valid boxes, update the target
                target["boxes"] = torch.stack(valid_boxes)
                target["labels"] = torch.tensor(valid_labels, dtype=torch.int64)
            else:
                # No valid boxes, create empty tensor
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros(0, dtype=torch.int64)
        
        images.append(image)
        targets.append(target)
    
    # If all samples were invalid, return empty batch
    if len(images) == 0:
        return [], []
        
    return images, targets

    
def unfreeze_model_progressively(model, epoch):
    """Coarse progressive unfreezing based on two high-level blocks."""
    if isinstance(model, torch.nn.DataParallel):
        base_model = model.module
    else:
        base_model = model
    # Assume model is not wrapped in DataParallel for now
    if not hasattr(model, 'backbone'):
        print(f"Epoch {epoch}: Could not locate backbone. Skipping unfreezing.")
        return
    
    backbone = model.backbone
    if not hasattr(backbone, 'features'):
        print(f"Epoch {epoch}: Backbone has no attribute 'features'. Skipping unfreezing.")
        return
    
    print(f"Epoch {epoch}: Unfreezing strategy in effect.")
    
    print(f"Epoch {epoch}: Unfreezing entire backbone.")
    for param in backbone.parameters():
            param.requires_grad = True
            
    # Wrap the updated model in DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print("Wrapping model into DataParallel.")
        model = torch.nn.DataParallel(base_model)
    else:
        model = base_model
        print("Using single GPU; no DataParallel wrapping needed.")

    return model

def create_transforms(target_size=(320, 320), is_training=True):
    """Create optimized transforms for SSD training and validation"""
    if is_training:
        return A.Compose([
            A.Resize(target_size[0], target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.GaussianBlur(blur_limit=3, p=0.1),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['class_labels']))
    else:
        return A.Compose([
            A.Resize(target_size[0], target_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['class_labels']))


def train_ssd(model, train_loader, val_loader, optimizer, device, num_epochs, scheduler, early_stopping, checkpoint_dir):
    """
    Train SSD model with optimized GPU utilization and reduced CPU bottlenecks
    """
    # Move model to device and wrap with DataParallel if multiple GPUs available
    model = model.to(device)
    
    # Initialize gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler()
    
    # Training metrics tracking
    history = {
        'train_loss': [],
        'val_mAP': [],
        'learning_rates': []
    }
    
    # Enable cuDNN benchmark for faster training
    torch.backends.cudnn.benchmark = True
    
    # Enable async data transfers when possible
    if isinstance(device, torch.device) and device.type == 'cuda':
        if device.index is None:
            torch.cuda.set_device(0)  # Default to first GPU
        else:
            torch.cuda.set_device(device.index)
        
    # Prepare for training loop
    best_mAP = 0.0
    
    # Pre-fetch size for validation mAP calculation to save time
    max_eval_samples = None  # Set to None for full evaluation
    
    # Gradient accumulation steps (useful for larger effective batch sizes)
    grad_accum_steps = 1
    
    # Main training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        unfreeze_model_progressively(model, epoch)
        
        # Training phase
        model.train()
        running_train_loss = 0.0
        processed_batches = 0
        
        # Use tqdm for progress bar if available
        try:
            from tqdm import tqdm
            batch_iterator = tqdm(enumerate(train_loader), total=len(train_loader), 
                                 desc=f"Epoch {epoch+1}/{num_epochs}")
        except ImportError:
            batch_iterator = enumerate(train_loader)
        
        # Reset gradients at the beginning of each epoch
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (images, targets) in batch_iterator:
            # Skip empty batches
            if len(images) == 0:
                continue
                
            # Move data to device
            images = [img.to(device, non_blocking=True) for img in images]
            targets_device = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            
            try:
                # Mixed precision training
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    # Forward pass 
                    loss_dict = model(images, targets_device)
                    # Sum up losses
                    losses = sum(loss for loss in loss_dict.values())
                    # Normalize loss for gradient accumulation
                    losses = losses / grad_accum_steps
                
                # Check for invalid loss values
                if torch.isnan(losses) or torch.isinf(losses):
                    print(f"WARNING: Loss is {losses}. Skipping this batch.")
                    continue
                
                # Backward pass with scaled gradients
                scaler.scale(losses).backward()
                
                # Only update weights after accumulating gradients
                if (batch_idx + 1) % grad_accum_steps == 0:
                    # Gradient clipping with scaled gradients
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update weights with scaled gradients
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                
                # Track loss (using the original non-normalized loss value)
                batch_loss = losses.item() * grad_accum_steps
                running_train_loss += batch_loss
                processed_batches += 1
                
                # Update progress bar if using tqdm
                if hasattr(batch_iterator, 'set_postfix'):
                    batch_iterator.set_postfix(loss=batch_loss)
                elif (batch_idx + 1) % 25 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], "
                          f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                          f"Batch Loss: {batch_loss:.4f}")
                    
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        
        # Calculate average training loss
        epoch_train_loss = running_train_loss / processed_batches if processed_batches > 0 else float('inf')
        history['train_loss'].append(epoch_train_loss)
        
        # Log learning rate
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            history['learning_rates'].append(current_lr)
            break
        
        # Validation phase - calculate mAP
        print("Evaluating on validation set...")
        val_mAP = evaluate_model(model, val_loader, device, max_eval_samples)
        history['val_mAP'].append(val_mAP)
        
        # For consistency with early stopping (lower is better)
        val_loss = 1 - val_mAP
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_time:.2f}s")
        print(f"Train Loss: {epoch_train_loss:.4f}, Validation mAP: {val_mAP:.4f}")
        
        # Update scheduler
        scheduler.step()
        
        # Check early stopping
        improved = early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
        
        # Save checkpoint if improved
        if val_mAP > best_mAP:
            best_mAP = val_mAP
            checkpoint_path = os.path.join(checkpoint_dir, f'ssdlite_epoch_{epoch + 1}_mAP_{val_mAP:.4f}.pth')
            if isinstance(model, DataParallel):
                torch.save(model.module.state_dict(), checkpoint_path)
            else:
                torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved at {checkpoint_path}")
            
        # Clear GPU cache periodically
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
    
    return history



def create_optimized_data_loader(dataset, batch_size, is_training=True, num_workers=None):
    """
    Create optimized dataloader with proper settings for GPU usage
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        is_training: Whether this is a training dataloader (for shuffling)
        num_workers: Number of workers for data loading
        
    Returns:
        PyTorch DataLoader
    """
    # Calculate optimal number of workers if not specified
    if num_workers is None:
        num_workers = min(8, os.cpu_count())
    
    # Use larger prefetch factor for GPU training
    prefetch_factor = 2 if is_training else 1
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=is_training,  # Drop incomplete batches during training for more consistent batch sizes
    )

class WarmupMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_epochs=3, warmup_factor=0.1, last_epoch=-1):
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # MultiStepLR behavior after warmup
            return [base_lr * self.gamma ** len([m for m in self.milestones if m <= self.last_epoch])
                    for base_lr in self.base_lrs]

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check GPU count for parallel training
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # Clear GPU cache at the beginning
    torch.cuda.empty_cache()
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set up logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Set CUDA parameters for optimization
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
    try:
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
        torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for convolutions
    except:
        logger.info("TF32 optimization not supported on this GPU")
    
    # Load the SSDLite320 model with pre-trained weights
    logger.info("Loading base SSDLite model...")
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    )
    
    # Get number of classes from annotation file
    json_file_path = "/kaggle/input/smartcart-dataset/ImageAnnotation_augmented.json"
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    num_classes = len(data["categories"]) + 1  # +1 for background
    logger.info(f"Number of classes: {num_classes}")
    
    # Modify the classification head for your number of classes
    in_channels = [672, 480, 512, 256, 256, 128]  # Based on the architecture
    num_anchors = model.anchor_generator.num_anchors_per_location()
    
    # Replace classification head for new number of classes
    logger.info("Replacing classification head...")
    classification_head = SSDLiteClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=torch.nn.BatchNorm2d  # Added normalization layer
    )
    model.head.classification_head = classification_head
    """
    checkpoint = torch.load("/kaggle/working/ssdlite_checkpoints/ssdlite_epoch_4_mAP_0.0012.pth", map_location=device, weights_only = True)
    
    # Load only model weights
    model.load_state_dict(checkpoint)
    """
    # Target size for SSDLite
    target_size = (320, 320)
    
    # Create transforms
    train_transform = create_transforms(target_size=target_size, is_training=True)
    val_transform = create_transforms(target_size=target_size, is_training=False)
    
    # Create training and validation datasets
    logger.info("Creating datasets...")
    train_dataset = CustomSSD_Dataset(
        json_file="/kaggle/input/smartcart-dataset/ImageAnnotation_augmented.json",
        img_dir="/kaggle/input/smartcart-dataset/",
        transform=train_transform,
        target_size=target_size
    )
    
    # Split dataset - use a fixed seed for reproducibility
    val_percent = 0.15
    val_size = int(len(train_dataset) * val_percent)
    train_size = len(train_dataset) - val_size
    
    # Create a more strict split with generator
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        train_dataset, 
        [train_size, val_size], 
        generator=generator
    )
    
    # Calculate optimal batch size based on GPU memory
    batch_size = 27 
    logger.info(f"Using batch size: {batch_size}")
    
    # Calculate optimal number of workers
    num_workers = 4
    logger.info(f"Using {num_workers} dataloader workers")
    
    # Create optimized dataloaders
    train_loader = create_optimized_data_loader(
        train_subset, 
        batch_size=batch_size, 
        is_training=True,
        num_workers=num_workers
    )
    
    val_loader = create_optimized_data_loader(
        val_subset, 
        batch_size=batch_size*2,  # Use larger batches for validation
        is_training=False,
        num_workers=num_workers
    )
    
    # Move model to device and enable DataParallel if multiple GPUs are available
    model = model.to(device)
    num_epochs = 20
    # Use mixed precision training by default on supported GPUs
    use_amp = device.type == 'cuda' and torch.cuda.is_available()
    logger.info(f"Using mixed precision training: {use_amp}")
    
    # Create directory for checkpoints
    checkpoint_dir = "ssdlite_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup optimizer with layer-specific learning rates
    # Use higher learning rate for the classification head, lower for backbone
    optimizer = torch.optim.SGD([
    {'params': model.backbone.parameters(), 'lr': 0.001},  # Lower LR for backbone
    {'params': model.head.parameters(), 'lr': 0.005}       # Higher LR for detection head
    ], momentum=0.9, weight_decay=5e-4)

# Use custom scheduler with warmup
    scheduler = WarmupMultiStepLR(
    optimizer,
    milestones=[10, 15],  # Reduce LR at epochs 10 and 15
    gamma=0.1,           # Reduce by factor of 10
    warmup_epochs=3,     # 3 epochs of warmup
    warmup_factor=0.1    # Start with 10% of base LR and linearly increase to full LR
    )
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=8,
        min_delta=0.001,
        verbose=True,
        path=os.path.join(checkpoint_dir, "best_model.pth")
    )
    
    # Number of training epochs
    
    
    # Prefetch loaders for faster data transfer
    if device.type == 'cuda':
        train_loader = PrefetchLoader(train_loader, device)
        val_loader = PrefetchLoader(val_loader, device)
    
    # Train the model
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    history = train_ssd(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        scheduler=scheduler,
        early_stopping=early_stopping,
        checkpoint_dir=checkpoint_dir
    )
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
    if isinstance(model, DataParallel):
        torch.save(model.module.state_dict(), final_model_path)
    else:
        torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Final evaluation
    logger.info("Performing final evaluation...")
    final_mAP = evaluate_model(model, val_loader, device)
    logger.info(f"Final mAP: {final_mAP:.4f}")
    
    # Save training history
    with open(os.path.join(checkpoint_dir, "training_history.json"), "w") as f:
        json.dump({
            "train_loss": [float(loss) for loss in history["train_loss"]],
            "val_mAP": [float(mAP) for mAP in history["val_mAP"]],
            "learning_rates": [float(lr) for lr in history["learning_rates"]]
        }, f)
    
    # Plot training history if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        # Plot training loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(history["train_loss"])
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        # Plot validation mAP
        plt.subplot(1, 3, 2)
        plt.plot(history["val_mAP"])
        plt.title("Validation mAP")
        plt.xlabel("Epoch")
        plt.ylabel("mAP")
        
        # Plot learning rate
        plt.subplot(1, 3, 3)
        plt.plot(history["learning_rates"])
        plt.title("Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        
        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_dir, "training_history.png"))
        plt.close()
    except:
        logger.info("Matplotlib not available, skipping plot generation")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()