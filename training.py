import torch, torchvision, torch.multiprocessing as mp
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.models.detection import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead, SSDHead
from torch.utils.data import Dataset, DataLoader, random_split
import json, os, cv2, numpy as np
from PIL import Image as img
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
from torch.nn.parallel import DataParallel

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

def calculate_class_AP(class_det_boxes, class_det_scores, class_true_boxes, iou_threshold=0.5):
    """
    Calculate Average Precision for a specific class.
    
    Args:
        class_det_boxes: List of tensors, detected boxes for this class across all images
        class_det_scores: List of tensors, confidence scores for this class across all images
        class_true_boxes: List of tensors, ground truth boxes for this class across all images
        iou_threshold: IoU threshold to determine a true positive
        
    Returns:
        float: Average Precision value for this class
    """
    # Flatten all detections and scores
    all_detections = []
    all_scores = []
    for i in range(len(class_det_boxes)):
        if len(class_det_boxes[i]) > 0:
            all_detections.append(class_det_boxes[i])
            all_scores.append(class_det_scores[i])
    
    if not all_detections:  # No detections for this class
        return 0.0
    
    # Concatenate if there are detections
    all_detections = torch.cat(all_detections, dim=0) if all_detections else torch.tensor([])
    all_scores = torch.cat(all_scores, dim=0) if all_scores else torch.tensor([])
    
    # Count total number of ground truths across all images
    n_gt = sum(boxes.size(0) for boxes in class_true_boxes)
    
    if n_gt == 0:  # No ground truth for this class
        return 0.0
    
    if all_detections.size(0) == 0:  # No detections for this class
        return 0.0
    
    # Sort detections by score (highest first)
    _, sort_ind = torch.sort(all_scores, descending=True)
    all_detections = all_detections[sort_ind]
    all_scores = all_scores[sort_ind]
    
    # Initialize true positives and false positives arrays
    tp = torch.zeros(all_detections.size(0))
    fp = torch.zeros(all_detections.size(0))
    
    # Mark which ground truths have been detected
    detected_gt = [torch.zeros(boxes.size(0)) for boxes in class_true_boxes]
    
    # Go through all detections and mark TPs and FPs
    for d in range(all_detections.size(0)):
        det_box = all_detections[d]
        max_iou = 0.0
        max_idx = -1
        max_img_idx = -1
        
        # Find the ground truth with the highest IoU for this detection
        for img_idx, gt_boxes in enumerate(class_true_boxes):
            if gt_boxes.size(0) == 0:
                continue
                
            for g in range(gt_boxes.size(0)):
                # If this ground truth has already been detected, skip
                if detected_gt[img_idx][g] == 1:
                    continue
                    
                iou = calculate_iou(det_box, gt_boxes[g])
                if iou > max_iou:
                    max_iou = iou
                    max_idx = g
                    max_img_idx = img_idx
        
        # Assign detection as true positive or false positive
        if max_iou >= iou_threshold:
            if detected_gt[max_img_idx][max_idx] == 0:
                tp[d] = 1  # True positive
                detected_gt[max_img_idx][max_idx] = 1  # Mark as detected
            else:
                fp[d] = 1  # False positive (already detected)
        else:
            fp[d] = 1  # False positive (no matching ground truth)
    
    # Compute precision and recall
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    recalls = tp_cumsum / n_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # Append start and end points for integration
    recalls = torch.cat((torch.tensor([0]), recalls, torch.tensor([1])))
    precisions = torch.cat((torch.tensor([0]), precisions, torch.tensor([0])))
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = torch.max(precisions[i], precisions[i + 1])
    
    # Compute average precision (area under PR curve)
    indices = torch.where(recalls[1:] != recalls[:-1])[0]
    ap = torch.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap.item()

def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, iou_threshold=0.5):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.
    
    Args:
        det_boxes: List of tensors, each tensor containing the detected boxes for an image
        det_labels: List of tensors, each tensor containing the detected labels for an image
        det_scores: List of tensors, each tensor containing the confidence scores for an image
        true_boxes: List of tensors, each tensor containing the ground truth boxes for an image
        true_labels: List of tensors, each tensor containing the ground truth labels for an image
        iou_threshold: IoU threshold to determine a true positive
        
    Returns:
        tuple: (APs, mAP) - Average Precision for each class and the mean Average Precision
    """
    # Determine number of classes (assume labels start from 0 or 1)
    n_classes = 0
    for labels in det_labels + true_labels:
        if labels.numel() > 0:
            n_classes = max(n_classes, labels.max().item() + 1)
    
    # Initialize AP for each class
    APs = [0.] * n_classes
    
    # Calculate AP for each class
    for c in range(1, n_classes):  # Typically class 0 is background in object detection
        # Get all detections and ground truths for this class
        class_det_boxes = [det_boxes[i][det_labels[i] == c] for i in range(len(det_boxes))]
        class_det_scores = [det_scores[i][det_labels[i] == c] for i in range(len(det_scores))]
        class_true_boxes = [true_boxes[i][true_labels[i] == c] for i in range(len(true_boxes))]
        
        # Calculate AP for this class
        APs[c] = calculate_class_AP(class_det_boxes, class_det_scores, class_true_boxes, iou_threshold)
    
    # Calculate mAP (mean of APs for all classes except background)
    valid_classes = [c for c in range(1, n_classes) if sum(boxes.size(0) for boxes in 
                                                         [true_boxes[i][true_labels[i] == c] 
                                                          for i in range(len(true_boxes))]) > 0]
    
    if len(valid_classes) == 0:
        mAP = 0.0
    else:
        mAP = sum(APs[c] for c in valid_classes) / len(valid_classes)
    
    return APs, mAP

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model using mean Average Precision (mAP).
    
    Args:
        model: The SSD model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
    
    Returns:
        float: The mean Average Precision (mAP)
    """
    # Make sure model is in eval mode
    model.eval()
    
    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            
            # Get predictions for this batch
            predictions = model(images)
            
            # Process each image in the batch
            for i in range(len(predictions)):
                pred = predictions[i]
                # Get ground truth
                target = targets[i]
                
                # Store predictions
                det_boxes.append(pred['boxes'])
                det_scores.append(pred['scores'])
                det_labels.append(pred['labels'])
                
                # Store ground truth
                true_boxes.append(target['boxes'].to(device))
                true_labels.append(target['labels'].to(device))
                
            if (batch_idx + 1) % 10 == 0:
                print(f"Evaluation: processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Calculate mAP
    APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels)
    
    # Print AP for each class
    for c, ap in enumerate(APs):
        if ap > 0:  # Only print classes that have AP > 0
            print(f"AP for class {c}: {ap:.4f}")
    
    print(f"Mean Average Precision (mAP): {mAP:.4f}")
    
    return mAP
class CustomSSD_Dataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        """
        Args:
            json_file (str): Path to the annotation JSON file.
            img_dir (str): Path to the directory containing images.
            transform (albumentations.Compose, optional): Data augmentation transformations.
        """
        self.img_dir = img_dir
        self.transform = transform

        # Load JSON file
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        # Create a mapping from image_id to image metadata
        self.image_info = {img["id"]: img for img in self.data["images"]}

        # Organize annotations by image_id
        self.annotations = {}
        for ann in self.data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in self.annotations:
                self.annotations[image_id] = []
            self.annotations[image_id].append(ann)

        # Create a mapping from category_id to category name
        self.category_mapping = {cat["id"]: cat["name"] for cat in self.data["categories"]}

        # List of image IDs
        self.image_ids = list(self.image_info.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        """
        Loads an image and its corresponding bounding boxes, applies augmentations, and returns a tensor.
        """
        image_id = self.image_ids[index]
        img_metadata = self.image_info[image_id]
        file_name = img_metadata["file_name"]

        # Check if file_name already includes the img_dir
        if file_name.startswith(self.img_dir):
            img_path = file_name
        else:
            img_path = os.path.join(self.img_dir, file_name)

        # Try to load the image
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a placeholder to avoid breaking the batch
            placeholder_image = np.zeros((300, 300, 3), dtype=np.uint8)
            return torch.zeros((3, 300, 300), dtype=torch.float32), torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

        # Get bounding boxes and labels
        bboxes = []
        labels = []
        for ann in self.annotations.get(image_id, []):
            bbox = ann["bbox"]
            x_min, y_min, width, height = bbox
            # Ensure coordinates are valid
            x_max, y_max = x_min + width, y_min + height
            if x_min >= 0 and y_min >= 0 and x_max > x_min and y_max > y_min:
                bboxes.append([x_min, y_min, x_max, y_max])
                labels.append(ann["category_id"])

        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Apply augmentations
        if self.transform and len(bboxes) > 0:
            try:
                transformed = self.transform(image=image, bboxes=bboxes, class_labels=labels)
                image = transformed["image"]
                bboxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
                labels = torch.tensor(transformed["class_labels"], dtype=torch.int64)
            except Exception as e:
                print(f"Augmentation error for image {img_path}: {str(e)}")
                # If augmentation fails, use the original image
                image = cv2.resize(image, (300, 300))
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                bboxes = torch.tensor(bboxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)
        else:
            # Resize and convert to tensor if no augmentations or no boxes
            image = cv2.resize(image, (300, 300))
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            if len(bboxes) > 0:
                bboxes = torch.tensor(bboxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)
            else:
                bboxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)

        return image, bboxes, labels

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


"""def get_transforms():
    return A.Compose([
        # Resize all images to 300x300
        A.Resize(300, 300),
        
        # Random flip horizontally
        A.HorizontalFlip(p=0.5),
        
        # Adjust brightness and contrast
        A.RandomBrightnessContrast(p=0.3),
        
        # Random 90-degree rotation
        A.RandomRotate90(p=0.3),
        
        # Apply Gaussian Blur
        A.GaussianBlur(p=0.2),
        
        # Adding random noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2), # type: ignore
        
        # Random scaling of the image
        A.RandomScale(scale_limit=0.3, p=0.5),
        
        # Random cropping
        A.RandomCrop(width=250, height=250, p=0.4),
        
        # Elastic transform to simulate distortions
        A.ElasticTransform(alpha=1.0, sigma=50, p=0.3),
        
        # Color jittering (hue, saturation, brightness, contrast)
        A.HueSaturationValue(p=0.3),
        
        # Random sharpen (simulate camera focus)
        A.Sharpen(p=0.2),
        
        # Random hue/saturation adjustments for more color variation
        A.HueSaturationValue(p=0.2),
        
        # Normalize using ImageNet values (if your model was pretrained on ImageNet)
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        
        # Convert to tensor for model input
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))  # Use 'coco' format for [xmin, ymin, width, height]
    
    return transform
"""
def collate_fn(batch):
    """
    Custom collate function to handle variable sized boxes and empty boxes
    """
    images = []
    targets = []
    
    for image, boxes, labels in batch:
        images.append(image)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        targets.append(target)
    
    images = torch.stack(images, 0)
    return images, targets

def train_ssd(model, train_loader, val_loader, optimizer, device, num_epochs, scheduler):
    # Create directory for checkpoints
    checkpoint_dir = 'ssd_model_checkpoints/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, verbose=True, path='best_ssd_model.pth')
    
    # Training metrics tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    # Use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # -----------------
        # Training Phase
        # -----------------
        model.train()
        running_train_loss = 0.0
        batch_losses = []
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets_list = []
            
            for target in targets:
                target_dict = {
                    "boxes": target["boxes"].to(device),
                    "labels": target["labels"].to(device)
                }
                targets_list.append(target_dict)
            
            optimizer.zero_grad()
            
            try:
                loss_dict = model(images, targets_list)
                losses = sum(loss for loss in loss_dict.values())
                
                # Check for NaN or Inf values
                if torch.isnan(losses) or torch.isinf(losses):  # type: ignore
                    print(f"WARNING: Loss is {losses}. Skipping this batch.")
                    continue
                
                losses.backward() # type: ignore
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                optimizer.step()
                
                batch_loss = losses.item()  # type: ignore
                running_train_loss += batch_loss
                batch_losses.append(batch_loss)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Train Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {batch_loss:.4f}")
                    
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        
        epoch_train_loss = running_train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        history['train_loss'].append(epoch_train_loss)
        
        # Log current learning rate
        for param_group in optimizer.param_groups:
            history['learning_rates'].append(param_group['lr'])
            break  # Just log the first group's learning rate
        
        # -----------------
        # Validation Phase
        # -----------------
        model.eval()
        mAP = evaluate_model(model, val_loader, device)
        
        # Use mAP as validation metric (higher is better, so negate for optimizer)
        epoch_val_loss = -mAP  # Lower is better for optimize
        history['val_loss'].append(epoch_val_loss)
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_time:.2f}s")
        print(f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)
        
        # Check early stopping
        improved = early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        
        # Save checkpoint if improved
        if improved:
            checkpoint_path = os.path.join(checkpoint_dir, f'ssd_epoch_{epoch + 1}_{epoch_val_loss:.4f}.pth')
            if isinstance(model, DataParallel):
                torch.save(model.module.state_dict(), checkpoint_path)
            else:
                torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
    
    print("Training finished.")
    return history

if __name__ == "__main__":
    try:
        # Set up multi-GPU environment
        torch.multiprocessing.set_sharing_strategy('file_system')
        
        # Check GPU availability
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        
        if num_gpus > 0:
            for i in range(num_gpus):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Create the model
        model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
        
        # Modify the classification head for your number of classes
        num_classes = 11  # Change this based on your dataset
        in_channels = [512, 1024, 512, 256, 256, 256]
        classification_head = SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=[4, 6, 6, 6, 4, 4],
            num_classes=num_classes
        )
        model.head.classification_head = classification_head
        
        # Freeze backbone for transfer learning
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training device: {device}")
        
        # Dataset creation and train-validation split
        dataset = CustomSSD_Dataset(
            json_file="image_annotation_2.json",
            img_dir="",
        )
        
        # Define train-validation split
        val_percent = 0.20
        val_size = int(len(dataset) * val_percent)
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(69)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        
        # Create data loaders
        batch_size = 16
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size * 2,
            shuffle=True,
            pin_memory=True,
            num_workers=4 * max(1, num_gpus),
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=4 * max(1, num_gpus),
            collate_fn=collate_fn
        )
        
        # Set up optimizer
        learning_rate = 0.001
        params = [
            {'params': [p for n, p in model.named_parameters() if "classification_head" not in n], 'lr': learning_rate},
            {'params': [p for n, p in model.named_parameters() if "classification_head" in n], 'lr': learning_rate * 10}
        ]
        
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.25, min_lr=1e-6
        )
        
        # Train the model
        num_epochs = 40
        history = train_ssd(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            scheduler=scheduler
        )
        
        # Save the final model
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), "SSD_final_model.pth")
        else:
            torch.save(model.state_dict(), "SSD_final_model.pth")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

