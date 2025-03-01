import torch, torchvision, torch.multiprocessing as mp
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.models.detection import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead, SSDHead
from torch.utils.data import Dataset, DataLoader,random_split
import json, os, cv2, numpy as np
from PIL import Image as img
import albumentations as A
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

        # Debug: Print the image path
        #print("Loading image from:", img_path)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get bounding boxes and labels
        bboxes = []
        labels = []
        for ann in self.annotations.get(image_id, []):
            bbox = ann["bbox"]
            x_min, y_min, width, height = bbox
            x_max, y_max = x_min + width, y_min + height
            bboxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann["category_id"])

        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=labels)
            image = transformed["image"]
            bboxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["class_labels"], dtype=torch.int64)

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
            torch.save(model.state_dict(), self.path)
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.4f}. Saving model to {self.path}.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation loss. Counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# Define Static Augmentations
def get_transforms():
    return A.Compose([
        A.Resize(300, 300),  # Resize all images to 300x300
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomRotate90(p=0.3),
        A.GaussianBlur(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize for SSD
        ToTensorV2()  # Convert to tensor
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))  # SSD uses Pascal VOC format

def train_ssd(model, train_loader, val_loader, optimizer, device, num_epochs, scheduler):
    checkpoint_dir = 'ssd_model_checkpoints/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, verbose=True, path='best_ssd_model.pth')

    for epoch in range(num_epochs):
        # -----------------
        # Training Phase
        # -----------------
        model.train()
        running_train_loss = 0.0
        
        for batch_idx, (images, bboxes, labels) in enumerate(train_loader):
            images = images.to(device)
            targets_list = []
            for boxes, lbl in zip(bboxes, labels):
                target = {
                    "boxes": boxes.to(device),
                    "labels": lbl.to(device)
                }
                targets_list.append(target)
            
            optimizer.zero_grad()
            loss_dict = model(images, targets_list)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward() # type: ignore
            optimizer.step()
            
            running_train_loss += loss.item() # type: ignore
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Train Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")  # type: ignore
        
        epoch_train_loss = running_train_loss / len(train_loader)
        print(f"Train Epoch [{epoch + 1}/{num_epochs}] completed with average loss: {epoch_train_loss:.4f}")
        
        # -----------------
        # Validation Phase
        # -----------------
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (images, bboxes, labels) in enumerate(val_loader):
                images = images.to(device)
                targets_list = []
                for boxes, lbl in zip(bboxes, labels):
                    target = {
                        "boxes": boxes.to(device),
                        "labels": lbl.to(device)
                    }
                    targets_list.append(target)
                
                loss_dict = model(images, targets_list)
                loss = sum(loss for loss in loss_dict.values())
                running_val_loss += loss.item() # type: ignore
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Val Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(val_loader)}], Loss: {loss.item():.4f}")  # type: ignore
        
        epoch_val_loss = running_val_loss / len(val_loader)
        print(f"Validation Epoch [{epoch + 1}/{num_epochs}] completed with average loss: {epoch_val_loss:.4f}")

        scheduler.step(epoch_val_loss)
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        # -----------------
        # Checkpointing & Scheduler Step
        # -----------------
        checkpoint_path = os.path.join(checkpoint_dir, f'ssd_epoch_{epoch + 1}_{epoch_val_loss:.4f}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    
    print("Training finished.")


# Usage Example
if __name__ == "__main__":
    model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
    #torch.save(ssd,"ssd300_vgg16_coco-b556d3b4.pth")
    num_classes = 11
    in_channels = [512, 1024, 512, 256, 256, 256]
    classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=[4, 6, 6, 6, 4, 4],
        num_classes=num_classes
    )
    model.head.classification_head = classification_head
    for param in model.backbone.parameters():
        param.requires_grad = False

    #print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f"Training device: {device}")

    dataset = CustomSSD_Dataset(
    json_file="image_annotation.json",
    img_dir="label_images/",
    transform=get_transforms()
    )

    # Define the proportion for validation (e.g., 20% for validation)
    val_percent = 0.20  # Change to 0.205 if you specifically want 20.5% validation
    val_size = int(len(dataset) * val_percent)
    train_size = len(dataset) - val_size

    # Randomly split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Total images: {len(dataset)}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
        
    train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True, pin_memory=True,num_workers = 4)
    val_loader = DataLoader(val_dataset,batch_size=32,shuffle=True, pin_memory=True,num_workers = 4)
    # Example usage
    """
    image, bboxes, labels = dataset[0]
    print("Image shape:", image.shape)
    print("Bounding Boxes:", bboxes)
    print("Labels:", labels)
    """
    learning_rate = 0.001  # Example learning rate, adjust as needed
    params = [
        {'params': [p for n, p in model.named_parameters() if "classification_head" not in n], 
        'lr': learning_rate},
        {'params': [p for n, p in model.named_parameters() if "classification_head" in n], 
        'lr': learning_rate}
    ]
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.258207361906123)

    # Example usage
    num_epochs = 20
    history = train_ssd(
        model=model,
        train_loader=train_loader,
        val_loader = val_loader, 
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        scheduler=scheduler
    )
    torch.save(model, "SSD_demo.pth")