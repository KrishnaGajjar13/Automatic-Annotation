import torch
import torchvision
import cv2
import json
import numpy as np
import random
import os, time
torch.serialization.add_safe_globals([torchvision.models.detection.ssd.SSD])

StartTime = time.time()

# Load SSD Model for CPU
"""
model = torchvision.models.detection.ssd300_vgg16(weights=None)
num_classes = 10  # Number of classes (10 objects + 1 background)
in_channels = [512, 1024, 512, 256, 256, 256]

# Modify SSD Classification Head
classification_head = torchvision.models.detection.ssd.SSDClassificationHead(
    in_channels=in_channels,
    num_anchors=[4, 6, 6, 6, 4, 4],
    num_classes=num_classes
)
model.head.classification_head = classification_head

# Freeze Backbone to Speed Up CPU Execution
for param in model.backbone.parameters():
    param.requires_grad = False

"""

# Load Model Weights (on CPU)
device = torch.device('cpu')
model = torch.load("Models/ssd_lite.pth", map_location=device, weights_only=False)


# Load Class Names from JSON
with open("ImageAnnotation_augmented.json", "r") as f:
    data = json.load(f)
classNames = [category["name"] for category in data["categories"]]

# Get a random image from the annotation file
image_annotations = data["images"]
random_image_info = random.choice(image_annotations)
image_path = random_image_info["file_name"]

# Load Image
image = cv2.imread(image_path)
if image is None:
    print(f"Failed to load image: {image_path}")
    exit()

original_height, original_width = image.shape[:2]

# Resize & Normalize Image for SSD
img_resized = cv2.resize(image, (300, 300))  # Resize for SSD input
img_tensor = torch.from_numpy(img_resized).float() / 255.0  # Normalize to [0,1]
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # Convert HWC -> CHW
img_tensor = img_tensor.to(device)

# Run Model Inference
with torch.no_grad():
    detections = model(img_tensor)

# Process Detections
boxes = detections[0]['boxes'].cpu().numpy()
scores = detections[0]['scores'].cpu().numpy()
labels = detections[0]['labels'].cpu().numpy()

CONF_THRESHOLD = 0.75  # Minimum confidence threshold

# Draw Bounding Boxes on Image
for i in range(len(boxes)):
    if scores[i] < CONF_THRESHOLD:
        continue

    label_idx = labels[i]
    if label_idx >= len(classNames):
        label_idx = len(classNames) - 1

    # Scale Bounding Box Back to Original Image Size
    x_min, y_min, x_max, y_max = boxes[i]
    x_min = int(x_min * original_width)
    y_min = int(y_min * original_height)
    x_max = int(x_max * original_width)
    y_max = int(y_max * original_height)

    print(f"Drawing bbox: {x_min, y_min, x_max, y_max} for {classNames[label_idx]} with confidence {scores[i]:.2f}")

    # Draw Bounding Box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    label_text = f"{classNames[label_idx]}: {scores[i]:.2f}"
    cv2.putText(image, label_text, (x_min, max(25, y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print(f"Predicted Class: {classNames[label_idx]}, Confidence: {scores[i]:.2f}")

# Create Output Folder
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

# Save Processed Image
output_path = os.path.join(output_folder, f"output_{os.path.basename(image_path)}")
cv2.imwrite(output_path, image)
print(f"Saved output image to {output_path}")

EndTime = time.time()
print("Inference Time:" ,EndTime - StartTime)