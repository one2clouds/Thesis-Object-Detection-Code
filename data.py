import glob 
import os 
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import numpy as np 
from PIL import Image
import torch 
from torch.utils.data import Subset


class CustomDataset(Dataset):
    def __init__(self, image_paths, label_paths, img_size, transforms=None):
        self.transforms = transforms
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.img_size = img_size
        
        # Get all image files
        self.all_images = glob.glob(os.path.join(self.image_paths, "*.jpg"))
        self.all_images = sorted(self.all_images)
        
        # Get corresponding label files
        self.all_labels = []
        for img_path in self.all_images:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.label_paths, f"{img_name}.txt")
            self.all_labels.append(label_path)
    
    def polygon_to_bbox(self, polygon_points):
        """
        Convert polygon points to bounding box coordinates
        Args:
            polygon_points: List of [x1, y1, x2, y2, ..., xn, yn] coordinates
        Returns:
            [x_min, y_min, x_max, y_max] bounding box coordinates
        """
        # Reshape points into pairs of (x, y)
        points = np.array(polygon_points).reshape(-1, 2)
        
        # Find min and max coordinates
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        x_min = np.min(x_coords)
        y_min = np.min(y_coords)
        x_max = np.max(x_coords)
        y_max = np.max(y_coords)
        
        return [x_min, y_min, x_max, y_max]
    
    def parse_polygon_annotation(self, line, img_width, img_height):
        """
        Parse a single line of polygon annotation
        Args:
            line: String containing "class_id x1 y1 x2 y2 x3 y3 ... xn yn"
            img_width, img_height: Image dimensions for coordinate conversion
        Returns:
            class_id, bounding_box coordinates
        """
        parts = line.strip().split()
        if len(parts) < 7:  # At least class_id + 3 points (6 coordinates)
            return None, None
            
        class_id = int(parts[0])
        
        # Extract polygon coordinates (assuming they are normalized 0-1)
        polygon_coords = list(map(float, parts[1:]))
        
        # Convert normalized coordinates to pixel coordinates
        pixel_coords = []
        for i in range(0, len(polygon_coords), 2):
            x = polygon_coords[i] * img_width
            y = polygon_coords[i + 1] * img_height
            pixel_coords.extend([x, y])
        
        # Convert polygon to bounding box
        bbox = self.polygon_to_bbox(pixel_coords)
        
        return class_id, bbox
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.all_images[idx]
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size
        
        # Load labels
        label_path = self.all_labels[idx]
        boxes = []
        labels = []
        polygon_masks = []  # Store original polygon data if needed later
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if line:
                    class_id, bbox = self.parse_polygon_annotation(line, img_width, img_height)
                    
                    if class_id is not None and bbox is not None:
                        # Ensure bbox coordinates are valid
                        x_min, y_min, x_max, y_max = bbox
                        
                        # Clamp coordinates to image boundaries
                        x_min = max(0, min(x_min, img_width - 1))
                        y_min = max(0, min(y_min, img_height - 1))
                        x_max = max(0, min(x_max, img_width))
                        y_max = max(0, min(y_max, img_height))
                        
                        # Ensure valid box (x_max > x_min and y_max > y_min)
                        if x_max > x_min and y_max > y_min:
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(class_id + 1)  # +1 because class 0 is reserved for background
                            
                            # Store original polygon data for future use (optional)
                            polygon_data = list(map(float, line.split()[1:]))
                            polygon_masks.append(polygon_data)
        
        # Handle case where no valid annotations exist
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            # Calculate areas
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # Apply transforms to image
        if self.transforms:
            image = self.transforms(image)
        else:
            # Default transform: convert PIL to tensor
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ])
            image = transform(image)
        
        # Create target dictionary for Faster R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx], dtype=torch.int64),
            'area': areas,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64) if len(boxes) > 0 else torch.tensor([], dtype=torch.int64),
            # Optional: store original polygon data for advanced use cases
            'polygons': polygon_masks  # This won't be used by Faster R-CNN but available for other purposes
        }
        
        return image, target
    
    def __len__(self):
        return len(self.all_images)

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))


# Visualization helper function (optional)
def visualize_polygon_to_bbox_conversion(dataset, idx):
    """
    Helper function to visualize how polygons are converted to bounding boxes
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    image, target = dataset[idx]
    
    # Convert tensor image back to PIL for visualization
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image_np)
    
    # Draw bounding boxes
    for i, box in enumerate(target['boxes']):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        
        rect = patches.Rectangle((x_min, y_min), width, height, 
                               linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        label = target['labels'][i].item() - 1  # Convert back to original class id
        ax.text(x_min, y_min - 5, f'Class {label}', color='red', fontsize=12, weight='bold')
    
    ax.set_title(f'Polygon to Bounding Box Conversion - Image {idx}')
    plt.axis('off')
    plt.show()
    



def get_data(TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS, batch_size):

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=3),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.01),
        transforms.RandomAutocontrast(p=0.01),
        transforms.RandomEqualize(p=0.01),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),    
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    train_dataset = CustomDataset(TRAIN_IMAGES, TRAIN_LABELS, (640, 640), transforms=train_transforms)
    val_dataset = CustomDataset(VAL_IMAGES, VAL_LABELS, (640,640), transforms=val_transforms)

    # train_dataset = Subset(train_dataset, [0])
    # val_dataset = Subset(train_dataset, [0])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, shuffle=False, collate_fn=collate_fn)

    return train_dataset, val_dataset, train_loader, val_loader



if __name__ == "__main__":

    TRAIN_IMAGES = "/home/shirshak/Thesis_Data/Steel_Scrap_detection/train/images"
    TRAIN_LABELS = "/home/shirshak/Thesis_Data/Steel_Scrap_detection/train/labels"
    VAL_IMAGES = "/home/shirshak/Thesis_Data/Steel_Scrap_detection/valid/images"
    VAL_LABELS = "/home/shirshak/Thesis_Data/Steel_Scrap_detection/valid/labels"

    train_dataset, val_dataset, _, _ = get_data(TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS, batch_size=1)
    
    print(train_dataset[0])
    print(val_dataset[0])