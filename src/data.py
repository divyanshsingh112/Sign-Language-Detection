import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset 
import os 
from PIL import Image 
import albumentations as A
from matplotlib import pyplot as plt 
from utils.boxes import rescale_bboxes, stacker
from utils.setup import get_classes

class DETRData(Dataset): 
    def __init__(self, path, train=True):
        super().__init__()
        self.path = path
        self.labels_path = os.path.join(self.path, 'labels')
        self.images_path = os.path.join(self.path, 'images')
        self.label_files = os.listdir(self.labels_path) 
        self.labels = list(filter(lambda x: x.endswith('.txt'), self.label_files))
        self.train = train
        
        mode = "Training" if train else "Testing"
        print(f"--- Initializing {mode} Dataset ---")
        print(f"Dataset Path: {self.path}")
        print(f"Total Samples: {len(self.labels)}")
        print("---------------------------------")

    def safe_transform(self, image, bboxes, labels):
        transform_list = [   
            A.Resize(500, 500),
            *([A.RandomCrop(width=224, height=224, p=0.33)] if self.train else []),
            A.Resize(224, 224),
            *([A.HorizontalFlip(p=0.5)] if self.train else []),
            *([A.ColorJitter(p=0.5)] if self.train else []),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.ToTensorV2()
        ]
        
        transform = A.Compose(
            transform_list, 
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )
        
        for _ in range(50):
            try:
                transformed = transform(image=image, bboxes=bboxes, class_labels=labels)
                if len(transformed['bboxes']) > 0 or not self.train or len(bboxes) == 0:
                    return transformed
            except Exception:
                continue
        
        base_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        return base_transform(image=image, bboxes=bboxes, class_labels=labels)

    def __len__(self): 
        return len(self.labels) 

    def __getitem__(self, idx): 
        label_path = os.path.join(self.labels_path, self.labels[idx]) 
        image_name = self.labels[idx].split('.')[0]
        image_path = os.path.join(self.images_path, f'{image_name}.jpg') 
        
        img = Image.open(image_path).convert("RGB")
      
        try:
            with open(label_path, 'r') as f: 
                annotations = f.readlines()
        except FileNotFoundError:
            annotations = []
            
        class_labels = []
        bounding_boxes = []
        if annotations:
            for annotation in annotations: 
                parts = annotation.strip().split(' ')
                if len(parts) == 5: # Ensure the line is valid
                    class_labels.append(parts[0]) 
                    bounding_boxes.append(parts[1:])
        
        # Handle cases with no bounding boxes (negative samples)
        if not bounding_boxes:
            bounding_boxes = np.zeros((0, 4))
            class_labels = np.zeros((0, 1))

        class_labels = np.array(class_labels, dtype=int) 
        bounding_boxes = np.array(bounding_boxes, dtype=float) 

        augmented = self.safe_transform(image=np.array(img), bboxes=bounding_boxes, labels=class_labels)
        
        labels = torch.tensor(augmented['class_labels'], dtype=torch.long)  
        boxes = torch.tensor(augmented['bboxes'], dtype=torch.float32)
        
        return augmented['image'], {'labels': labels, 'boxes': boxes}

if __name__ == '__main__':
    # This block is for testing the data loader
    dataset = DETRData('data/train', train=True) 
    dataloader = DataLoader(dataset, collate_fn=stacker, batch_size=4, drop_last=True)

    X, y = next(iter(dataloader))
    print("Sample batch targets:", y) 
    
    CLASSES = get_classes() 
    fig, ax = plt.subplots(2, 2) 
    axs = ax.flatten()
    for img, annotations, ax_item in zip(X, y, axs): 
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img_display = img.permute(1, 2, 0) * std + mean
        img_display = torch.clamp(img_display, 0, 1)
        ax_item.imshow(img_display)
        
        box_classes = annotations['labels'] 
        boxes = rescale_bboxes(annotations['boxes'], (224, 224))
        for box_class, bbox in zip(box_classes, boxes): 
            xmin, ymin, xmax, ymax = bbox.detach().numpy()
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
            ax_item.add_patch(rect)
            text = f'{CLASSES[box_class]}'
            ax_item.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
        ax_item.axis('off')

    fig.tight_layout() 
    plt.savefig('data_loader_test.png')
    print("\nData loader test plot saved to data_loader_test.png")
