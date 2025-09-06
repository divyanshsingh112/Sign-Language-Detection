from data import DETRData
from model import DETR
import torch
from torch.utils.data import DataLoader 
from matplotlib import pyplot as plt 
from utils.boxes import rescale_bboxes
from utils.setup import get_classes
import time

print("--- Running Test Script ---")

num_classes = 3
test_dataset = DETRData('data/test', train=False) 
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=4, drop_last=True) 
model = DETR(num_classes=num_classes)
model.eval()
model.load_pretrained('checkpoints/40_model.pt')

X, y = next(iter(test_dataloader))

print("Running inference on a test batch...")
start_time = time.time()
result = model(X) 
inference_time = (time.time() - start_time) * 1000
print(f"Inference Time for batch: {inference_time:.2f} ms")

probabilities = result['pred_logits'].softmax(-1)[:,:,:-1] 
max_probs, max_classes = probabilities.max(-1)
keep_mask = max_probs > 0.95
batch_indices, query_indices = torch.where(keep_mask) 

bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices,:], (224, 224))
classes = max_classes[batch_indices, query_indices]
probas = max_probs[batch_indices, query_indices]

print("\nDetections found:")
for i in range(len(classes)):
    print(f"- Class: {get_classes()[classes[i].item()]}, Confidence: {probas[i].item():.4f}")

CLASSES = get_classes()

fig, ax = plt.subplots(2, 2) 
axs = ax.flatten()
for idx, (img, ax_item) in enumerate(zip(X, axs)): 
    # Denormalize image for correct display
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img_display = img.permute(1, 2, 0) * std + mean
    img_display = torch.clamp(img_display, 0, 1)

    ax_item.imshow(img_display)
    ax_item.set_title(f"Image {idx+1}")
    ax_item.axis('off')
    for batch_idx, box_class, box_prob, bbox in zip(batch_indices, classes, probas, bboxes): 
        if batch_idx == idx: 
            xmin, ymin, xmax, ymax = bbox.detach().numpy()
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
            ax_item.add_patch(rect)
            text = f'{CLASSES[box_class]}: {box_prob:.2f}'
            ax_item.text(xmin, ymin, text, fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

fig.tight_layout() 
plt.savefig('test_result.png')
print("Plot saved to data_visualization.png")