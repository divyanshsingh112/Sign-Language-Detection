from src.data import DETRData
from src.model import DETR
import torch
from torch.utils.data import DataLoader 
from matplotlib import pyplot as plt 
from utils.boxes import rescale_bboxes, stacker # Added stacker for consistency
from src.utils.setup import get_classes
import time

print("--- Running Test Script for Keyboard Project ---")

# --- KEY CHANGE: Updated for the keyboard project ---
num_classes = 29 # 26 letters + Backspace, Tab, Enter
# ----------------------------------------------------

# Make sure your test data path is correct for the new dataset
test_dataset = DETRData('data/test', train=False) 
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=4, drop_last=True, collate_fn=stacker) 

model = DETR(num_classes=num_classes)
model.eval()

# --- KEY CHANGE: Load the best model saved from training ---
model.load_pretrained('pretrained/keyboard_model.pt')
# ----------------------------------------------------------

X_list, y = next(iter(test_dataloader))
X = torch.stack(X_list, dim=0)

print("Running inference on a test batch...")
start_time = time.time()
with torch.no_grad():
    result = model(X) 
inference_time = (time.time() - start_time) * 1000
print(f"Inference Time for batch: {inference_time:.2f} ms")

probabilities = result['pred_logits'].softmax(-1)[:,:,:-1] 
max_probs, max_classes = probabilities.max(-1)

# You can adjust this threshold to be more or less strict
keep_mask = max_probs > 0.95
batch_indices, query_indices = torch.where(keep_mask) 

bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices,:], (224, 224))
classes = max_classes[batch_indices, query_indices]
probas = max_probs[batch_indices, query_indices]

print("\nDetections found:")
CLASSES = get_classes() # Ensure get_classes() returns the new 29 classes
for i in range(len(classes)):
    print(f"- Class: {CLASSES[classes[i].item()]}, Confidence: {probas[i].item():.4f}")

fig, ax = plt.subplots(2, 2, figsize=(10, 10)) 
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
            ax_item.text(xmin, ymin, text, fontsize=9, bbox=dict(facecolor='yellow', alpha=0.5))

fig.tight_layout() 
plt.savefig('keyboard_test_result.png')
print("Plot saved to keyboard_test_result.png")