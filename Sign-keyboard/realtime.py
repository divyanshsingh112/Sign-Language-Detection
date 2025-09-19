import cv2
import torch
from src.model import DETR
import albumentations as A
from torchvision.ops import nms
from src.utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors
import time 

print("--- Initializing Real-Time Sign Language Keyboard ---")

# --- State Management Variables for Keyboard ---
current_sentence = ""
cooldown_period = 1.5 # Seconds to wait before detecting the next character
last_detection_time = time.time()
# ---------------------------------------------

transforms = A.Compose(
    [   
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ToTensorV2()
    ]
)

# --- Model Loading ---
# ⚠️ IMPORTANT: Update these values to match your retrained model
NUM_CLASSES = 29 # e.g., 26 letters + Backspace, Tab, Enter
CHECKPOINT_PATH = 'checkpoints/your_new_trained_model.pt' 

model = DETR(num_classes=NUM_CLASSES)
model.eval()
model.load_pretrained(CHECKPOINT_PATH)
# -----------------------

CLASSES = get_classes() 
COLORS = get_colors() 

print("Starting camera capture...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Failed to open camera.")
    exit()

while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame from camera.")
        break

    height, width, _ = frame.shape
        
    # --- Model Inference ---
    transformed = transforms(image=frame)
    with torch.no_grad(): # Use torch.no_grad() for faster inference
        result = model(torch.unsqueeze(transformed['image'], dim=0))

    # --- Post-Processing and Filtering for Accuracy ---
    
    # 1. ADJUST THIS THRESHOLD: Increase to reduce false detections (e.g., 0.9 or 0.95)
    CONFIDENCE_THRESH = 0.9 
    
    # Get predictions for the first (and only) image in the batch
    probabilities = result['pred_logits'].softmax(-1)[0,:,:-1] 
    
    # Filter out predictions below the confidence threshold
    keep_mask = probabilities.max(-1).values > CONFIDENCE_THRESH
    
    conf_scores = probabilities.max(-1).values[keep_mask]
    pred_classes = probabilities.argmax(-1)[keep_mask]
    pred_boxes_raw = result['pred_boxes'][0, keep_mask, :]
    
    # --- Keyboard Logic ---
    # We find the single most confident detection to use as a "key press"
    if conf_scores.numel() > 0 and (time.time() - last_detection_time > cooldown_period):
        # Get the index of the highest probability detection
        best_idx = conf_scores.argmax()
        best_class_idx = pred_classes[best_idx].item()
        best_class_name = CLASSES[best_class_idx]

        # Apply keyboard actions
        if best_class_name == 'Enter':
            print(f"\nFinal Sentence: {current_sentence}")
            current_sentence = "" # Reset
        elif best_class_name == 'Backspace':
            current_sentence = current_sentence[:-1]
        elif best_class_name == 'Tab':
            current_sentence += '\t'
        else: # It's an alphabet sign
            current_sentence += best_class_name

        last_detection_time = time.time() # Reset cooldown timer
    
    # --- Bounding Box Drawing Logic ---
    if pred_boxes_raw.shape[0] > 0:
        # 2. APPLY NON-MAXIMUM SUPPRESSION (NMS): Cleans up overlapping boxes
        # Convert boxes from [center_x, center_y, w, h] to [x1, y1, x2, y2] for NMS
        pred_boxes_xyxy = rescale_bboxes(pred_boxes_raw, (width, height))
        
        # nms_indices holds the indices of the boxes to keep
        nms_indices = nms(boxes=pred_boxes_xyxy, scores=conf_scores, iou_threshold=0.5)

        # 3. LIMIT TO TOP 2 DETECTIONS:
        final_boxes = pred_boxes_xyxy[nms_indices]
        final_classes = pred_classes[nms_indices]
        final_scores = conf_scores[nms_indices]
        
        # Sort by score and keep only the top 2
        top_k_indices = torch.topk(final_scores, k=min(2, len(final_scores))).indices
        
        # Draw the final, filtered bounding boxes
        for i in top_k_indices:
            bclass_idx = final_classes[i].item()
            bprob_val = final_scores[i].item()
            x1, y1, x2, y2 = final_boxes[i].detach().numpy()
            
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[bclass_idx], 2)
            
            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.8
            font_thickness = 1
            frame_text = f"{CLASSES[bclass_idx]} - {round(float(bprob_val), 2)}"

            (text_width, text_height), baseline = cv2.getTextSize(frame_text, font_face, font_scale, font_thickness)

            cv2.rectangle(
                frame,
                (int(x1), int(y1) - text_height - baseline),
                (int(x1) + text_width, int(y1)),
                COLORS[bclass_idx],
                -1
            )
            cv2.putText(
                frame,
                frame_text,
                (int(x1), int(y1) - baseline),
                font_face,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )

    # --- Display current sentence on screen ---
    cv2.rectangle(frame, (0, height - 40), (width, height), (0, 0, 0), -1)
    cv2.putText(frame, current_sentence, (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # ----------------------------------------

    cv2.imshow('Sign Language Keyboard', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        print("\nStopping...")
        break

cap.release() 
cv2.destroyAllWindows()