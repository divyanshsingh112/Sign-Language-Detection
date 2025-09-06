import cv2
import torch
from model import DETR
import albumentations as A
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors
import time 

print("--- Initializing Real-Time Sign Language Detection ---")

transforms = A.Compose(
        [   
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.ToTensorV2()
        ]
    )

model = DETR(num_classes=3)
model.eval()
model.load_pretrained('checkpoints/40_model.pt')
CLASSES = get_classes() 
COLORS = get_colors() 

print("Starting camera capture...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Failed to open camera.")
    exit()

frame_count = 0
fps_start_time = time.time()

while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame from camera.")
        break

    height, width, _ = frame.shape
        
    inference_start = time.time()
    transformed = transforms(image=frame)
    result = model(torch.unsqueeze(transformed['image'], dim=0))
    inference_time = (time.time() - inference_start) * 1000  # ms

    probabilities = result['pred_logits'].softmax(-1)[:,:,:-1] 
    max_probs, max_classes = probabilities.max(-1)
    keep_mask = max_probs > 0.8

    batch_indices, query_indices = torch.where(keep_mask) 

    bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices,:], (width, height))
    
    classes = max_classes[batch_indices, query_indices]
    probas = max_probs[batch_indices, query_indices]

    for bclass, bprob, bbox in zip(classes, probas, bboxes): 
        bclass_idx = bclass.detach().numpy()
        bprob_val = bprob.detach().numpy() 
        x1, y1, x2, y2 = bbox.detach().numpy()
        
        frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[bclass_idx], 2)
        
        #Define font properties (smaller size)
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.8
        font_thickness = 1
        frame_text = f"{CLASSES[bclass_idx]} - {round(float(bprob_val), 2)}"

        #Calculate the size of the text to make the background adaptive
        (text_width, text_height), baseline = cv2.getTextSize(frame_text, font_face, font_scale, font_thickness)

        #Draw a filled rectangle as the text background
        #Position it slightly above the main bounding box
        cv2.rectangle(
            frame,
            (int(x1), int(y1) - text_height - baseline),
            (int(x1) + text_width, int(y1)),
            COLORS[bclass_idx],
            -1 
            #(-1) thickness for a filled rectangle
        )

        #Put the text on top of the background rectangle
        cv2.putText(
            frame,
            frame_text,
            (int(x1), int(y1) - baseline), #Position text correctly
            font_face,
            font_scale,
            (255, 255, 255), #White text for better contrast
            font_thickness,
            cv2.LINE_AA
        )

        
    frame_count += 1
    elapsed_time = time.time() - fps_start_time
    if elapsed_time > 1.0: #Update FPS every second
        fps = frame_count / elapsed_time
        print(f"\rFPS: {fps:.2f} | Inference Time: {inference_time:.2f} ms", end="")
        frame_count = 0
        fps_start_time = time.time()

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        print("\nStopping real-time detection...")
        break

cap.release() 
cv2.destroyAllWindows()