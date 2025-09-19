from src.data import DETRData
from src.model import DETR
from loss import DETRLoss, HungarianMatcher
from torch.utils.data import DataLoader 
from torch import optim, save
import torch
from src.utils.boxes import stacker
import sys
import os
import json

# Define the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {device} ---")

if __name__ == '__main__': 
    print("--- Starting Training Script for Keyboard Project ---")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    
    # Ensure your data paths point to the new alphabet dataset
    train_dataset = DETRData('data/train', train=True) 
    train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=stacker, drop_last=True) 

    test_dataset = DETRData('data/test', train=False) 
    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=stacker, drop_last=True) 

    # --- KEY CHANGE: Updated for the keyboard project ---
    num_classes = 29 # 26 letters + Backspace, Tab, Enter
    # ----------------------------------------------------
    model = DETR(num_classes=num_classes)
    
    # Optional: Load a pretrained backbone if you have one, or start fresh
    # model.load_pretrained('pretrained/final_model.pt')
    
    model.to(device)
    model.log_model_info()

    # Learning rate can be adjusted. 1e-5 is a safer, lower value to start with.
    opt = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, len(train_dataloader)*30, T_mult=2)

    weights = {'class_weighting': 1, 'bbox_weighting': 5, 'giou_weighting': 2}
    matcher = HungarianMatcher(weights)
    criterion = DETRLoss(num_classes=num_classes, matcher=matcher, weight_dict=weights, eos_coef=0.1)
    criterion.to(device)

    # Consider reducing epochs and monitoring the loss curve to find the best value
    epochs = 4000
    
    # --- NEW: Add lists to store loss history for plotting ---
    train_loss_history = []
    test_loss_history = []
    best_test_loss = float('inf') # Keep track of the best model
    # --------------------------------------------------------

    print("\n--- Training Configuration ---")
    print(f"Total Epochs: {epochs}")
    print(f"Number of Classes: {num_classes}")
    print(f"Batch Size: 4")
    print(f"Initial Learning Rate: {opt.defaults['lr']:.1e}")
    print("----------------------------\n")

    for epoch in range(epochs): 
        model.train()
        train_epoch_loss = 0.0 
        
        print(f"--- Epoch {epoch+1}/{epochs} ---")
        for batch_idx, batch in enumerate(train_dataloader): 
            X, y = batch
            X_list = [img.to(device) for img in X]
            y = [{k: v.to(device) for k, v in t.items()} for t in y]
            X_stacked = torch.stack(X_list, dim=0)
            
            try: 
                yhat = model(X_stacked)
                loss_dict = criterion(yhat, y) 
                weight_dict = criterion.weight_dict
                
                losses = (loss_dict['labels']['loss_ce'] * weight_dict['class_weighting'] + 
                          loss_dict['boxes']['loss_bbox'] * weight_dict['bbox_weighting'] + 
                          loss_dict['boxes']['loss_giou'] * weight_dict['giou_weighting'])
                
                train_epoch_loss += losses.item() 
                
                opt.zero_grad()
                losses.backward()
                opt.step()
                
                print(f"\rTraining Batch {batch_idx+1}/{len(train_dataloader)} | Loss: {losses.item():.4f}", end="")
                
            except Exception as e: 
                print(f"\nERROR: Training error at epoch {epoch+1}, batch {batch_idx+1}: {str(e)}")
                sys.exit()
        
        avg_train_loss = train_epoch_loss / len(train_dataloader)
        train_loss_history.append(avg_train_loss)
        print(f"\nAverage Training Loss: {avg_train_loss:.4f}")
        
        scheduler.step()
    
        # --- Validation Phase ---
        model.eval()
        test_epoch_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                X, y = batch
                X_list = [img.to(device) for img in X]
                y = [{k: v.to(device) for k, v in t.items()} for t in y]
                X_stacked = torch.stack(X_list, dim=0)
                
                yhat = model(X_stacked)
                loss_dict = criterion(yhat, y) 
                weight_dict = criterion.weight_dict
                losses = (loss_dict['labels']['loss_ce'] * weight_dict['class_weighting'] + 
                          loss_dict['boxes']['loss_bbox'] * weight_dict['bbox_weighting'] + 
                          loss_dict['boxes']['loss_giou'] * weight_dict['giou_weighting'])
                test_epoch_loss += losses.item() 
        
        avg_test_loss = test_epoch_loss / len(test_dataloader)
        test_loss_history.append(avg_test_loss)
        print(f"Average Test Loss: {avg_test_loss:.4f}\n")

        # --- NEW: Early stopping and saving the best model ---
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            save(model.state_dict(), "checkpoints/keyboard_model.pt")
            print(f"🎉 New best model saved with test loss: {avg_test_loss:.4f}")
        # ----------------------------------------------------

    # After the loop, save the loss history to a file to plot later
    with open('loss_history.json', 'w') as f:
        json.dump({'train_loss': train_loss_history, 'test_loss': test_loss_history}, f)
    print("Loss history saved to loss_history.json")

    final_path = f"checkpoints/final_keyboard_model_{epochs}.pt"
    save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")