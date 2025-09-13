from data import DETRData
from model import DETR
from loss import DETRLoss, HungarianMatcher
from torch.utils.data import DataLoader 
from torch import optim, save
import torch
from utils.boxes import stacker
import sys
import os

#Define the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {device} ---")

if __name__ == '__main__': 
    print("--- Starting Training Script ---")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    
    train_dataset = DETRData('data/train', train=True) 
    train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=stacker, drop_last=True) 

    test_dataset = DETRData('data/test', train=False) 
    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=stacker, drop_last=True) 

    num_classes = 8 
    model = DETR(num_classes=num_classes)
    
    model.load_pretrained('pretrained/final_model.pt')
    
    #Move the model to the selected device (GPU)
    model.to(device)
    
    model.log_model_info()
    model.train() 

    opt = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, len(train_dataloader)*30, T_mult=2)

    weights = {'class_weighting': 1, 'bbox_weighting': 5, 'giou_weighting': 2}
    matcher = HungarianMatcher(weights)
    criterion = DETRLoss(num_classes=num_classes, matcher=matcher, weight_dict=weights, eos_coef=0.1)
    criterion.to(device)

    train_batches = len(train_dataloader)
    test_batches = len(test_dataloader)
    epochs = 4000
    
    print("\n--- Training Configuration ---")
    print(f"Total Epochs: {epochs}")
    print(f"Batch Size: 4")
    print(f"Train Batches per Epoch: {train_batches}")
    print(f"Test Batches per Epoch: {test_batches}")
    print(f"Initial Learning Rate: 1e-5")
    print("----------------------------\n")

    for epoch in range(epochs): 
        model.train()
        train_epoch_loss = 0.0 
        
        print(f"--- Epoch {epoch+1}/{epochs} ---")
        for batch_idx, batch in enumerate(train_dataloader): 
            X, y = batch
            
            #Move the data batch to the device
            X_list = [img.to(device) for img in X]
            y = [{k: v.to(device) for k, v in t.items()} for t in y]
            
            #Stack the list of tensors into a single batch tensor
            X_stacked = torch.stack(X_list, dim=0)
            
            try: 
                yhat = model(X_stacked) #Pass the stacked tensor to the model
                loss_dict = criterion(yhat, y) 
                weight_dict = criterion.weight_dict
                
                losses = (loss_dict['labels']['loss_ce'] * weight_dict['class_weighting'] + 
                          loss_dict['boxes']['loss_bbox'] * weight_dict['bbox_weighting'] + 
                          loss_dict['boxes']['loss_giou'] * weight_dict['giou_weighting'])
                
                train_epoch_loss += losses.item() 
                
                opt.zero_grad()
                losses.backward()
                opt.step()
                
                print(f"\rTraining Batch {batch_idx+1}/{train_batches} | Loss: {losses.item():.4f}", end="")
                
            except Exception as e: 
                print(f"\nERROR: Training error at epoch {epoch+1}, batch {batch_idx+1}: {str(e)}")
                print(f"Batch targets causing error: {str(y)}")
                sys.exit()
        
        print()
        avg_train_loss = train_epoch_loss / train_batches
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        
        scheduler.step()
    
        # --- Validation Phase ---
        model.eval()
        test_epoch_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                X, y = batch
                
                #Also move and stack validation data
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
        
        avg_test_loss = test_epoch_loss / test_batches
        print(f"Average Test Loss: {avg_test_loss:.4f}\n")

        if (epoch + 1) % 500 == 0: 
            checkpoint_path = f"checkpoints/{epoch+1}_model.pt"
            save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
    final_path = f"checkpoints/final_model.pt"
    save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

