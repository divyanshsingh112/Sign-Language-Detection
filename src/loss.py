import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from utils.boxes import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcher(nn.Module):
    def __init__(self, weight_dict:dict):
        super().__init__()
        assert weight_dict.get('class_weighting') is not None and \
               weight_dict.get('bbox_weighting') is not None and \
               weight_dict.get('giou_weighting') is not None, \
               "Weight dict must contain weighting for all three losses: giou, class, and bbox."
        assert weight_dict.get('class_weighting') != 0 or \
               weight_dict.get('bbox_weighting') != 0 or \
               weight_dict.get('giou_weighting') != 0, \
               "All loss weights can't be 0."

        self.class_weighting = weight_dict.get('class_weighting') 
        self.bbox_weighting = weight_dict.get('bbox_weighting') 
        self.giou_weighting = weight_dict.get('giou_weighting') 

    @torch.no_grad()
    def forward(self, yhat, y):
        indices = []        
        for batch_idx, target in enumerate(y):
            batch_logits = yhat["pred_logits"][batch_idx]
            batch_boxes = yhat["pred_boxes"][batch_idx]
            batch_prob = batch_logits.softmax(-1)
            
            tgt_labels = target["labels"].to(torch.long)
            tgt_boxes = target["boxes"].to(batch_boxes.dtype)
            
            cost_class = -batch_prob[:, tgt_labels]
            cost_bbox = torch.cdist(batch_boxes, tgt_boxes, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(batch_boxes), 
                box_cxcywh_to_xyxy(tgt_boxes)
            )
            
            C_batch = (self.bbox_weighting * cost_bbox + 
                      self.class_weighting * cost_class + 
                      self.giou_weighting * cost_giou).cpu()
            
            ii, jj = linear_sum_assignment(C_batch)
            indices.append(
                (torch.as_tensor(ii, dtype=torch.int64), torch.as_tensor(jj, dtype=torch.int64))
            )
        return indices

class DETRLoss(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def classification_loss(self, yhat, y, indices):
        src_logits = yhat['pred_logits']
        idx = self.get_matched_query_indices(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(y, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                  dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_ce': loss_ce}

    def box_loss(self, yhat, y, indices, num_boxes):
        idx = self.get_matched_query_indices(indices)
        src_boxes = yhat['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(y, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def get_matched_query_indices(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, yhat, y):
        indices = self.matcher(yhat, y)
        device = next(iter(yhat.values())).device
        y = [{'labels': t['labels'].to(torch.long), 'boxes': t['boxes'].to(torch.float32)} for t in y]
        num_boxes = sum(len(t["labels"]) for t in y)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device).clamp(min=1)
        return {'labels': self.classification_loss(yhat, y, indices), 
                'boxes': self.box_loss(yhat, y, indices, num_boxes)}

if __name__ == "__main__":
    num_classes = 5
    weight_dict = {'class_weighting': 1, 'bbox_weighting': 5, 'giou_weighting': 2}
    matcher = HungarianMatcher(weight_dict)
    criterion = DETRLoss(num_classes=num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1)
    
    batch_size = 2
    num_queries = 10
    
    yhat = {
        'pred_logits': torch.randn(batch_size, num_queries, num_classes + 1),
        'pred_boxes': torch.rand(batch_size, num_queries, 4)
    }
    y = [
        {'labels': torch.tensor([1, 1, 2]), 'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.7, 0.1, 0.2], [0.8, 0.2, 0.15, 0.25]])},
        {'labels': torch.tensor([1]), 'boxes': torch.tensor([[0.4, 0.6, 0.3, 0.4]])}
    ]
    
    loss_dict = criterion(yhat, y)
    losses = (loss_dict['labels']['loss_ce'] * weight_dict['class_weighting'] + 
              loss_dict['boxes']['loss_bbox'] * weight_dict['bbox_weighting'] + 
              loss_dict['boxes']['loss_giou'] * weight_dict['giou_weighting'])
    
    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"{k}: {v}")
    print(f"\nTotal weighted loss: {losses.item():.4f}")