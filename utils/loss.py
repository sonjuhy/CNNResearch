import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.postprocess import decode_boxes

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p = torch.sigmoid(inputs)
        pt = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'sum': return focal_loss.sum()
        elif self.reduction == 'mean': return focal_loss.mean()
        return focal_loss

class BoxGIoULoss(nn.Module):
    def __init__(self, reduction='sum'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, preds, targets):
        lt = torch.max(preds[:, :2], targets[:, :2])
        rb = torch.min(preds[:, 2:], targets[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        
        area_preds = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
        area_targets = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
        union = area_preds + area_targets - inter
        iou = inter / union.clamp(min=1e-6)
        
        lt_enclose = torch.min(preds[:, :2], targets[:, :2])
        rb_enclose = torch.max(preds[:, 2:], targets[:, 2:])
        wh_enclose = (rb_enclose - lt_enclose).clamp(min=0)
        area_enclose = wh_enclose[:, 0] * wh_enclose[:, 1]
        
        giou = iou - (area_enclose - union) / area_enclose.clamp(min=1e-6)
        loss = 1 - giou
        
        if self.reduction == 'sum': return loss.sum()
        return loss.mean()

class TargetAssigner(nn.Module):
    def __init__(self, center_radius=1.5):
        super().__init__()
        self.center_radius = center_radius
        
    @torch.no_grad()
    def forward(self, points, strides, gt_bboxes, gt_labels):
        N = points.shape[0]
        device = points.device
        if gt_bboxes.numel() == 0:
            return (torch.zeros(N, dtype=torch.bool, device=device),
                    torch.zeros(N, dtype=torch.long, device=device))

        cx, cy = points[:, 0:1], points[:, 1:2]

        l = cx - gt_bboxes[:, 0].unsqueeze(0)
        t = cy - gt_bboxes[:, 1].unsqueeze(0)
        r = gt_bboxes[:, 2].unsqueeze(0) - cx
        b = gt_bboxes[:, 3].unsqueeze(0) - cy
        in_box = torch.stack([l, t, r, b], dim=-1).min(dim=-1).values > 0

        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) * 0.5
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) * 0.5
        radius = (self.center_radius * strides).unsqueeze(1)
        in_center = ((cx - gt_cx.unsqueeze(0)).abs() < radius) & \
                    ((cy - gt_cy.unsqueeze(0)).abs() < radius)

        REG_RANGES = {8: (0, 32), 16: (32, 64), 32: (64, float('inf'))}
        max_reg = torch.stack([l, t, r, b], dim=-1).max(dim=-1).values
        lo = torch.where(strides == 8, 0.0,
             torch.where(strides == 16, 32.0, 64.0)).unsqueeze(1).to(device)
        hi = torch.where(strides == 8, 32.0,
             torch.where(strides == 16, 64.0, float('inf'))).unsqueeze(1).to(device)
        in_level = (max_reg >= lo) & (max_reg <= hi)

        candidate = in_box & in_center & in_level

        areas = ((gt_bboxes[:, 2] - gt_bboxes[:, 0]) *
                 (gt_bboxes[:, 3] - gt_bboxes[:, 1])).unsqueeze(0).expand(N, -1).clone()
        areas[~candidate] = float('inf')
        min_area, matched_gt = areas.min(dim=1)
        fg_mask = min_area < float('inf')
        return fg_mask, matched_gt

class DetectionLoss(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.assigner = TargetAssigner()
        self.cls_loss = FocalLoss()
        self.box_loss = BoxGIoULoss()
        self.obj_loss = nn.BCEWithLogitsLoss(reduction='sum')
        
    def forward(self, predictions, targets):
        cls_pred  = predictions['cls']
        reg_pred  = predictions['reg']
        obj_pred  = predictions['obj']
        points    = predictions['points']
        strides   = predictions['strides']

        B, N, _ = cls_pred.shape
        device = cls_pred.device
        total_cls = cls_pred.new_zeros(())
        total_box = cls_pred.new_zeros(())
        total_obj = cls_pred.new_zeros(())
        num_pos_total = 0

        for b in range(B):
            gt_boxes  = targets[b]['boxes'].to(device)
            gt_labels = targets[b]['labels'].to(device)

            fg_mask, matched_gt = self.assigner(points, strides, gt_boxes, gt_labels)
            num_pos = int(fg_mask.sum())
            num_pos_total += num_pos

            obj_target = fg_mask.to(cls_pred.dtype).unsqueeze(-1)
            total_obj = total_obj + self.obj_loss(obj_pred[b], obj_target)

            if num_pos == 0: continue

            cls_target = torch.zeros(num_pos, self.num_classes, device=device, dtype=cls_pred.dtype)
            cls_target[torch.arange(num_pos, device=device), gt_labels[matched_gt[fg_mask]]] = 1.0
            total_cls = total_cls + self.cls_loss(cls_pred[b][fg_mask], cls_target)

            pred_boxes = decode_boxes(points[fg_mask], reg_pred[b][fg_mask])
            total_box  = total_box + self.box_loss(pred_boxes, gt_boxes[matched_gt[fg_mask]])

        norm = max(num_pos_total, 1)
        loss_cls = total_cls / norm
        loss_box = total_box / norm
        loss_obj = total_obj / norm

        return {
            "loss_cls": loss_cls,
            "loss_box": loss_box * 5.0,
            "loss_obj": loss_obj,
            "loss_total": loss_cls + loss_box * 5.0 + loss_obj,
        }
