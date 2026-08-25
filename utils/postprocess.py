import torch
import torchvision

def decode_boxes(points, reg_preds):
    x1 = points[:, 0] - reg_preds[:, 0]
    y1 = points[:, 1] - reg_preds[:, 1]
    x2 = points[:, 0] + reg_preds[:, 2]
    y2 = points[:, 1] + reg_preds[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)

def postprocess(predictions, conf_threshold=0.05, iou_threshold=0.5, batch_idx=0, img_size=640):
    cls_pred = predictions['cls'][batch_idx]
    reg_pred = predictions['reg'][batch_idx]
    obj_pred = predictions['obj'][batch_idx]
    points   = predictions['points']

    scores = torch.sigmoid(cls_pred) * torch.sigmoid(obj_pred)
    max_scores, labels = scores.max(dim=1)

    keep = max_scores > conf_threshold
    if not keep.any():
        return torch.zeros((0, 4), device=cls_pred.device), torch.zeros((0,), device=cls_pred.device), torch.zeros((0,), dtype=torch.long, device=cls_pred.device)

    # 박스 좌표가 이미지 밖으로 벗어나지 않도록 clamp
    boxes = decode_boxes(points[keep], reg_pred[keep]).clamp(0, img_size)

    # NMS 전 연산량을 줄이기 위해 점수 상위 1000개만 필터링
    if len(boxes) > 1000:
        scores_kept = max_scores[keep]
        topk_idx = scores_kept.topk(1000)[1]
        boxes = boxes[topk_idx]
        scores_kept = scores_kept[topk_idx]
        labels_kept = labels[keep][topk_idx]
    else:
        scores_kept = max_scores[keep]
        labels_kept = labels[keep]

    idx = torchvision.ops.batched_nms(boxes, scores_kept, labels_kept, iou_threshold)
    return boxes[idx], scores_kept[idx], labels_kept[idx]
