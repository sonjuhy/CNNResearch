import torch
import argparse
import os
from data.coco import create_dataloader
from models.detector import SOTADetector
from utils.postprocess import postprocess
from data.transforms import ValTransforms

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

def run_evaluation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Evaluation Started on {device}")
    
    if not METRICS_AVAILABLE:
        print("❌ torchmetrics 패키지가 없습니다. mAP를 계산하려면 다음을 설치하세요: pip install torchmetrics")
        return
        
    model = SOTADetector(scale=args.scale).to(device)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"✅ 가중치 로드 완료: {args.weights}")
    else:
        print("⚠️ 가중치 파일이 없습니다. 초기화 상태로 평가를 진행합니다.")
        
    model.eval()
    
    # 평가 시에는 데이터 증강(Flip, ColorJitter)을 끕니다.
    from data.transforms import ValTransforms
    val_transforms = ValTransforms(img_size=(640, 640))
    val_loader = create_dataloader(
        root_dir="./data/val_images", 
        ann_file="./data/val_annotations.json", 
        batch_size=args.batch_size, 
        is_train=False, 
        transforms=val_transforms
    )
    
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = torch.stack(images).to(device)
            predictions = model(images)
            
            preds = []
            gts = []
            for i in range(len(images)):
                boxes, scores, labels = postprocess(predictions, batch_idx=i)
                preds.append(dict(boxes=boxes, scores=scores, labels=labels))
                
                gt_boxes = targets[i]['boxes'].to(device)
                gt_labels = targets[i]['labels'].to(device)
                gts.append(dict(boxes=gt_boxes, labels=gt_labels))
                
            metric.update(preds, gts)
            print(f"Batch {batch_idx+1}/{len(val_loader)} evaluated.")
            
    result = metric.compute()
    print("\n📊 평가 결과 (mAP):")
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                print(f"{k}: {v.item():.4f}")
            else:
                # 텐서 요소가 여러 개(ex. 클래스별 mAP)일 경우 예외 처리
                print(f"{k}: {['{:.4f}'.format(val) for val in v.tolist()]}")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=str, default='nano')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--weights', type=str, default='checkpoints/best_nano.pt')
    args = parser.parse_args()
    run_evaluation(args)
