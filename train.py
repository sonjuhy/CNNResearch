import torch
import argparse
import time
import os
from data.coco import create_dataloader
from data.transforms import TrainTransforms
from models.detector import SOTADetector
from utils.loss import DetectionLoss
from utils.logger import setup_logger

# 구글 코랩 TPU (PyTorch XLA) 지원 모듈 로드
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

# TensorBoard 로거 지원 모듈 로드
try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, logger, writer=None, is_tpu=False):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    # TPU 분산 환경일 경우 Dataloader 매핑
    if is_tpu:
        dataloader = pl.ParallelLoader(dataloader, [device]).per_device_loader(device)
        
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = torch.stack(images).to(device)
        for i in range(len(targets)):
            targets[i]['boxes'] = targets[i]['boxes'].to(device)
            targets[i]['labels'] = targets[i]['labels'].to(device)
            
        optimizer.zero_grad()
        predictions = model(images)
        loss_dict = criterion(predictions, targets)
        loss = loss_dict['loss_total']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        # TPU 분산 환경은 별도의 optimizer step 호출 사용
        if is_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            msg = (f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                   f"Loss: {loss.item():.4f}")
            
            if is_tpu:
                xm.master_print(msg)
            else:
                logger.info(msg)
                
            # TensorBoard에 학습 곡선 기록
            if writer:
                step = (epoch - 1) * len(dataloader) + batch_idx
                writer.add_scalar('Loss/total', loss.item(), step)
                writer.add_scalar('Loss/cls', loss_dict['loss_cls'].item(), step)
                writer.add_scalar('Loss/box', loss_dict['loss_box'].item(), step)
                writer.add_scalar('Loss/obj', loss_dict['loss_obj'].item(), step)
                
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(dataloader)
    
    msg = f"✅ Epoch {epoch} 완료 (소요 시간: {epoch_time:.2f}초) | 평균 Loss: {avg_loss:.4f}\n"
    if is_tpu:
        xm.master_print(msg)
    else:
        logger.info(msg)
        
    return avg_loss

def _main_tpu(index, args):
    device = xm.xla_device()
    run_training(args, device, is_tpu=True)

def run_training(args, device, is_tpu=False):
    # DDP/TPU 환경에서는 Master 노드에서만 로깅 수행
    logger = setup_logger(log_dir="logs") if not is_tpu or xm.is_master_ordinal() else None
    writer = SummaryWriter(log_dir="runs/sota_detector") if TB_AVAILABLE and (not is_tpu or xm.is_master_ordinal()) else None
    
    train_transforms = TrainTransforms(img_size=(640, 640))
    train_loader = create_dataloader(
        root_dir=args.train_root, 
        ann_file=args.train_ann, 
        batch_size=args.batch_size, 
        is_train=True, 
        num_workers=args.workers,
        transforms=train_transforms
    )
    
    model = SOTADetector(scale=args.scale).to(device)
    criterion = DetectionLoss(num_classes=80).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        msg = f"✅ 체크포인트 로드 완료: {args.resume} (Epoch {start_epoch}부터 재개)"
        if is_tpu and xm: xm.master_print(msg)
        elif logger: logger.info(msg)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, writer, is_tpu)
        scheduler.step()
        
        # 모델 가중치(Checkpoint) 저장 
        if not is_tpu or xm.is_master_ordinal():
            save_path = os.path.join(args.checkpoint_dir, f"last_{args.scale}.pt")
            save_payload = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss
            }
            if is_tpu: xm.save(save_payload, save_path)
            else: torch.save(save_payload, save_path)
            
            # 베스트 모델 갱신 저장 (평가/배포 편의를 위해 순수 state_dict도 함께 호환)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(args.checkpoint_dir, f"best_{args.scale}.pt")
                if is_tpu: xm.save(model.state_dict(), best_path)
                else: torch.save(model.state_dict(), best_path)

    if writer: writer.close()

def main(args):
    if args.tpu and TPU_AVAILABLE:
        print("🚀 Starting Google Colab TPU Training via PyTorch XLA...")
        try:
            supported_devices = xm.get_xla_supported_devices()
            num_devices = len(supported_devices)
        except Exception:
            num_devices = 1
            
        if num_devices > 1:
            print(f"✅ 멀티코어 TPU 감지됨 ({num_devices} 코어): xmp.spawn 병렬 실행")
            xmp.spawn(_main_tpu, args=(args,), nprocs=num_devices, start_method='fork')
        else:
            print(f"✅ 단일코어 TPU 감지됨 (v5e-1 등): 단일 프로세스로 즉시 실행")
            device = xm.xla_device()
            run_training(args, device, is_tpu=True)
    else:
        if args.tpu and not TPU_AVAILABLE:
            print("⚠️ Colab TPU를 요청했으나 torch_xla 패키지가 없습니다. 일반 환경으로 fallback 합니다.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        run_training(args, device, is_tpu=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOTADetector Training Script")
    parser.add_argument('--config', type=str, default=None, help='YAML 설정 파일 경로 (e.g. configs/nano.yaml)')
    parser.add_argument('--scale', type=str, default='nano', choices=['nano', 'medium', 'xlarge'])
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None, help='재개할 체크포인트 경로')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='체크포인트 저장 디렉터리')
    parser.add_argument('--train-root', type=str, default='./data/images', help='학습 이미지 디렉터리')
    parser.add_argument('--train-ann', type=str, default='./data/annotations.json', help='학습 어노테이션 파일 경로')
    parser.add_argument('--tpu', action='store_true', help='Google Colab TPU 활성화 (DDP)')
    args = parser.parse_args()
    
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            k_opt = k.replace('-', '_')
            if hasattr(args, k_opt) and getattr(args, k_opt) == parser.get_default(k_opt):
                setattr(args, k_opt, v)
        print(f"📄 Config 로드 완료: {args.config}")
        
    main(args)
