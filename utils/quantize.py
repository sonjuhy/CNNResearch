import torch
import torch.ao.quantization as quant
from torch.ao.quantization import fuse_modules
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.detector import SOTADetector
from models.neck import ConvModule
from data.coco import create_dataloader
from data.transforms import TrainTransforms

def calibrate_model(model, data_loader, num_batches=10):
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            images = torch.stack(images)
            model(images)
            print(f"Calibrating batch {i+1}/{num_batches}...")

def export_int8_ptq(scale='nano', calib_batches=2):
    print(f"🚀 NPU 배포를 위한 Int8 PTQ 파이프라인 시작...")
    
    train_transforms = TrainTransforms(img_size=(640, 640))
    calib_loader = create_dataloader(
        root_dir="./data/images", 
        ann_file="./data/annotations.json", 
        batch_size=4, 
        is_train=True, 
        num_workers=0,
        transforms=train_transforms
    )
    if getattr(calib_loader.dataset, 'dummy_mode', False):
        raise RuntimeError("PTQ 캘리브레이션에는 실제 이미지가 필요합니다. 랜덤 노이즈로 뽑은 스케일은 배포에 사용할 수 없습니다.")

    model_fp32 = SOTADetector(scale=scale, quantizable=True)
    
    ckpt = f"checkpoints/best_{scale}.pt"
    if not os.path.exists(ckpt):
        raise RuntimeError(
            f"PTQ 는 학습된 가중치가 필요합니다: {ckpt} 없음. "
            "무작위 초기화 모델은 LayerScale(1e-6) 때문에 양자화 스케일이 붕괴합니다."
        )
    model_fp32.load_state_dict(torch.load(ckpt, map_location='cpu'))
    
    model_fp32.eval()
    
    from models.backbone import ConvNeXtBlockNPU
    for module in model_fp32.modules():
        if isinstance(module, ConvModule):
            fuse_modules(module, [['conv', 'bn']], inplace=True)
        elif isinstance(module, ConvNeXtBlockNPU):
            module.fold_gamma()
            fuse_modules(module, [['dwconv', 'norm']], inplace=True)
            
    # 백본의 stem과 downsamples Conv+BN Fusion 추가
    fuse_modules(model_fp32.backbone.stem, [['0', '1']], inplace=True)
    for ds in model_fp32.backbone.downsamples:
        fuse_modules(ds, [['0', '1']], inplace=True)
            
    backend = "qnnpack" 
    torch.backends.quantized.engine = backend
    model_fp32.qconfig = quant.get_default_qconfig(backend)
    
    model_prepared = quant.prepare(model_fp32)
    
    calibrate_model(model_prepared, calib_loader, num_batches=calib_batches)
    
    model_int8 = quant.convert(model_prepared)
    
    try:
        with torch.no_grad():
            model_int8(torch.randn(1, 3, 640, 640))
    except Exception as e:
        raise RuntimeError(f"Int8 모델이 추론 불가 상태입니다. 저장을 중단합니다: {e}") from e
        
    os.makedirs("exports", exist_ok=True)
    save_path = f"exports/sota_detector_{scale}_int8.pt"
    torch.save(model_int8.state_dict(), save_path)
    
    print(f"\n✅ Int8 PTQ 양자화 변환 완료! (저장 위치: {save_path})")

if __name__ == "__main__":
    export_int8_ptq(scale='nano')
