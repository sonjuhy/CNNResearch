import torch
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.detector import SOTADetector

def export_onnx(scale='nano', img_size=640, output_dir="exports", weights=None):
    print(f"🚀 ONNX Export 파이프라인 시작 (모델 스케일: {scale.upper()})")
    os.makedirs(output_dir, exist_ok=True)
    
    model = SOTADetector(scale=scale)
    ckpt = weights or f"checkpoints/best_{scale}.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        print(f"✅ 가중치 로드 완료: {ckpt}")
    else:
        raise RuntimeError(f"학습된 가중치가 필요합니다: {ckpt} 없음. 무작위 초기화 상태로는 올바른 ONNX 모델을 생성할 수 없습니다.")
    model.eval() 
    
    dummy_input = torch.randn(1, 3, img_size, img_size)
    output_path = os.path.join(output_dir, f"sota_detector_{scale}_{img_size}.onnx")
    
    try:
        torch.onnx.export(
            model, 
            dummy_input, 
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['cls', 'reg', 'obj', 'points', 'strides'],
            dynamic_axes={
                'images': {0: 'batch'},
                'cls': {0: 'batch'}, 
                'reg': {0: 'batch'}, 
                'obj': {0: 'batch'}
            },
            dynamo=False # legacy exporter to respect opset
        )
        print(f"✅ ONNX 변환 및 저장 완료! ({output_path})")
        
    except Exception as e:
        print(f"❌ ONNX 변환 실패: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SOTA Detector to ONNX")
    parser.add_argument('--scale', type=str, default='nano', choices=['nano', 'medium', 'xlarge'])
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--weights', type=str, default=None, help="가중치 파일 경로 (기본값: checkpoints/best_<scale>.pt)")
    args = parser.parse_args()
    export_onnx(scale=args.scale, img_size=args.img_size, weights=args.weights)
