import torch
import time
import sys
import os
import torchvision.models.detection as det_models

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.detector import SOTADetector

def profile_model(model, model_name, device, batch_size=1, img_size=640, num_runs=50):
    """ Profiles a given PyTorch model for latency, FPS, and Parameters. """
    model.eval()
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)
    
    # baseline의 내부 리사이즈를 무력화해 동일 해상도로 비교 (공정성 확보)
    if hasattr(model, 'transform'):
        model.transform.min_size = (img_size,)
        model.transform.max_size = img_size
        if getattr(model.transform, 'fixed_size', None) is not None:
            model.transform.fixed_size = (img_size, img_size)
        
    inp = [dummy_input[0]] if "Baseline" in model_name else dummy_input
    
    # 1. Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(inp)
            
    # 2. Measure
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(inp)
    end_time = time.time()
    
    avg_latency = ((end_time - start_time) / num_runs) * 1000 # in ms
    fps = 1000 / avg_latency
    
    params = sum(p.numel() for p in model.parameters())
    macs = 0.0
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(model, display=False) as fcm:
            _ = model(inp)
            macs = fcm.get_total_flops() / 2 / 1e9  # GMACs
    except Exception:
        pass
        
    return avg_latency, fps, params, macs

def run_comparison_benchmark():
    device = torch.device('cpu')
    print(f"🚀 Starting Benchmark on {device.type.upper()} (Batch Size: 1, Image: 640x640)")
    print(f"{'Model Name':<40} | {'Params (M)':<10} | {'GMACs':<7} | {'Latency (ms)':<15} | {'FPS':<10}")
    print("-" * 95)
    
    benchmark_suite = [
        ("Baseline: SSDLite-MobileNetV3 (Edge)", det_models.ssdlite320_mobilenet_v3_large(weights=None).to(device)),
        ("Ours: SOTADetector (Nano)", SOTADetector(scale='nano').to(device)),
        
        ("Baseline: FCOS-ResNet50-FPN", det_models.fcos_resnet50_fpn(weights=None).to(device)),
        ("Ours: SOTADetector (Medium)", SOTADetector(scale='medium').to(device)),
        
        ("Baseline: FasterRCNN-ResNet50-FPN-V2", det_models.fasterrcnn_resnet50_fpn_v2(weights=None).to(device)),
        ("Ours: SOTADetector (XLarge)", SOTADetector(scale='xlarge').to(device))
    ]
    
    for name, model in benchmark_suite:
        try:
            latency, fps, params, macs = profile_model(model, name, device)
            print(f"{name:<40} | {params / 1e6:>8.2f} M | {macs:>7.2f} | {latency:>10.2f} ms | {fps:>7.2f}")
        except Exception as e:
            print(f"{name:<40} | {'ERROR':>10} | {'ERROR':>10} | {'ERROR':>7}")
            print(e)
            
if __name__ == "__main__":
    run_comparison_benchmark()
