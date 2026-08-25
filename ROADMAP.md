# 🚀 CNN Object Detection Model Development Roadmap

본 로드맵은 COCO 데이터셋 기반의 2026년 SOTA급 CNN Object Detection 모델(nano, medium, xlarge) 개발 및 최적화 과정을 4단계로 나누어 정의합니다.

## Phase 1: 기반 구축 및 데이터 파이프라인 설계 (Data & Foundation)
- [ ] **프로젝트 뼈대 구성**: `train.py`, `eval.py`, `models/`, `utils/` 등 모듈화된 디렉토리 구조 셋업
- [ ] **COCO Dataset 파이프라인**: 
  - COCO 2017 데이터셋 다운로드 스크립트 작성
  - PyTorch `DataLoader` 및 병렬 데이터 로딩(Pre-fetching) 최적화
- [ ] **고급 데이터 증강(Augmentation)**: 
  - 최신 Object Detection에 필수적인 증강 기법 적용 (Mosaic, MixUp, RandomAffine, ColorJitter 등)
- [ ] **실험 추적 도구 연동**: TensorBoard 또는 WandB 셋업을 통한 Loss 및 mAP(mean Average Precision) 트래킹

## Phase 2: 아키텍처 설계 (Architecture Design)
- [ ] **백본(Backbone) 네트워크 설계**: 
  - 하드웨어(CPU/NPU) 친화적인 구조 적용 (예: RepVGG 스타일의 재매개변수화, Depthwise Conv 최적화)
  - `nano`, `medium`, `xlarge` 티어별 채널 수 및 레이어 깊이(Depth) 스케일링 룰 정의
- [ ] **넥(Neck) 설계**: 
  - 멀티스케일 특징 융합을 위한 FPN(Feature Pyramid Network) 또는 BiFPN 아키텍처 구현
- [ ] **헤드(Head) 설계**: 
  - 앵커 프리(Anchor-free) 기반의 Decoupled Head (Classification과 Bounding Box Regression의 분리) 적용
  - 최신 Loss 함수(GIoU/DIoU/CIoU, Focal Loss 등) 적용

## Phase 3: 학습 및 모델 최적화 (Training & Optimization)
- [ ] **학습 루프 구현 및 검증**: 
  - 오버피팅 테스트(작은 데이터셋으로 Loss가 0으로 수렴하는지 확인)
  - 코사인 어닐링(Cosine Annealing) 등 최신 학습률 스케줄러(LR Scheduler) 적용
- [ ] **스케일업 및 전체 훈련**: 
  - `nano` 모델부터 학습을 시작하여 파이프라인 검증 후 `medium`, `xlarge` 모델 학습
- [ ] **추론 성능 프로파일링 (CPU/NPU)**: 
  - 모델의 파라미터 수, FLOPs/MACs, 추론 레이턴시(Latency) 측정 및 병목 구간 개선

## Phase 4: 양자화 및 배포 준비 (Quantization & Deployment)
- [ ] **Int8 양자화 평가**: 
  - PTQ(Post-Training Quantization)를 적용하여 기본 Int8 성능 및 정확도(mAP) 하락폭 측정
- [ ] **QAT (Quantization-Aware Training)**: 
  - 양자화 시 정확도 저하가 큰 모델(`nano` 등)을 대상으로 QAT 파이프라인 적용
- [ ] **모델 내보내기 (Export)**: 
  - 완성된 PyTorch 모델을 ONNX 또는 TorchScript 형태로 변환하여 C++ 환경이나 NPU 컴파일러에서 사용할 수 있도록 준비
