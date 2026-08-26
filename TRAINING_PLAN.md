# 🎯 SOTADetector 학습 계획서

- **작성일**: 2026-08-26
- **대상**: `nano` / `medium` / `xlarge` 3티어, COCO 2017 Detection
- **근거**: [review/](./review/) 6차 리뷰까지의 실측 결과 + 본 문서를 위해 새로 측정한 티어별 연산량·스텝 비용
- **관련 문서**: [ROADMAP.md](./ROADMAP.md) Phase 3 · [MODEL_ARCHITECTURE.md](./MODEL_ARCHITECTURE.md)

> **핵심 결론 3가지**
> 1. **학습 시작 전 반드시 처리해야 할 블로커가 6개 있습니다** (§1). 지금 학습을 시작하면 연산 자원이 낭비됩니다.
> 2. **ImageNet 사전학습 가중치가 없어 from-scratch 학습이 강제되므로, 세 티어 모두 300 에포크가 기본입니다** (§3).
> 3. **전통적 Early Stopping(patience → 중단)은 권장하지 않습니다.** 코사인 스케줄의 마지막 저 LR 구간에서 mAP가 크게 오르기 때문입니다. 대신 **val mAP 기준 best 체크포인트 선정 + 발산 감지 중단**을 씁니다 (§5).

---

## 목차

1. [학습 전 필수 블로커](#1-학습-전-필수-블로커)
2. [연산량 실측 및 소요 시간 추정](#2-연산량-실측-및-소요-시간-추정)
3. [티어별 에포크 계획](#3-티어별-에포크-계획)
4. [하이퍼파라미터 레시피](#4-하이퍼파라미터-레시피)
5. [Early Stopping 정책](#5-early-stopping-정책)
6. [단계별 실행 계획](#6-단계별-실행-계획)
7. [모니터링 및 중단 기준](#7-모니터링-및-중단-기준)
8. [체크포인트 정책](#8-체크포인트-정책)
9. [양자화 학습 (QAT) 계획](#9-양자화-학습-qat-계획)
10. [리스크와 대응](#10-리스크와-대응)

---

## 1. 학습 전 필수 블로커

**아래 6개를 처리하기 전에 본격 학습을 시작하면 안 됩니다.** 각 항목은 학습 결과를 무효화하거나, 며칠~몇 주의 연산을 낭비시킵니다.

| # | 블로커 | 현재 상태 | 왜 치명적인가 |
| :---: | :--- | :--- | :--- |
| **B1** | **실제 COCO 데이터** | `data/val_images`·`annotations.json` 없음, `scripts/` 없음 | 지금은 랜덤 노이즈 더미 100장으로 학습·평가합니다. 모든 지표가 무의미합니다 |
| **B2** | **val 검증 루프 + val mAP 기준 best 선정** | `train.py:125`가 **학습 loss** 기준 | 오버피팅한 가중치가 `best_*.pt`로 저장됩니다. 300 에포크를 돌려도 잘못된 모델을 고릅니다 |
| **B3** | **강한 증강 (Mosaic / MixUp)** | 미구현 (`ColorJitter`·`HFlip`만) | **from-scratch 300 에포크는 강한 증강 없이는 반드시 과적합합니다.** 최대 정확도 손실 요인 |
| **B4** | **resume (학습 재개)** | 미구현 | medium 3일 / xlarge 29일 런이 중단되면 처음부터 다시 해야 합니다 |
| **B5** | **AMP (bf16 혼합정밀)** | 미구현 | 속도 ~2배, 메모리 ~2배 손해. medium/xlarge에서 사실상 필수 |
| **B6** | **`num_workers` 설정** | `train.py` 기본 `--workers 0` | GPU가 초당 700장을 요구하는데 단일 프로세스로 JPEG 디코딩·증강을 합니다. **데이터 로더가 병목이 되어 GPU가 놀게 됩니다** |

### 권장 추가 항목 (블로커는 아니나 강력 권장)

| 항목 | 이유 |
| :--- | :--- |
| **EMA (Exponential Moving Average)** | 현대 검출기 표준. 추가 학습 비용 없이 mAP +0.5~1.5 |
| **seed 고정** | 현재 `manual_seed` 0건. 실험 재현 불가 |
| **CUDA DDP** | `xlarge`는 단일 GPU로 29일 → 다중 GPU 없이는 비현실적 |
| **SimOTA 할당** | 현재 assigner는 GT당 positive가 4~12개로 매우 적어 수렴이 느립니다 |
| **configs 배선** | `configs/*.yaml`은 유효하나 읽는 코드가 없음. 300 에포크 실험을 CLI 인자로 관리하는 것은 위험 |

---

## 2. 연산량 실측 및 소요 시간 추정

### 2.1 실측값 (CPU 6 threads, 640×640)

| Scale | Params | **GMACs** | 순전파 | 학습 스텝 (fwd+bwd+step) |
| :--- | ---: | ---: | ---: | ---: |
| **nano** | 5.61 M | **11.76** | 200.9 ms/img | **731.4 ms/img** |
| **medium** | 31.49 M | **59.21** | 748.8 ms/img | **2,638.6 ms/img** |
| **xlarge** | 365.02 M | **594.05** | 5,529 ms/img | ~20,100 ms/img *(추정)* |

> `xlarge` 학습 스텝은 CPU에서 측정 불가(타임아웃)하여 nano/medium의 fwd→train 배율(×3.64)로 추정했습니다.

### 2.2 참고: 동급 모델과의 비교

| 모델 | GMACs @640 |
| :--- | ---: |
| SSDLite-MobileNetV3 | 2.75 |
| YOLOv8n | ~4.4 |
| **Ours nano** | **11.76** |
| YOLOv8x | ~129 |
| **Ours xlarge** | **594.05** |

> ⚠️ `nano`는 YOLOv8n의 **2.7배**, `xlarge`는 YOLOv8x의 **4.6배** 연산량입니다.
> 이는 학습 비용에 직결되며, `MODEL_ARCHITECTURE.md`의 티어 정의(nano = 라즈베리파이 실시간)와도 어긋납니다.
> **학습 착수 전에 티어를 재설계할지, 문서 정의를 조정할지 결정하십시오** (§10 R4).

### 2.3 소요 시간 추정

COCO train2017 = **118,287 장**.

#### CPU 학습은 불가능

| Scale | 1 에포크 | 300 에포크 |
| :--- | ---: | ---: |
| nano | 24.0 시간 | 300 일 |
| medium | 86.7 시간 | 3.6 년 |
| xlarge | 660 시간 | 27 년 |

**GPU가 반드시 필요합니다.**

#### GPU 추정 (1× A100 80GB, bf16)

가정: 학습 FLOPs ≈ 6 × GMACs (순전파 2×MACs, 역전파 2배) · 유효 연산 성능 **50 TFLOPS**

| Scale | 학습 FLOPs/img | 1 에포크 | **300 에포크 (1× A100)** | **300 에포크 (8× A100)** |
| :--- | ---: | ---: | ---: | ---: |
| nano | 70.6 GFLOPs | ~2.8 분 | **~14 시간** | ~1.7 시간 |
| medium | 355 GFLOPs | ~14 분 | **~2.9 일** | ~8.8 시간 |
| xlarge | 3,564 GFLOPs | ~2.3 시간 | **~29 일** | ~3.7 일 |

> ⚠️ **이 표는 추정치입니다.** 이 아키텍처는 depthwise conv 비중이 높아 실제 MFU가 가정치(50 TFLOPS)보다 낮을 가능성이 큽니다.
> 아래 프로토콜로 **직접 측정한 값으로 반드시 교체하십시오.**

#### 측정 프로토콜 (Stage 2에서 수행)

```bash
# COCO 5k 서브셋으로 1 에포크를 돌려 실제 초당 처리량을 측정
python train.py --config configs/nano.yaml --epochs 1 --subset 5000
# 로그의 에포크 소요 시간(초)으로부터:
#   images_per_sec = 5000 / epoch_seconds
#   full_epoch_hours = 118287 / images_per_sec / 3600
#   total_days = full_epoch_hours * EPOCHS / 24
```

**데이터 로더 처리량도 함께 확인하십시오.** nano가 GPU에서 초당 ~700장을 소화하므로, 데이터 파이프라인이 그 이상을 공급하지 못하면 GPU 사용률이 떨어집니다. `nvidia-smi`의 GPU-Util이 90% 미만이면 `num_workers`를 올리십시오.

---

## 3. 티어별 에포크 계획

### 3.1 왜 300 에포크인가 — 사전학습 가중치가 없기 때문

**이 프로젝트는 ImageNet 사전학습을 쓸 수 없습니다.**
백본이 무작위 초기화이며, NPU 최적화를 위해 `LayerNorm → BatchNorm` · `GELU → Hardswish` · `kernel 7 → 5`로 바꾼 결과 **공개 ConvNeXt 가중치와 구조가 호환되지 않습니다.**

| 학습 방식 | 통상 에포크 | 사례 |
| :--- | :--- | :--- |
| ImageNet 사전학습 + 파인튜닝 | 12 ~ 36 (1x ~ 3x) | Faster R-CNN, ConvNeXt-det |
| **From-scratch + 강한 증강** | **300** | **YOLOX, YOLOv8** |

따라서 **300 에포크가 선택이 아니라 전제**입니다. 이것이 §1의 B3(Mosaic/MixUp)를 블로커로 분류한 이유이기도 합니다 — 강한 증강 없는 300 에포크 from-scratch는 과적합으로 끝납니다.

### 3.2 티어별 계획

| Scale | **총 에포크** | Warmup | Mosaic 종료 | 배치(권장) | 근거 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **nano** | **300** | 5 | 마지막 15 ep | 64 | YOLOX-Nano/Tiny 표준. 파이프라인 검증 겸용 |
| **medium** | **300** | 5 | 마지막 20 ep | 32 | 용량이 커 과적합 위험 ↑ → 증강 유지 구간 확대 |
| **xlarge** | **200** *(조건부 300)* | 10 | 마지막 20 ep | 8 (+ grad accum ×4 → 유효 32) | 29일(1×A100) 제약. 다중 GPU 확보 시 300 |

#### `xlarge` 에포크를 200으로 낮춘 이유와 대안

594 GMACs는 실용적 검출기 범위를 크게 벗어나며, 1× A100 기준 300 에포크에 **29일**이 걸립니다. 세 가지 선택지가 있습니다.

| 안 | 내용 | 비용 | 비고 |
| :---: | :--- | :--- | :--- |
| **A** | 200 에포크로 단축 | ~19일 (1×A100) | mAP 손실 예상 1~2. **DDP 없이 가능한 유일한 안** |
| **B** | 8× A100 + DDP로 300 에포크 | ~3.7일 | **CUDA DDP 구현 필요** (현재 TPU/XLA 경로만 존재) |
| **C** | 백본을 ImageNet 사전학습 후 50 에포크 파인튜닝 | 사전학습 ~12일 + 파인튜닝 ~5일 | 총 비용은 비슷하나 재사용 가능한 백본 확보 |

**권장: B.** DDP 구현 비용이 3주치 GPU 시간보다 훨씬 쌉니다. DDP를 못 넣는다면 A.

### 3.3 마지막 15~20 에포크 — 증강 종료 구간

YOLOX/YOLOv8의 핵심 기법입니다. 학습 말미에 Mosaic·MixUp을 끄면:
- 학습 분포가 실제 추론 분포(원본 이미지)와 일치하게 됩니다
- **이 구간에서만 mAP가 +1~3 오릅니다**

```
epoch 1 ────────────── 285 ──────── 300
        Mosaic ON               Mosaic OFF
        MixUp ON                MixUp OFF
        (LR 코사인 감쇠 진행)      (LR 최저 구간)
```

> 이 구간이 있기 때문에 **중간에 학습을 끊으면 안 됩니다** (§5).

---

## 4. 하이퍼파라미터 레시피

### 4.1 공통

| 항목 | 값 | 비고 |
| :--- | :--- | :--- |
| Optimizer | **AdamW** | 현재 코드와 동일 |
| Base LR | **1e-3 @ batch 64** | 배치에 선형 스케일: `lr = 1e-3 × batch/64` |
| Weight decay | **0.05** | ⚠️ 현재 코드는 `1e-4` — ConvNeXt 계열 from-scratch에는 **너무 낮습니다** |
| WD 제외 대상 | **norm 파라미터, bias, `gamma`(LayerScale), `scales`** | 이들에 WD를 걸면 성능이 떨어집니다 |
| Warmup | **5 에포크 linear** (xlarge 10) | `1e-6 → base_lr` |
| Scheduler | **CosineAnnealingLR**, `eta_min = base_lr × 0.01` | 현재 코드와 동일 (T_max 확인 필요) |
| Grad clip | **max_norm 10.0** | 현재 코드와 동일 |
| AMP | **bf16** | B5 |
| EMA decay | **0.9998** | 권장 |
| 입력 해상도 | **640 × 640** | grid buffer가 640 전용이므로 변경 시 §10 R5 참조 |

#### WD 파라미터 그룹 분리 (필수)

```python
decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    # 1차원 파라미터(norm weight/bias, LayerScale gamma, head scales)는 WD 제외
    if p.ndim <= 1 or n.endswith(".bias") or "gamma" in n or "scales" in n:
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = torch.optim.AdamW(
    [{"params": decay,    "weight_decay": 0.05},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=base_lr, betas=(0.9, 0.999),
)
```

### 4.2 증강 스케줄

| 증강 | nano | medium | xlarge | 비고 |
| :--- | :---: | :---: | :---: | :--- |
| Mosaic | p=1.0 | p=1.0 | p=1.0 | **B3 — 미구현. 최우선 구현 대상** |
| MixUp | — | p=0.1 | p=0.15 | 용량 큰 모델일수록 유효 |
| RandomAffine (scale) | 0.5~1.5 | 0.1~2.0 | 0.1~2.0 | Mosaic와 함께 적용 |
| HFlip | p=0.5 | p=0.5 | p=0.5 | ✅ 구현됨 |
| ColorJitter | p=0.5 | p=0.5 | p=0.5 | ✅ 구현됨 |
| **종료 시점** | 마지막 15 ep | 마지막 20 ep | 마지막 20 ep | Mosaic·MixUp·Affine 모두 off |

### 4.3 손실 가중치

현재 구현: `loss_total = loss_cls + 5.0 × loss_box + loss_obj` (positive 개수로 정규화)

**초기 손실 실측** (무작위 초기화):

```
cls = 11.60    box = 4.69    obj = 245.32
```

> ⚠️ **`obj`가 `cls`의 21배, `box`의 52배로 지배적입니다.**
> 8,400개 grid point 중 positive가 4~12개뿐인 구조상 자연스러운 현상이며 YOLOX와 동일한 정규화 방식이지만,
> 학습 초기에 모델이 "아무것도 없다"로 수렴할 위험이 있습니다.

**대응**:
1. Stage 2(서브셋)에서 세 손실의 **상대 비율 추이**를 TensorBoard로 확인
2. 50 에포크 시점에 `obj`가 여전히 `cls+box`의 10배 이상이면 → `obj` 가중치를 0.5로 낮추거나 SimOTA 도입으로 positive 수를 늘림
3. `box` 가중치 5.0은 YOLOX 관행값. 서브셋 실험에서 3.0 / 5.0 / 7.5 비교 권장

### 4.4 배치 크기와 메모리

| Scale | 배치(1×A100 80GB, bf16) | Grad Accum | 유효 배치 | LR |
| :--- | ---: | ---: | ---: | ---: |
| nano | 64 | 1 | 64 | 1e-3 |
| medium | 32 | 2 | 64 | 1e-3 |
| xlarge | 8 | 4 | 32 | 5e-4 |

> ⚠️ **BatchNorm 주의**: 이 모델은 BN을 쓰므로 GPU당 배치가 작으면 통계가 불안정합니다.
> `xlarge`처럼 GPU당 배치가 8 이하이면 **SyncBatchNorm**(다중 GPU) 또는 GroupNorm 전환을 검토하십시오.
> 단, GroupNorm은 Int8 fold가 불가능하므로 **NPU 배포 목표와 상충**합니다 — SyncBN을 우선하십시오.

---

## 5. Early Stopping 정책

### 5.1 결론: 전통적 Early Stopping은 쓰지 않습니다

**"val mAP가 N 에포크 동안 개선되지 않으면 학습 중단"** 방식은 이 프로젝트에 부적합합니다.

| 이유 | 설명 |
| :--- | :--- |
| **코사인 스케줄의 말단 이득** | LR이 최저로 떨어지는 마지막 10~20% 구간에서 mAP가 **+1~3** 오릅니다. 중간에 끊으면 이 이득을 통째로 잃습니다 |
| **증강 종료 구간** | §3.3의 마지막 15~20 에포크는 설계상 가장 큰 개선이 일어나는 구간입니다 |
| **mAP의 에포크 간 노이즈** | COCO mAP는 에포크마다 ±0.3 내외로 흔들립니다. patience 10~20으로는 **오탐 중단**이 빈번합니다 |
| **긴 정체 구간** | from-scratch 학습은 중반에 30~50 에포크씩 정체하다가 다시 오르는 패턴이 흔합니다 |

### 5.2 대신 쓰는 것 — 3층 방어

```
① best 체크포인트 선정   → 사실상의 Early Stopping (학습은 끝까지, 모델은 최적점을 채택)
② 발산 감지 중단          → 안전장치 (모델 선택이 아닌 사고 방지)
③ 정체 중단 (탐색 런 한정) → 하이퍼파라미터 스윕에서만 사용
```

#### ① best 체크포인트 선정 — 필수 (B2)

**학습은 계획된 에포크까지 전부 돌리되, 저장하는 모델은 val mAP 최고점**을 씁니다.
"조기 종료"의 이득(과적합 회피)은 얻고, 손실(말단 이득 상실)은 없습니다.

| 항목 | 값 |
| :--- | :--- |
| 평가 주기 | 초반 1~50 ep: 10 에포크마다 / 이후: 5 에포크마다 / **마지막 30 ep: 1 에포크마다** |
| 선정 지표 | **`map` (COCO mAP@[.5:.95])** — `map_50` 아님 |
| 평가 대상 | **EMA 가중치** (EMA 사용 시) |

> 평가 주기를 후반에 촘촘하게 두는 이유는, 실제 최고점이 거의 항상 마지막 구간에 나오기 때문입니다.
> `nano` 기준 val 5k 평가는 GPU에서 1분 이내이므로 부담이 없습니다.

#### ② 발산 감지 중단 — 필수 안전장치

며칠~몇 주짜리 런에서 **사고를 조기에 잡기 위한** 장치입니다. 모델 선택과는 무관합니다.

| 조건 | 조치 |
| :--- | :--- |
| `loss_total`이 `NaN`/`Inf` | **즉시 중단** |
| `loss_total`이 직전 100 스텝 평균의 **5배 초과** | **즉시 중단** (LR 과대 의심) |
| warmup 종료 후 20 에포크 동안 `loss_total`이 초기값 대비 **10% 미만 감소** | **중단 후 LR 재검토** |
| 첫 val 평가(ep 10)에서 `map == 0.0` | **중단** — 데이터/할당 배선 오류 의심 |

```python
if not torch.isfinite(loss):
    raise RuntimeError(f"Epoch {epoch} step {i}: loss 발산 ({loss.item()}). 학습 중단.")
```

#### ③ 정체 중단 — 탐색 런에서만

하이퍼파라미터 스윕이나 서브셋 실험에서 **자원을 아끼기 위해서만** 사용합니다.

| 항목 | 값 |
| :--- | :--- |
| 감시 지표 | val `map` |
| **patience** | **50 에포크** (300 에포크 계획 기준) |
| min_delta | +0.002 (0.2 mAP) |
| 적용 대상 | **Stage 2 서브셋 실험 · 하이퍼파라미터 스윕 전용** |
| 본 학습(Stage 3~5) 적용 | ❌ **사용 금지** |

> patience를 50으로 크게 잡는 이유는 §5.1의 "긴 정체 구간" 때문입니다.
> patience 10~20은 정상적으로 학습 중인 런을 죽입니다.

### 5.3 요약

| 런 종류 | Early Stop | best 선정 | 발산 감지 |
| :--- | :---: | :---: | :---: |
| Stage 1 오버피팅 테스트 | ❌ | ❌ (loss만 확인) | ✅ |
| Stage 2 서브셋 / 스윕 | ✅ patience 50 | ✅ val mAP | ✅ |
| **Stage 3~5 본 학습** | ❌ **금지** | ✅ val mAP | ✅ |

---

## 6. 단계별 실행 계획

### Stage 0 — 블로커 해소 *(학습 없음)*

§1의 B1~B6 구현. 완료 판정:

- [ ] `scripts/download_coco.sh` 실행 → `data/{train2017,val2017,annotations}` 확보
- [ ] `train.py --config configs/nano.yaml` 로 실행되고 에포크마다 val mAP가 로그·TensorBoard에 남는다
- [ ] `best_*.pt`가 **val mAP 기준**으로 갱신된다
- [ ] Mosaic 적용 결과를 이미지로 저장해 **박스가 정확히 따라가는지 육안 확인**했다
- [ ] `--resume checkpoints/last_nano.pt` 로 정확히 이어서 학습된다 (optimizer·scheduler·epoch·EMA 포함)
- [ ] AMP 활성 상태에서 loss가 `NaN` 없이 100 스텝 진행된다
- [ ] `num_workers` 조정 후 `nvidia-smi` GPU-Util ≥ 90%

### Stage 1 — 오버피팅 테스트 *(수 분)*

**목적**: 손실·할당·디코딩 배선 검증

```bash
python train.py --config configs/nano.yaml --epochs 200 --subset 8 --no-aug
```

| 기대 | 판정 |
| :--- | :--- |
| `loss_cls → 0`, `loss_box → 0` | ✅ 현재 코드에서 이미 통과 확인됨 |
| 8장에 대한 시각화에서 박스가 GT와 겹침 | 신규 확인 필요 |

> ⚠️ 현재는 `cls`만 0으로 수렴하고 `box`는 0.6~1.6에서 정체합니다.
> **실데이터·Mosaic 적용 후 이 구간이 0에 더 가까워지는지 반드시 확인하십시오.** 정체하면 assigner(positive 부족)를 먼저 의심해야 합니다.

### Stage 2 — 서브셋 레시피 검증 *(nano, ~2시간)*

**목적**: 300 에포크를 태우기 전에 레시피가 맞는지 확인. **가장 중요한 단계입니다.**

```bash
python train.py --config configs/nano.yaml --epochs 50 \
    --train-subset 5000 --val-subset 1000
```

| 확인 항목 | 기대 |
| :--- | :--- |
| **실제 처리량 측정** → §2.3 표 교체 | images/sec 기록 |
| 50 에포크 시점 val `map` | **> 0.05** (5k 서브셋 기준). 0이면 배선 오류 |
| `obj / (cls + box)` 비율 | 10배 미만으로 하락. 아니면 §4.3 대응 |
| Mosaic on/off 비교 | Mosaic 적용 쪽이 유의미하게 높아야 함 |
| WD 0.05 vs 1e-4 비교 | 어느 쪽이 나은지 확정 |
| box 가중치 3.0 / 5.0 / 7.5 | 최적값 확정 |

**이 단계를 건너뛰면 안 됩니다.** 2시간 투자로 최대 3주(xlarge)의 오학습을 막습니다.

### Stage 3 — nano 본 학습 *(300 ep, ~14시간 @1×A100)*

```bash
python train.py --config configs/nano.yaml --epochs 300
```

- 파이프라인 전체를 실증하는 기준 런
- 완료 후 **FP32 mAP 확정** → 이후 모든 비교의 기준선

### Stage 4 — medium 본 학습 *(300 ep, ~2.9일 @1×A100)*

Stage 3에서 검증된 레시피를 배치·LR만 조정해 적용. nano 대비 mAP 향상 폭으로 스케일링 룰이 유효한지 확인합니다.

### Stage 5 — xlarge 본 학습 *(조건부)*

**선행 조건: CUDA DDP 구현 + 다중 GPU 확보.** 미충족 시 §3.2의 A안(200 에포크)으로 축소하거나, **티어 재설계 결정(§10 R4) 이후로 연기**하십시오.

### Stage 6 — 양자화 및 배포 *(§9)*

### 실행 순서 요약

```
Stage 0 (블로커)  →  Stage 1 (오버피팅, 수 분)  →  Stage 2 (서브셋, 2h)
                                                        │
                          ┌─────────────────────────────┘
                          ▼
                   Stage 3 nano (14h)  →  Stage 4 medium (2.9d)  →  Stage 5 xlarge (조건부)
                          │
                          └──→ Stage 6 PTQ / QAT
```

---

## 7. 모니터링 및 중단 기준

### 7.1 매 스텝 기록

| 지표 | 이유 |
| :--- | :--- |
| `loss_total`, `loss_cls`, `loss_box`, `loss_obj` | 개별 손실 비율 감시 (§4.3) |
| `lr` | warmup·코사인이 의도대로 도는지 |
| `num_pos` (배치당 positive 수) | **현재 코드에 미기록. 추가 권장** — assigner 이상을 가장 빨리 드러냅니다 |
| `grad_norm` (clip 전) | 발산 조기 감지 |

### 7.2 매 평가 시점 기록

`map`, `map_50`, `map_75`, `map_small`, `map_medium`, `map_large`

> **`map_small`을 특히 주시하십시오.** 이 아키텍처는 stride 4 스템에서 시작해 P3 해상도가 큰 대신,
> assigner가 레벨별 size range로 소형 객체를 stride 8에만 할당합니다.
> `map_small`이 유독 낮으면 `center_radius`(현재 1.5) 또는 size range 경계를 조정해야 합니다.

### 7.3 중단 기준

§5.2 ②의 발산 감지와 동일. 요약:

| 시점 | 조건 | 조치 |
| :--- | :--- | :--- |
| 매 스텝 | loss가 `NaN`/`Inf` | 즉시 중단 |
| 매 스텝 | loss > 직전 100스텝 평균 × 5 | 즉시 중단, LR 하향 |
| ep 10 | val `map == 0.0` | 중단, 데이터·할당 점검 |
| ep 30 | val `map < 0.02` | 중단, Stage 2로 복귀 |
| warmup+20 ep | loss 감소율 < 10% | 중단, LR 재검토 |

---

## 8. 체크포인트 정책

| 파일 | 저장 시점 | 용도 |
| :--- | :--- | :--- |
| `last_{scale}.pt` | **매 에포크** | resume 전용 |
| `best_{scale}.pt` | **val mAP 갱신 시** | 배포·평가·PTQ 입력 |
| `ep{N}_{scale}.pt` | 50 에포크마다 | 사후 분석·롤백 |

### 저장 내용 (resume 필수 요건)

```python
torch.save({
    'epoch': epoch,
    'model': model.state_dict(),
    'ema': ema.state_dict(),              # EMA 사용 시
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'scaler': scaler.state_dict(),        # AMP 사용 시
    'best_map': best_map,
    'config': cfg,
}, path)
```

> 현재 코드는 `model.state_dict()`만 저장합니다. **optimizer/scheduler 없이는 resume이 불가능**하며,
> AdamW의 모멘텀 상태를 잃으면 재개 직후 학습이 크게 흔들립니다.

### PTQ와의 연결

`utils/quantize.py`는 `checkpoints/best_{scale}.pt`를 로드합니다.
`best`가 **val mAP 기준으로 선정된 EMA 가중치**여야 양자화 결과가 의미를 갖습니다 (B2가 블로커인 또 다른 이유).

---

## 9. 양자화 학습 (QAT) 계획

Int8 PTQ는 이미 동작하며, 무작위 가중치 기준 FP32 대비 **argmax 일치율 99.8% / 로짓 상대오차 0.5~1.8%** 를 확인했습니다 ([6차 리뷰](./review/20260826-review-4.md) §1.5).

### 9.1 순서

```
Stage 3~5 완료 (FP32 mAP 확정)
   ↓
PTQ 적용 → Int8 mAP 측정 → FP32 대비 하락폭 산출
   ↓
하락폭 ≤ 1.0 mAP  →  PTQ로 종료 ✅
하락폭 >  1.0 mAP  →  QAT 진행
```

### 9.2 QAT 레시피 (필요 시)

| 항목 | 값 |
| :--- | :--- |
| 시작 가중치 | FP32 `best_{scale}.pt` |
| 에포크 | **10 ~ 20** (전체 학습의 3~7%) |
| LR | FP32 최종 LR의 **1/10** (`base_lr × 0.001` 수준) |
| Scheduler | Cosine 또는 고정 |
| 증강 | **Mosaic/MixUp 없음** — 추론 분포와 맞춤 |
| BN | 초반 몇 에포크 후 **freeze** (통계 흔들림 방지) |
| 대상 우선순위 | `nano` → `medium` → `xlarge` |

> `nano`는 파라미터가 적어 양자화 손실이 가장 크게 나타나는 티어이므로, QAT가 필요하다면 nano부터입니다.

---

## 10. 리스크와 대응

| ID | 리스크 | 영향 | 대응 |
| :---: | :--- | :--- | :--- |
| **R1** | **Mosaic 미구현 상태로 300 ep 학습** | 과적합으로 mAP 수 포인트 손실. **수 주 낭비** | B3를 Stage 0에서 반드시 완료. Stage 2에서 on/off 비교로 효과 확인 |
| **R2** | **positive 샘플 부족** (GT당 4~12개) | 수렴 지연, `map_small` 저조 | Stage 2에서 `num_pos` 로깅. 부족하면 `center_radius` 상향 또는 **SimOTA 도입** |
| **R3** | **`obj` 손실 지배** (cls의 21배) | 초기에 "배경만 예측"으로 수렴 | Stage 2에서 비율 추이 감시. §4.3 대응 |
| **R4** | **nano 11.76 / xlarge 594 GMACs** — 티어 정의와 괴리 | 학습을 마쳐도 "엣지 실시간" 목표 미달 | **학습 착수 전** 결정: 모델 축소 vs 문서의 티어 정의 수정. 학습 후 아키텍처를 바꾸면 전부 재학습 |
| **R5** | **해상도 640 고정** | grid buffer가 640 전용, 다중 스케일 학습 불가 | 640 고정으로 진행. 다중 스케일이 필요하면 buffer 로직 일반화 선행 |
| **R6** | **resume 없이 장기 런 중단** | medium 3일 / xlarge 29일 소실 | B4 필수. 추가로 저장은 `last` → 임시파일 → `os.replace` 원자적 교체 |
| **R7** | **BN 통계 불안정** (xlarge 배치 8) | mAP 저하 | SyncBN 적용. GroupNorm은 Int8 fold 불가하므로 회피 |
| **R8** | **데이터 로더 병목** | GPU 유휴, 학습 시간 수 배 증가 | B6. Stage 2에서 GPU-Util 확인 |
| **R9** | **재현 불가** (seed 미고정) | 실험 비교 무의미 | `manual_seed` + `cudnn.benchmark=True` (deterministic은 속도 손해가 커 비권장) |

---

## 부록: 완료 판정 체크리스트

### Stage 0

- [ ] COCO 2017 train/val 확보 및 `COCODataset`이 `dummy_mode=False`
- [ ] val 검증 루프 + val mAP 기준 best 선정
- [ ] Mosaic / MixUp / RandomAffine 구현 + 박스 정합 육안 확인
- [ ] resume (model·EMA·optimizer·scheduler·scaler·epoch)
- [ ] AMP(bf16) 적용 후 100 스텝 NaN 없음
- [ ] `num_workers` 튜닝 후 GPU-Util ≥ 90%
- [ ] EMA · seed 고정 · configs 배선

### Stage 2 (게이트 — 통과 못 하면 Stage 3 진입 금지)

- [ ] 실제 images/sec 측정 완료, §2.3 추정치 교체
- [ ] 5k 서브셋 50 ep에서 val `map` > 0.05
- [ ] `obj / (cls + box)` < 10
- [ ] Mosaic on > off 확인
- [ ] WD·box 가중치 최적값 확정

### Stage 3~5

- [ ] FP32 `map` / `map_50` / `map_small` 기록
- [ ] best 체크포인트가 val mAP 기준이며 EMA 가중치임
- [ ] 학습 곡선(loss 3종 + mAP)이 TensorBoard에 남음
- [ ] 동일 seed 재실행 시 mAP 편차 < 0.5

### Stage 6

- [ ] PTQ Int8 mAP 측정, FP32 대비 하락폭 기록
- [ ] 하락폭 > 1.0이면 QAT 수행 후 재측정
- [ ] ONNX export 후 ONNX Runtime 수치 일치 확인 (rtol 1e-3)
