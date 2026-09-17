#!/bin/bash
# COCO 2017 Dataset Download & Setup Script
set -e

DATA_DIR="./data"
mkdir -p "$DATA_DIR"

echo "=== 1. 다운로드 대상 선택 ==="
DOWNLOAD_FULL=false
if [ "$1" == "--full" ]; then
    DOWNLOAD_FULL=true
fi

cd "$DATA_DIR"

echo "=== 2. Annotations 다운로드 및 압축 해제 ==="
if [ ! -f "annotations/instances_val2017.json" ]; then
    echo "Downloading annotations_trainval2017.zip (~241MB)..."
    curl -L -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q annotations_trainval2017.zip
    rm annotations_trainval2017.zip
fi

cp annotations/instances_val2017.json val_annotations.json
if [ -f "annotations/instances_train2017.json" ]; then
    cp annotations/instances_train2017.json annotations.json
fi

echo "=== 3. Validation 이미지 (val2017, ~1GB, 5000장) 다운로드 ==="
if [ ! -d "val_images" ]; then
    echo "Downloading val2017.zip..."
    curl -L -O http://images.cocodataset.org/zips/val2017.zip
    unzip -q val2017.zip
    mv val2017 val_images
    rm val2017.zip
fi

if [ "$DOWNLOAD_FULL" = true ]; then
    echo "=== 4. Training 이미지 (train2017, ~19GB, 118,287장) 다운로드 ==="
    if [ ! -d "images" ]; then
        echo "Downloading train2017.zip (대용량 다운로드)..."
        curl -L -O http://images.cocodataset.org/zips/train2017.zip
        unzip -q train2017.zip
        mv train2017 images
        rm train2017.zip
    fi
else
    echo "💡 전체 학습 데이터셋(19GB)을 다운로드하려면 다음 명령을 실행하세요:"
    echo "   bash scripts/download_coco.sh --full"
    if [ ! -d "images" ]; then
        echo "임시 학습용으로 val_images를 images로 미러링합니다."
        ln -s val_images images
        cp val_annotations.json annotations.json
    fi
fi

echo "=== ✅ COCO 2017 데이터 준비 완료! ==="
ls -lh
