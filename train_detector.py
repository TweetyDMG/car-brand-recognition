#!/usr/bin/env python3
"""
Обучение YOLO-детектора логотипов автомобилей.

Датасет:
    datasets/car_logos_detection/  (формат YOLO, 1 класс — logo)

Использование:
    python train_detector.py
"""

import os
import sys
import yaml
from ultralytics import YOLO

from src.config import (
    PROJECT_ROOT,
    DETECTION_DATASET_DIR,
    DETECTOR_EPOCHS,
    DETECTOR_BATCH,
    DETECTOR_IMG_SIZE,
    DETECTOR_MODEL_SIZE,
)


def prepare_dataset_yaml() -> bool:
    """Проверить наличие и вывести структуру датасета."""
    yaml_path = DETECTION_DATASET_DIR / "data.yaml"
    if not yaml_path.exists():
        print(f"Файл {yaml_path} не найден!")
        print("Скачайте датасет через: python download_datasets.py")
        return False

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    print(f"Датасет загружен: {data.get('nc', 0)} классов")
    print(f"Классы: {data.get('names', [])}")
    print(f"Train: {data.get('train', 'N/A')}")
    print(f"Valid: {data.get('val', 'N/A')}")
    print(f"Test:  {data.get('test', 'N/A')}")

    return True


def train_detector():
    """Полный цикл обучения YOLO-детектора."""
    if not prepare_dataset_yaml():
        sys.exit(1)

    yaml_path = str(DETECTION_DATASET_DIR / "data.yaml")

    print(f"\nЗагрузка модели {DETECTOR_MODEL_SIZE}...")
    model = YOLO(DETECTOR_MODEL_SIZE)

    print("\nПараметры обучения:")
    print(f"  Эпохи:       {DETECTOR_EPOCHS}")
    print(f"  Batch size:  {DETECTOR_BATCH}")
    print(f"  Размер:      {DETECTOR_IMG_SIZE}")
    print(f"  Устройство:  GPU (0) / CPU")
    print(f"\nНачало обучения...\n")

    model.train(
        data=yaml_path,
        epochs=DETECTOR_EPOCHS,
        imgsz=DETECTOR_IMG_SIZE,
        batch=DETECTOR_BATCH,
        device=0,
        workers=8,
        project="runs/detect",
        name="car_logo_detector",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        save_period=10,
    )

    print("\nОбучение завершено.")
    print("Результаты сохранены в: runs/detect/car_logo_detector")

    # Валидация
    print("\nВалидация модели...")
    metrics = model.val()
    print(f"\nМетрики:")
    print(f"  mAP@0.50:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.50:0.95: {metrics.box.map:.4f}")
    print(f"  Precision:     {metrics.box.mp:.4f}")
    print(f"  Recall:        {metrics.box.mr:.4f}")

    # Сохраняем лучшую модель в единую директорию models/
    best_model_path = "runs/detect/car_logo_detector/weights/best.pt"
    model = YOLO(best_model_path)
    print(f"\nЛучшая модель: {best_model_path}")

    return model


if __name__ == "__main__":
    model = train_detector()
    print("\n✓ ПРОЦЕСС ЗАВЕРШЕН")
