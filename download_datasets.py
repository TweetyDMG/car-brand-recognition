#!/usr/bin/env python3
"""
Скрипт для скачивания датасетов с Roboflow.

Использование:
    export ROBOFLOW_API_KEY=your_key
    python download_datasets.py

Или через .env (скопируйте .env.example в .env и укажите ключ).
"""

import os
import shutil
import yaml
import sys

from roboflow import Roboflow
from src.config import (
    PROJECT_ROOT,
    ROBOFLOW_API_KEY,
    ROBOFLOW_WORKSPACE,
    ROBOFLOW_DETECTION_PROJECT,
    ROBOFLOW_DETECTION_VERSION,
    DATASETS_DIR,
    DETECTION_DATASET_DIR,
    CLASSIFICATION_DATASET_DIR,
)


def download_detection_dataset():
    print("=" * 50)
    print("СКАЧИВАНИЕ ДАТАСЕТА ДЛЯ ДЕТЕКЦИИ")
    print("=" * 50)

    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_DETECTION_PROJECT)
        version = project.version(ROBOFLOW_DETECTION_VERSION)

        temp_location = str(DATASETS_DIR / "temp_detection")
        dataset = version.download("yolov8", location=temp_location)
        downloaded_path = dataset.location
        print(f"Датасет скачан в: {downloaded_path}")

        target = str(DETECTION_DATASET_DIR)
        if downloaded_path != target:
            print(f"\nПеремещение датасета в {target}...")
            if DETECTION_DATASET_DIR.exists():
                shutil.rmtree(target)
            shutil.move(downloaded_path, target)
            temp_dir = DATASETS_DIR / "temp_detection"
            if temp_dir.exists():
                shutil.rmtree(str(temp_dir))
            print(f"Датасет перемещен в: {target}")

        print("\nСтруктура датасета:")
        check_dataset_structure(DETECTION_DATASET_DIR)

        yaml_path = DETECTION_DATASET_DIR / "data.yaml"
        if yaml_path.exists():
            print(f"\nФайл конфигурации найден: {yaml_path}")
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            print(f"\nИнформация о датасете:")
            print(f"   - Количество классов: {data.get('nc', 'N/A')}")
            print(f"   - Классы: {data.get('names', 'N/A')}")
            print(f"   - Train: {data.get('train', 'N/A')}")
            print(f"   - Valid: {data.get('val', 'N/A')}")
            print(f"   - Test: {data.get('test', 'N/A')}")

        return True

    except Exception as e:
        print(f"\nОшибка при скачивании датасета для детекции: {e}")
        return False


def check_dataset_structure(dataset_path):
    """Проверить структуру YOLO-датасета (images + labels)."""
    if not dataset_path.exists():
        print(f"Директория не найдена: {dataset_path}")
        return

    for split in ("train", "valid", "test"):
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"  {split}/ — не найден")
            continue

        images_path = split_path / "images"
        labels_path = split_path / "labels"

        if images_path.exists():
            count = len([
                f for f in images_path.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            ])
            print(f"  {split}/images/: {count} изображений")

        if labels_path.exists():
            count = len(list(labels_path.glob("*.txt")))
            print(f"  {split}/labels/: {count} меток")


def main():
    if not ROBOFLOW_API_KEY:
        print("ОШИБКА: ROBOFLOW_API_KEY не задан!")
        print()
        print("Скопируйте .env.example в .env и укажите ваш ключ:")
        print("  cp .env.example .env")
        print("  # отредактируйте .env, вставив ключ")
        print()
        print("Или передайте переменную в команде:")
        print("  ROBOFLOW_API_KEY=your_key python download_datasets.py")
        sys.exit(1)

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    detection_ok = download_detection_dataset()
    if detection_ok:
        print("\n✅ Датасет для детекции успешно скачан")
        print(f"   Расположение: {DETECTION_DATASET_DIR}")
    else:
        print("\n❌ Датасет для детекции: ошибка")


if __name__ == "__main__":
    main()
