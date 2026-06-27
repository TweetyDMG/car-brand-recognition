#!/usr/bin/env python3
"""
Автономный инференс для распознавания марок автомобилей.

Двухстадийный пайплайн:
  1. YOLO — детекция области с логотипом
  2. EfficientNet-B0 (PyTorch) — классификация бренда

Использование:
    python inference.py
    # или указать свой файл:
    python inference.py --image test_images/img_1.png

Поддерживаемые бренды: Hyundai, Lexus, Mazda, Mercedes, Opel, Skoda, Toyota, Volkswagen.
"""

import cv2
import os
import sys
import argparse
from ultralytics import YOLO
import torch
from torchvision import transforms, models
import torch.nn as nn

from src.config import (
    DETECTOR_PATH,
    CLASSIFIER_PATH,
    CLASS_LIST,
    NUM_CLASSES,
    DETECTION_CONF_THRESHOLD,
    CLASSIFIER_CONF_THRESHOLD,
    PROJECT_ROOT,
)


class CarBrandRecognizer:
    """
    Двухстадийный пайплайн: детектор YOLO + классификатор EfficientNet-B0.

    Args:
        detector_path: Путь к весам YOLO-детектора.
        classifier_path: Путь к весам PyTorch-классификатора (.pth).
        class_list: Список названий классов (в нижнем регистре).
    """

    def __init__(
        self,
        detector_path: str = DETECTOR_PATH,
        classifier_path: str = CLASSIFIER_PATH,
        class_list: list = None,
        img_size: tuple = (224, 224),
    ):
        self.class_list = class_list or CLASS_LIST
        self.num_classes = len(self.class_list)

        print("Инициализация моделей...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Устройство: {self.device}")

        # 1. Детектор YOLO
        print(f"Загрузка детектора: {detector_path}")
        self.detector = YOLO(detector_path)
        print("Детектор загружен.")

        # 2. Классификатор EfficientNet-B0
        print(f"Загрузка классификатора: {classifier_path}")
        self.classifier = models.efficientnet_b0(weights=None)
        num_features = self.classifier.classifier[1].in_features
        self.classifier.classifier[1] = nn.Linear(num_features, self.num_classes)
        state_dict = torch.load(classifier_path, map_location=self.device)
        self.classifier.load_state_dict(state_dict)
        self.classifier = self.classifier.to(self.device).eval()
        print(f"Классификатор загружен. Количество классов: {self.num_classes}")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def detect_logos(self, image, conf_threshold: float = DETECTION_CONF_THRESHOLD):
        """
        Найти все логотипы на изображении.

        Returns:
            Список кортежей (x1, y1, x2, y2, confidence).
        """
        results = self.detector(image, conf=conf_threshold, verbose=False)
        detections = []
        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append((x1, y1, x2, y2, conf))
        return detections

    def classify_crop(self, crop_bgr):
        """
        Классифицировать вырезанную область с логотипом.

        Returns:
            (top_class, top_confidence, top3_list)
        """
        img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.classifier(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)

        top_probs, top_indices = torch.topk(probs, 3)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        top3 = [
            (self.class_list[i], float(top_probs[j]))
            for j, i in enumerate(top_indices)
        ]
        top_idx = top_indices[0]
        top_conf = float(top_probs[0])
        top_class = self.class_list[top_idx]
        return top_class, top_conf, top3

    def process_image(
        self,
        image_path: str,
        output_path: str = None,
        detection_threshold: float = DETECTION_CONF_THRESHOLD,
    ):
        """
        Полный пайплайн: детекция → классификация.

        Args:
            image_path: Путь к входному изображению.
            output_path: Куда сохранить аннотированный результат (если None — не сохранять).
            detection_threshold: Порог confidence для YOLO.

        Returns:
            (results_list, annotated_image)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение {image_path}")

        print(f"\nОбработка изображения: {image_path}")
        detections = self.detect_logos(image, detection_threshold)
        print(f"Найдено логотипов: {len(detections)}")

        out = image.copy()
        results = []

        for idx, (x1, y1, x2, y2, det_conf) in enumerate(detections, start=1):
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            brand, conf, top3 = self.classify_crop(crop)
            results.append({
                "brand": brand,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "detection_conf": det_conf,
                "top3": top3,
            })

            # Визуализация
            color = (0, 255, 0) if conf >= 0.7 else (0, 165, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{brand} {conf:.2f}"
            cv2.putText(
                out, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            cv2.imwrite(output_path, out)
            print(f"Результат сохранен: {output_path}")

        return results, out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Car Brand Recognition — инференс изображений"
    )
    parser.add_argument(
        "--image", "-i",
        default=str(PROJECT_ROOT / "test_images" / "img_1.png"),
        help="Путь к входному изображению",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(PROJECT_ROOT / "results" / "result.jpg"),
        help="Путь для сохранения результата",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=DETECTION_CONF_THRESHOLD,
        help="Порог уверенности YOLO (по умолчанию 0.4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"Файл не найден: {args.image}")
        sys.exit(1)

    try:
        recognizer = CarBrandRecognizer()
        results, annotated = recognizer.process_image(
            args.image,
            output_path=args.output,
            detection_threshold=args.threshold,
        )

        print("\nРезультаты:")
        if not results:
            print("  Логотип не обнаружен.")
        for r in results:
            print(f"  - {r['brand']}: {r['confidence']:.2%} (bbox: {r['bbox']})")
            print(f"    Top-3: {[(b, f'{c:.2%}') for b, c in r['top3']]}")

    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
