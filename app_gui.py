#!/usr/bin/env python3
"""
Десктопное GUI-приложение для распознавания марок автомобилей.

Использование:
    python app_gui.py

Зависит от обученных моделей:
    - models/car_logo_detector.pt (YOLO)
    - models/car_logo_classifier.pth (EfficientNet-B0)
"""

import cv2
from ultralytics import YOLO
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import numpy as np
import os
import sys

from src.config import (
    DETECTOR_PATH,
    CLASSIFIER_PATH,
    CLASS_LIST_RU,
    DETECTION_CONF_THRESHOLD,
    CLASSIFIER_CONF_THRESHOLD,
    NUM_CLASSES,
)


class CarBrandRecognizer:
    """Двухстадийный пайплайн для GUI: YOLO-детекция + EfficientNet-классификация."""

    def __init__(self, detector_path: str, classifier_pth_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. YOLO-детектор
        self.detector = YOLO(detector_path)

        # 2. EfficientNet-B0 классификатор
        self.classifier = models.efficientnet_b0(weights=None)
        num_features = self.classifier.classifier[1].in_features
        self.classifier.classifier[1] = nn.Linear(num_features, NUM_CLASSES)

        state_dict = torch.load(classifier_pth_path, map_location=self.device)
        self.classifier.load_state_dict(state_dict)
        self.classifier = self.classifier.to(self.device).eval()

        self.class_list = CLASS_LIST_RU

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, image_path):
        """
        Полный пайплайн: детекция → классификация → аннотация.

        Returns:
            (brand_name, confidence_str, annotated_image_bgr)
        """
        img_cv2 = cv2.imread(image_path)
        if img_cv2 is None:
            return None, "Ошибка чтения изображения", None

        # Детекция
        results = self.detector(img_cv2, conf=DETECTION_CONF_THRESHOLD, verbose=False)
        best_conf = 0
        best_crop = None
        best_box = None

        for res in results:
            for box in res.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    best_conf = conf
                    best_crop = img_cv2[y1:y2, x1:x2]
                    best_box = (x1, y1, x2, y2)

        img_annotated = img_cv2.copy()

        if best_box is None:
            return None, "Логотип не найден", img_annotated

        x1, y1, x2, y2 = best_box

        # Аннотация через PIL
        pil_img = Image.fromarray(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except OSError:
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20
                )
            except OSError:
                font = ImageFont.load_default()

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Классификация
        img_rgb = cv2.cvtColor(best_crop, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.classifier(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)

        top_prob, top_idx = torch.topk(probs, 1)
        conf_val = float(top_prob[0])

        # Низкая уверенность — не показываем бренд
        if conf_val < CLASSIFIER_CONF_THRESHOLD:
            text = "Логотип"
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.rectangle(
                [(x1, y1 - th - 10), (x1 + tw + 4, y1 - 4)], fill="red"
            )
            draw.text((x1 + 2, y1 - th - 6), text, fill="white", font=font)
            img_annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return None, "Марка не определена (низкая уверенность)", img_annotated

        brand_name = self.class_list[top_idx.item()]
        label_text = f"{brand_name} ({conf_val:.0%})"
        bbox = draw.textbbox((0, 0), label_text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        draw.rectangle(
            [(x1, y1 - th - 10), (x1 + tw + 4, y1 - 4)], fill=(255, 165, 0)
        )
        draw.text((x1 + 2, y1 - th - 6), label_text, fill="white", font=font)

        img_annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return brand_name, f"{conf_val:.2%}", img_annotated


class CarBrandApp:
    """Tkinter GUI для загрузки фото и отображения результата."""

    def __init__(self, root, recognizer):
        self.root = root
        self.recognizer = recognizer
        self.current_image_path = None
        self.photo_image = None

        self.root.title("Распознавание марок автомобилей")
        self.root.minsize(600, 500)
        self.root.configure(bg="#F5F5F5")

        # Центрирование окна
        w, h = 800, 600
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

        self.create_widgets()

    def create_widgets(self):
        main_frame = tk.Frame(self.root, bg="#F5F5F5")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

        self.result_label = tk.Label(
            main_frame,
            text="Загрузите изображение для распознавания",
            font=("Arial", 24, "bold"),
            bg="#F5F5F5",
            fg="#333333",
        )
        self.result_label.pack(pady=(0, 20))

        self.image_frame = tk.Frame(main_frame, bg="white", relief=tk.FLAT, bd=0)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        self.image_label = tk.Label(
            self.image_frame,
            bg="white",
            text="Здесь появится изображение",
            font=("Arial", 14),
            fg="#999999",
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.load_button = tk.Button(
            main_frame,
            text="📷 Загрузить фото",
            font=("Arial", 18, "bold"),
            bg="#4CAF50",
            fg="white",
            activebackground="#45a049",
            cursor="hand2",
            relief=tk.FLAT,
            bd=0,
            padx=40,
            pady=15,
            command=self.load_image,
        )
        self.load_button.pack(pady=(0, 10))

        brands_info = tk.Label(
            main_frame,
            text="Поддерживаемые марки: "
                 "Hyundai • Lexus • Mazda • Mercedes • Opel • Skoda • Toyota • Volkswagen",
            font=("Arial", 10),
            bg="#F5F5F5",
            fg="#777777",
        )
        brands_info.pack(pady=(5, 0))

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png *.bmp"),
                ("Все файлы", "*.*"),
            ],
        )
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.process_image(file_path)

    def display_image(self, image_path):
        try:
            img = Image.open(image_path)
            fw = max(self.image_frame.winfo_width(), 740)
            fh = max(self.image_frame.winfo_height(), 400)
            img.thumbnail((fw - 20, fh - 20), Image.Resampling.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(img)
            self.image_label.configure(image=self.photo_image, text="")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{str(e)}")

    def process_image(self, image_path):
        self.result_label.configure(text="Обработка изображения...")
        self.load_button.configure(state=tk.DISABLED)
        thread = threading.Thread(target=self._process_worker, args=(image_path,))
        thread.daemon = True
        thread.start()

    def _process_worker(self, image_path):
        try:
            brand, confidence, img_annotated = self.recognizer.predict(image_path)
            img_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            self.root.after(0, self._update_result, brand, confidence, pil_img)
        except Exception as e:
            self.root.after(0, self._show_error, str(e))

    def _update_result(self, brand, confidence, pil_img):
        self.load_button.configure(state=tk.NORMAL)

        fw = max(self.image_frame.winfo_width(), 740)
        fh = max(self.image_frame.winfo_height(), 400)
        pil_img.thumbnail((fw - 20, fh - 20), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(pil_img)
        self.image_label.configure(image=self.photo_image, text="")

        if brand is None:
            self.result_label.configure(text=confidence, fg="#FF5722")
        else:
            self.result_label.configure(
                text=f"{brand}\n{confidence}", fg="#4CAF50"
            )

    def _show_error(self, error_message):
        self.load_button.configure(state=tk.NORMAL)
        self.result_label.configure(text="Ошибка при обработке", fg="#FF5722")
        messagebox.showerror("Ошибка", f"Произошла ошибка:\n{error_message}")


def main():
    if not os.path.exists(DETECTOR_PATH):
        messagebox.showerror(
            "Ошибка",
            f"Модель детектора не найдена:\n{DETECTOR_PATH}\n\n"
            "Поместите .pt файл в директорию models/",
        )
        sys.exit(1)
    if not os.path.exists(CLASSIFIER_PATH):
        messagebox.showerror(
            "Ошибка",
            f"Модель классификатора не найдена:\n{CLASSIFIER_PATH}\n\n"
            "Поместите .pth файл в директорию models/",
        )
        sys.exit(1)

    root = tk.Tk()

    try:
        print("Загрузка моделей...")
        recognizer = CarBrandRecognizer(DETECTOR_PATH, CLASSIFIER_PATH)
        print("Модели загружены успешно!")
        app = CarBrandApp(root, recognizer)
        root.mainloop()
    except Exception as e:
        messagebox.showerror(
            "Ошибка инициализации",
            f"Не удалось загрузить модели:\n{str(e)}\n\n"
            "Проверьте, что файлы моделей находятся в директории models/",
        )
        root.destroy()


if __name__ == "__main__":
    main()
