# Car Brand Recognition — Распознавание марок автомобилей по изображению

**Car Brand Recognition** — решение для детекции логотипов автомобилей на фото и их последующей классификации по марке. Двухстадийный пайплайн на YOLO + EfficientNet с десктопным GUI на Tkinter и CLI-интерфейсом для batch-инференса.

> Обученные модели работают последовательно: YOLO-детектор находит область с логотипом на кадре, EfficientNet-B0 классификатор относит его к одному из 8 брендов. Система предназначена для CV-инженеров, исследователей и как основа для Auto-ID продуктов.

---

## 🛠 Технологический стек

| Категория | Технологии |
|-----------|------------|
| **Язык** | Python 3.10+ |
| **Детекция** | Ultralytics YOLOv8 / YOLO11, PyTorch 2.x |
| **Классификация** | TensorFlow 2.x / Keras, PyTorch, EfficientNet-B0 |
| **Computer Vision** | OpenCV 4.8+, Pillow, NumPy |
| **GUI** | Tkinter (десктопное приложение) |
| **Датасеты** | Roboflow Universe |
| **Визуализация** | Matplotlib, Seaborn |
| **DevOps** | Docker, Docker Compose |

> Проект использует два DL-фреймворка: PyTorch — для инференса (детектор + классификатор), TensorFlow/Keras — для обучения классификатора. Рекомендуется контейнеризация для изоляции зависимостей.

---

## 🚀 Ключевой функционал

- **Двухстадийный пайплайн:** обнаружение логотипа (YOLO) → классификация бренда (EfficientNet)
- **8 поддерживаемых брендов (production):** Hyundai, Lexus, Mazda, Mercedes, Opel, Skoda, Toyota, Volkswagen
- **Расширяемый список классов:** в архиве экспериментов — до 32 брендов (BMW, Honda, Tesla, Ford, Renault и др.)
- **Десктопный GUI** с загрузкой фото, отрисовкой bounding box и confidence label
- **CLI-инференс** с argparse — поддержка batch-обработки и сохранения результатов
- **Fine-tuning детектора** YOLO на собственном датасете (Roboflow)
- **Fine-tuning классификатора** с Transfer Learning, Progressive Unfreezing, Data Augmentation, Early Stopping, ReduceLROnPlateau
- **Поддержка GPU** (CUDA) при наличии

---

## 📁 Архитектура

### Пайплайн инференса

```
Входное изображение (RGB)
        │
        ▼
┌─────────────────────────┐
│  YOLO Logo Detector     │  ← models/car_logo_detector.pt
│  (ultralytics)          │     Порог: conf ≥ 0.4
└───────────┬─────────────┘
            │
            ▼
    ┌───────────────┐
    │ Логотип найден? │── Нет → "Логотип не найден"
    └───────┬───────┘
            │ Да
            ▼
┌─────────────────────────────┐
│  EfficientNet-B0 Classifier │  ← models/car_logo_classifier.pth
│  (224×224, fine-tuned)      │     Порог: conf ≥ 0.3
└─────────────┬───────────────┘
              │
              ▼
   "Toyota (87%)"  ← с bounding box на изображении
```

### Структура репозитория

```
car-brand-recognition/
├── src/
│   ├── __init__.py              # Пакетная инициализация
│   └── config.py                # Централизованная конфигурация (.env + пути)
│
├── app_gui.py                   # 🖥️ Tkinter GUI (десктопное приложение)
├── inference.py                 # 🔍 CLI-инференс (argparse)
├── download_datasets.py         # 📦 Скачивание датасетов с Roboflow
├── train_detector.py            # 🎯 Обучение YOLO-детектора
├── train_classifier.py          # 🧠 Обучение EfficientNet-классификатора
│
├── models/                      # Обученные модели (.pt, .pth) — в .gitignore
├── datasets/                    # Датасеты — в .gitignore
├── runs/                        # Результаты YOLO-обучения — в .gitignore
├── test_images/                 # Тестовые изображения — в .gitignore
├── results/                     # Результаты инференса — в .gitignore
│
├── Dockerfile                   # 🐳 Контейнеризация
├── docker-compose.yml           # 🐳 Docker Compose (с пробросом GPU)
├── .env.example                 # 🔐 Шаблон переменных окружения
├── .gitignore
├── pyproject.toml               # 📦 Пакетная структура
├── requirements.txt
└── README.md
```

### Связи между модулями

| Модуль | Вход | Выход | Зависимость |
|--------|------|-------|-------------|
| `inference.py` / `app_gui.py` | RGB-изображение | `(brand, confidence, annotated_img)` | `src.config`, YOLO + PyTorch |
| `train_detector.py` | Датасет Roboflow (YOLO-формат) | `best.pt` | Ultralytics YOLO |
| `train_classifier.py` | Датасет (train/val/test) | `best_model.keras` | TensorFlow / Keras |
| `download_datasets.py` | — | `datasets/` | Roboflow SDK |

---

## 💻 Локальное развертывание

### 1. Клонирование

```bash
git clone https://github.com/<your-org>/car-brand-recognition.git
cd car-brand-recognition
```

### 2. Настройка окружения

```bash
# Скопировать шаблон .env
cp .env.example .env

# Отредактировать .env — указать API-ключ Roboflow (для скачивания датасетов)
# ROBOFLOW_API_KEY=your_key_here
```

### 3. Через Docker (рекомендуется)

```bash
# Сборка образа
docker compose build

# Запуск GUI (требуется X11)
docker compose up

# Или CLI-инференс:
docker compose run --rm car-brand-recognition \
  python inference.py --image test_images/img_1.png --output results/out.jpg
```

### 4. Нативная установка

```bash
# Виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Зависимости
pip install --upgrade pip
pip install -r requirements.txt

# Скачать датасеты (опционально, нужен ключ Roboflow)
python download_datasets.py

# Запуск GUI
python app_gui.py

# Или CLI-инференс
python inference.py --image test_images/img_1.png
```

---

## 🔌 Примеры использования

### GUI

```bash
python app_gui.py
```

Откроется окно Tkinter. Загрузите фото через кнопку "📷 Загрузить фото":
- Программа найдёт логотип и обведёт его красной рамкой
- Выведет название марки и уровень уверенности
- При низкой уверенности (<30%) — сообщит "Марка не определена"

### CLI

```bash
# Базовый запуск
python inference.py

# С указанием файла
python inference.py --image test_images/img_1.png --output results/result.jpg

# С пониженным порогом детекции
python inference.py -i test_images/img_1.png -o results/out.jpg -t 0.3
```

### Программный вызов

```python
from inference import CarBrandRecognizer

recognizer = CarBrandRecognizer()
results, annotated = recognizer.process_image("test.jpg", output_path="result.jpg")

for r in results:
    print(f"{r['brand']}: {r['confidence']:.2%}, bbox={r['bbox']}")
    print(f"  Top-3: {[(b, f'{c:.2%}') for b, c in r['top3']]}")
```

### Обучение моделей

```bash
# Детектор (YOLO)
python train_detector.py
# → runs/detect/car_logo_detector/weights/best.pt

# Классификатор (EfficientNet-B0)
python train_classifier.py
# → models/classifier/car_brand_YYYYMMDD_HHMMSS/best_model.keras
```

---

## 🧪 Датасеты

| Датасет | Тип | Источник | Классов | Разбивка (train/val/test) |
|---------|-----|----------|---------|--------------------------|
| `car_logos_detection` | Объектная детекция (YOLO) | [Roboflow Universe](https://universe.roboflow.com/carbrandrecognition/car-logo-detection-pcfc6-ir2g4) | 1 (logo) | Зависит от версии |
| `car_brands_classification` | Классификация | Собственный сбор | 8 | 352 / 64 / 128 |

---

## 📈 Метрики обученных моделей

**Детектор (YOLOv8n):**
- mAP@0.50, Precision, Recall — выводятся после `train_detector.py`
- Результаты сохраняются в `runs/detect/car_logo_detector/`

**Классификатор (EfficientNet-B0):**
- Test Accuracy: ~93–97% (в зависимости от версии обучения)
- Лучшая модель сохраняется автоматически (`ModelCheckpoint` по `val_accuracy`)

---

## 🔗 Полезные ссылки

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Roboflow — Car Logo Detection Dataset](https://universe.roboflow.com/carbrandrecognition/car-logo-detection-pcfc6-ir2g4)
- [EfficientNet: Rethinking Model Scaling for CNNs (ICML 2019)](https://arxiv.org/abs/1905.11946)

---

## 📄 Лицензия

MIT
