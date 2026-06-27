# Car Brand Recognition — Распознавание марок автомобилей по изображению

Двухстадийный пайплайн для детекции логотипов автомобилей на фото и их последующей классификации по марке. YOLO-детектор находит область с логотипом, EfficientNet-B0 классификатор относит его к одному из 8 брендов. Система предназначена для CV-инженеров, исследователей и как основа для Auto-ID продуктов.

---

## 🛠 Технологический стек

При проектировании архитектуры приложения упор делался на скорость обработки запросов и модульность.

*   **Язык разработки:** Python 3.10+
*   **Фреймворки и библиотеки:** PyTorch 2.x, TensorFlow 2.x / Keras
*   **Детекция:** Ultralytics YOLOv8 / YOLO11
*   **Классификация:** EfficientNet-B0
*   **Computer Vision:** OpenCV 4.8+, Pillow
*   **GUI:** Tkinter (десктопное приложение)
*   **Контейнеризация и DevOps:** Docker, Docker Compose
*   **Инструменты тестирования:** PyTest

---

## 🚀 Ключевой функционал

Система оцифровывает и автоматизирует следующие бизнес-процессы:

*   **Двухстадийный пайплайн:** обнаружение логотипа (YOLO) → классификация бренда (EfficientNet)
*   **8 поддерживаемых брендов (production):** Hyundai, Lexus, Mazda, Mercedes, Opel, Skoda, Toyota, Volkswagen
*   **Расширяемый список классов:** в архиве экспериментов — до 32 брендов
*   **Десктопный GUI** с загрузкой фото, отрисовкой bounding box и confidence label
*   **CLI-инференс** с argparse — поддержка batch-обработки и сохранения результатов
*   **Fine-tuning** YOLO-детектора и EfficientNet-классификатора на собственном датасете
*   **Поддержка GPU** (CUDA) при наличии

---

## 📁 Архитектура и структура проекта

В проекте используется модульная архитектура с разделением на детекцию и классификацию. Это обеспечивает независимость каждого этапа пайплайна.

```
car-brand-recognition/
├── src/
│   ├── __init__.py              # Пакетная инициализация
│   └── config.py                # Централизованная конфигурация (.env + пути)
├── app_gui.py                   # 🖥️ Tkinter GUI (десктопное приложение)
├── inference.py                 # 🔍 CLI-инференс (argparse)
├── download_datasets.py         # 📦 Скачивание датасетов с Roboflow
├── train_detector.py            # 🎯 Обучение YOLO-детектора
├── train_classifier.py          # 🧠 Обучение EfficientNet-классификатора
├── models/                      # Обученные модели (.pt, .pth) — в .gitignore
├── datasets/                    # Датасеты — в .gitignore
├── runs/                        # Результаты YOLO-обучения — в .gitignore
├── test_images/                 # Тестовые изображения — в .gitignore
├── results/                     # Результаты инференса — в .gitignore
├── Dockerfile                   # 🐳 Контейнеризация
├── docker-compose.yml           # 🐳 Docker Compose (с пробросом GPU)
├── .env.example                 # 🔐 Шаблон переменных окружения
├── .gitignore
├── pyproject.toml               # 📦 Пакетная структура
├── requirements.txt
└── README.md                    # Текущая документация
```

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

---

## 💻 Локальное развертывание

Для запуска проекта в изолированном окружении вам понадобятся **Docker** и **Docker Compose**, либо Python 3.10+ и pip.

### 1. Клонирование репозитория

```bash
git clone https://github.com/<ваш-username>/car-brand-recognition.git
cd car-brand-recognition
```

### 2. Настройка переменных окружения

Создайте файл `.env` в корневой директории проекта по образцу `.env.example`:

```env
ROBOFLOW_API_KEY=your_key_here
```

### 3. Запуск через Docker (рекомендуется)

```bash
docker compose build
docker compose up
# Или CLI-инференс:
docker compose run --rm car-brand-recognition \
  python inference.py --image test_images/img_1.png --output results/out.jpg
```

### 4. Нативная установка

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

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
```

---

## 👥 Разработчики

* [**Артем Рогачев**](https://github.com/TweetyDMG) — Backend Developer

## 📜 Лицензия

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Проект распространяется на условиях лицензии **MIT**. Полный текст лицензии находится в файле [LICENSE](./LICENSE).
