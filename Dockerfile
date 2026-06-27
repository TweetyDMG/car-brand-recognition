FROM python:3.11-slim

WORKDIR /app

# Системные зависимости для OpenCV и GUI
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libxcb-xinerama0 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY src/ src/
COPY *.py .
COPY .env.example .env

# Точка входа — по умолчанию inference (без GUI)
ENTRYPOINT ["python", "inference.py"]
