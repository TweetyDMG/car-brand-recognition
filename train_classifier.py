#!/usr/bin/env python3
"""
Обучение классификатора брендов автомобилей (EfficientNet-B0 + TensorFlow/Keras).

Конвейер:
  1. Transfer Learning — база EfficientNetB0 (ImageNet) заморожена
  2. Progressive Unfreezing — разморозка на эпохе UNFREEZE_EPOCH
  3. Data Augmentation — rotation, zoom, shift, flip
  4. Callbacks — ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

Датасет:
    datasets/car_brands_classification/
    ├── train/  (352 изображений, 8 классов)
    ├── valid/  (64 изображений)
    └── test/   (128 изображений)

Использование:
    python train_classifier.py
"""

import os
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.config import (
    PROJECT_ROOT,
    CLASSIFICATION_DATASET_DIR,
    CLASSIFIER_IMG_SIZE,
    CLASSIFIER_BATCH_SIZE,
    CLASSIFIER_LEARNING_RATE,
    CLASSIFIER_EPOCHS,
    CLASSIFIER_UNFREEZE_EPOCH,
    CLASSIFIER_CONF_THRESHOLD,
)


class UnfreezeCallback(Callback):
    """
    Размораживает base_model на указанной эпохе и снижает learning rate.

    Позволяет сначала обучить голову классификатора,
    затем донастроить всю сеть с меньшей скоростью.
    """

    def __init__(self, base_model, unfreeze_epoch, new_lr):
        super().__init__()
        self.base_model = base_model
        self.unfreeze_epoch = unfreeze_epoch
        self.new_lr = new_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.unfreeze_epoch:
            print("\n>>> Размораживаем EfficientNetB0")
            self.base_model.trainable = True
            try:
                self.model.optimizer.learning_rate.assign(self.new_lr)
            except AttributeError:
                self.model.optimizer.lr.assign(self.new_lr)
            print(f">>> Новый learning rate: {self.new_lr}")


def build_model(num_classes: int):
    """
    Построить модель: EfficientNetB0 (заморожен) → GAP → Dropout → Dense.

    Returns:
        (model, base_model) — model для обучения, base_model для разморозки.
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=CLASSIFIER_IMG_SIZE + (3,),
    )
    base_model.trainable = False
    print("Базовая модель EfficientNet заморожена.")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CLASSIFIER_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model, base_model


def load_data():
    """
    Загрузить train/valid/test через ImageDataGenerator.

    Классы определяются автоматически по именам поддиректорий.
    """
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    test_aug = ImageDataGenerator(rescale=1.0 / 255.0)

    train_dir = str(CLASSIFICATION_DATASET_DIR / "train")
    val_dir = str(CLASSIFICATION_DATASET_DIR / "valid")
    test_dir = str(CLASSIFICATION_DATASET_DIR / "test")

    print(f"Train: {train_dir}")
    print(f"Valid: {val_dir}")
    print(f"Test:  {test_dir}")

    train = train_aug.flow_from_directory(
        train_dir,
        target_size=CLASSIFIER_IMG_SIZE,
        batch_size=CLASSIFIER_BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )
    val = test_aug.flow_from_directory(
        val_dir,
        target_size=CLASSIFIER_IMG_SIZE,
        batch_size=CLASSIFIER_BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    test = test_aug.flow_from_directory(
        test_dir,
        target_size=CLASSIFIER_IMG_SIZE,
        batch_size=CLASSIFIER_BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return train, val, test


def train_classifier():
    """Полный цикл обучения: загрузка → построение → тренировка → оценка."""
    train, val, test = load_data()
    num_classes = train.num_classes
    print(f"Количество классов: {num_classes}")
    print(f"Имена классов: {train.class_indices}")

    model, base_model = build_model(num_classes)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = PROJECT_ROOT / "models" / "classifier" / f"car_brand_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nМодель будет сохранена в: {save_dir}")

    callbacks = [
        ModelCheckpoint(
            filepath=str(save_dir / "best_model.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            verbose=1,
        ),
        UnfreezeCallback(
            base_model=base_model,
            unfreeze_epoch=CLASSIFIER_UNFREEZE_EPOCH,
            new_lr=CLASSIFIER_LEARNING_RATE * 0.1,
        ),
    ]

    print("\nНачинается обучение...")
    history = model.fit(
        train,
        validation_data=val,
        epochs=CLASSIFIER_EPOCHS,
        callbacks=callbacks,
    )

    print("\nОценка на тестовом наборе:")
    test_loss, test_acc = model.evaluate(test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    final_path = save_dir / "final_model.keras"
    model.save(str(final_path))
    print(f"\nФинальная модель сохранена: {final_path}")

    return model, history


if __name__ == "__main__":
    train_classifier()
