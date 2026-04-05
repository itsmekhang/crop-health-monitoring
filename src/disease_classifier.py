"""
Image-based crop disease classification using a ResNet-based CNN.
Dataset: PlantVillage (via TensorFlow Datasets or Kaggle)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50V2


IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def build_model(num_classes: int) -> Model:
    base = ResNet50V2(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    base.trainable = False  # freeze for transfer learning

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)


def get_data_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])


def load_dataset(data_dir: str):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    return train_ds, val_ds


def train(data_dir: str, epochs: int = 10, save_path: str = "results/disease_model.keras"):
    train_ds, val_ds = load_dataset(data_dir)
    num_classes = len(train_ds.class_names)

    augment = get_data_augmentation()
    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))

    model = build_model(num_classes)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    return model


if __name__ == "__main__":
    train(data_dir="data/raw/plantvillage")
