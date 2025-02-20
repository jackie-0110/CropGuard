import tensorflow as tf
from tensorflow.keras import layers, models

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure GPU is detected
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress warnings

# Load preprocessed PlantVillage dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage/",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)
num_classes = len(train_ds.class_names)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'PlantVillage/',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)
# Use pre-trained MobileNetV2 + fine-tuning
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze layers for transfer learning

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train for quick demo (5-10 epochs)
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Save the model
model.save('crop_disease_model.h5')