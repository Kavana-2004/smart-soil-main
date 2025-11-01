import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------
base_dir = os.path.join("Dataset", "images")
model_save_path = "soil_model.keras"

# -------------------------------------------------------------
# Data Preprocessing
# -------------------------------------------------------------
# The ImageDataGenerator automatically labels images based on folder names
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# -------------------------------------------------------------
# Model Architecture
# -------------------------------------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

# -------------------------------------------------------------
# Compile the Model
# -------------------------------------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------------------------------------
# Train the Model
# -------------------------------------------------------------
history = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen
)

# -------------------------------------------------------------
# Evaluate and Save
# -------------------------------------------------------------
print("\n✅ Model training complete!")

val_loss, val_acc = model.evaluate(val_gen)
print(f"Validation Accuracy: {val_acc*100:.2f}%")

# Save model
model.save(model_save_path)
print(f"✅ Model saved at: {model_save_path}")

# -------------------------------------------------------------
# (Optional) Plot Training Results
# -------------------------------------------------------------
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
