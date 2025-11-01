import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# -------------------------------------------------------------
# Load trained model
# -------------------------------------------------------------
model = tf.keras.models.load_model('soil_model.keras')

# -------------------------------------------------------------
# Class labels (must match the folder names used during training)
# -------------------------------------------------------------
class_labels = ['Alluvial', 'Black', 'Cinder', 'Clay', 'Laterite', 'Peat', 'Red', 'Yellow']

# -------------------------------------------------------------
# Load a test image
# -------------------------------------------------------------
# 👉 Change this to any image from your dataset
img_path = 'Dataset/images/Cinder/29.jpg'

# Preprocess the image
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

# -------------------------------------------------------------
# Predict
# -------------------------------------------------------------
pred = model.predict(x)
predicted_index = np.argmax(pred)
predicted_label = class_labels[predicted_index]
confidence = pred[0][predicted_index] * 100

# -------------------------------------------------------------
# Display Results
# -------------------------------------------------------------
print(f"\n🧠 Predicted Soil Type: {predicted_label}")
print(f"📊 Confidence: {confidence:.2f}%\n")

# (Optional) Show probabilities for all classes
print("Class probabilities:")
for label, prob in zip(class_labels, pred[0]):
    print(f"{label:<10}: {prob*100:.2f}%")
