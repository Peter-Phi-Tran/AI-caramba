import cv2
import numpy as np
import tensorflow as tf
import json

# -----------------------
# 1. Load Pretrained Model
# -----------------------
# MobileNetV2 pretrained on ImageNet (1000 classes)
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Preprocessing function for MobileNet
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
decode_preds = tf.keras.applications.mobilenet_v2.decode_predictions

# -----------------------
# 2. Dummy Plant Knowledge DB
# -----------------------
plant_db = {
    "dandelion": {
        "edible_parts": ["leaves", "roots", "flowers"],
        "uses": ["salads", "tea", "wound healing"],
        "cautions": ["sap may irritate skin"]
    },
    "oak_tree": {
        "edible_parts": ["acorns (processed)"],
        "uses": ["flour", "survival food"],
        "cautions": ["raw acorns are toxic (must be leached)"]
    }
}

# -----------------------
# 3. Camera Capture
# -----------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show camera feed
    cv2.imshow("Camera - Plant Identifier", frame)

    # Preprocess frame for MobileNet
    img = cv2.resize(frame, (224, 224))
    x = np.expand_dims(img, axis=0)
    x = preprocess(x)

    # Run inference
    preds = model.predict(x)
    decoded = decode_preds(preds, top=3)[0]

    # Get best guess
    label, description, confidence = decoded[0]

    # Look up plant info if in DB
    plant_info = plant_db.get(description.lower(), None)

    if plant_info:
        print(f"\n🌱 Plant: {description} ({confidence:.2f})")
        print(f"   Edible parts: {plant_info['edible_parts']}")
        print(f"   Uses: {plant_info['uses']}")
        print(f"   Cautions: {plant_info['cautions']}")
    else:
        print(f"\n🤔 Detected: {description} ({confidence:.2f}) - not in survival DB")

    # Quit if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
