import tensorflow as tf
import cv2
import numpy as np
import os

# Load model
print("Loading model...")
model = tf.keras.models.load_model('model/brain_tumor_classifier.keras')
print("✓ Model loaded\n")

# Test on a few images
test_images = [
    r"E:\Collage\CNN Brain Tumor\dataset\test\test_9.jpg",
    r"E:\Collage\CNN Brain Tumor\dataset\test\test_4.jpg",
    r"E:\Collage\CNN Brain Tumor\dataset\test\test_20.jpg",
    r"E:\Collage\CNN Brain Tumor\dataset\test\test_19.jpg",
    r"E:\Collage\CNN Brain Tumor\dataset\test\test_15.jpg",
    r"E:\Collage\CNN Brain Tumor\dataset\test\test_16.jpg",
    r"E:\Collage\CNN Brain Tumor\dataset\test\test_17.jpg",
    r"E:\Collage\CNN Brain Tumor\dataset\test\test_18.jpg",
]
   

print("="*70)
print("TESTING MODEL ON NEW IMAGES")
print("="*70 + "\n")

for i, image_path in enumerate(test_images, 1):
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"✗ Could not load image: {image_path}\n")
            continue
        
        # Preprocess
        img = cv2.resize(img, (224, 224)).astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        prob = model.predict(img, verbose=0)[0][0]
        prediction = "TUMOR DETECTED ⚠️" if prob > 0.5 else "NORMAL ✓"
        
        print(f"[Test {i}] {os.path.basename(image_path)}")
        print(f"  Prediction: {prediction}")
        print(f"  Probability: {prob:.4f}")
        print(f"  Confidence: {max(prob, 1-prob):.2%}\n")
    
    except Exception as e:
        print(f"✗ Error: {e}\n")

print("="*70)