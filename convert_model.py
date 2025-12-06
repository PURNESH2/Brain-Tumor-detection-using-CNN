import tensorflow as tf
import os

print("Converting model weights to .keras format...")

# Path to the best weights saved during training
model_path = 'model/best_model.weights.h5'

if not os.path.exists(model_path):
    print(f"✗ Model weights not found at {model_path}")
    exit()

try:
    # Load the model with weights
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("✓ Model loaded")
    
    # Save as .keras format
    keras_path = 'model/brain_tumor_classifier.keras'
    print(f"Saving to {keras_path}...")
    model.save(keras_path)
    
    print(f"✓ Model saved successfully to {keras_path}")
    
except Exception as e:
    print(f"✗ Error: {str(e)}")