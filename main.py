import os
import sys
import json
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

warnings.filterwarnings('ignore')


DATASET_ROOT = r"E:\Collage\CNN Brain Tumor" # Change this to your project path

# Enter location here: Set path to a test image for inference (optional)
#TEST_IMAGE_PATH = r"C:\Users\YourName\project_folder\dataset\brain\yes\test_image.jpg"

# Model configuration
IMG_SIZE = (224, 224)  # Input image dimensions for MobileNetV2
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4
TRAIN_VAL_SPLIT = 0.8
RANDOM_SEED = 42

# Data augmentation
USE_AUGMENTATION = True

# ============================================================================
# ============================= UTILITY FUNCTIONS ============================
# ============================================================================

def setup_random_seeds(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ Random seeds set to {seed} for reproducibility")


def detect_gpu():
    """Detect and log GPU/CPU availability."""
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    print("\n" + "="*70)
    print("HARDWARE INFORMATION")
    print("="*70)
    
    if gpus:
        print(f"✓ GPU detected: {len(gpus)} GPU(s) available")
        for gpu in gpus:
            print(f"  - {gpu}")
        print("✓ TensorFlow will use GPU for training")
    else:
        print("✗ No GPU detected. TensorFlow will use CPU.")
        print("  (Training will be slower on CPU)")
    
    print(f"✓ CPU(s): {len(cpus)} available")
    print("="*70 + "\n")


def validate_dataset_structure(dataset_root):
    """Validate that the dataset directory structure is correct."""
    print("VALIDATING DATASET STRUCTURE")
    print("-" * 70)
    
    # Check if dataset_root exists
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"✗ Dataset root not found: {dataset_root}")
    print(f"✓ Dataset root found: {dataset_root}")
    
    # Check for brain folder
    brain_dir = os.path.join(dataset_root, "dataset", "brain")
    if not os.path.isdir(brain_dir):
        raise FileNotFoundError(f"✗ Brain folder not found: {brain_dir}")
    print(f"✓ Brain directory found: {brain_dir}")
    
    # Check for yes/no folders
    yes_dir = os.path.join(brain_dir, "yes")
    no_dir = os.path.join(brain_dir, "no")
    
    if not os.path.isdir(yes_dir):
        raise FileNotFoundError(f"✗ 'yes' (tumor) folder not found: {yes_dir}")
    if not os.path.isdir(no_dir):
        raise FileNotFoundError(f"✗ 'no' (no tumor) folder not found: {no_dir}")
    
    print(f"✓ 'yes' (tumor) folder found: {yes_dir}")
    print(f"✓ 'no' (no tumor) folder found: {no_dir}")
    
    # Count images
    yes_count = len([f for f in os.listdir(yes_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    no_count = len([f for f in os.listdir(no_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if yes_count == 0 or no_count == 0:
        raise ValueError(f"✗ Empty folders detected. yes: {yes_count}, no: {no_count}")
    
    print(f"✓ Images found: yes={yes_count}, no={no_count}")
    
    # Check class imbalance
    imbalance_ratio = max(yes_count, no_count) / min(yes_count, no_count)
    if imbalance_ratio > 1.5:
        print(f"⚠ Class imbalance detected (ratio: {imbalance_ratio:.2f}). Class weights will be applied.")
    
    print("-" * 70 + "\n")
    return yes_dir, no_dir, yes_count, no_count


def load_images_from_directory(directory, label, img_size=IMG_SIZE):
    """Load images from a directory and convert to arrays."""
    images = []
    image_names = []
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    files = [f for f in os.listdir(directory) if f.endswith(valid_extensions)]
    
    for filename in tqdm(files, desc=f"Loading {label} images"):
        try:
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"  ⚠ Skipped corrupted image: {filename}")
                continue
            
            # Convert to grayscale if needed, then to RGB (3 channels)
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Resize to target size
            img = cv2.resize(img, img_size)
            
            # Normalize to [0, 1]
            img = img.astype('float32') / 255.0
            
            images.append(img)
            image_names.append(filename)
        
        except Exception as e:
            print(f"  ✗ Error loading {filename}: {str(e)}")
            continue
    
    return np.array(images), image_names


def create_augmentation_pipeline():
    """Create data augmentation pipeline for training."""
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )


def prepare_dataset(yes_dir, no_dir, train_val_split=TRAIN_VAL_SPLIT):
    """Load and split dataset into train and validation sets."""
    print("\nLOADING DATASET")
    print("-" * 70)
    
    # Load images
    yes_images, yes_names = load_images_from_directory(yes_dir, "tumor (yes)")
    no_images, no_names = load_images_from_directory(no_dir, "no tumor (no)")
    
    print(f"✓ Loaded {len(yes_images)} tumor images and {len(no_images)} normal images")
    
    # Create labels
    yes_labels = np.ones(len(yes_images))
    no_labels = np.zeros(len(no_images))
    
    # Combine and shuffle
    X = np.vstack([yes_images, no_images])
    y = np.hstack([yes_labels, no_labels])
    
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split into train and validation
    split_idx = int(len(X) * train_val_split)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"✓ Train set: {len(X_train)} images")
    print(f"✓ Validation set: {len(X_val)} images")
    print("-" * 70 + "\n")
    
    return X_train, X_val, y_train, y_val


def compute_class_weights(y_train):
    """Compute class weights to handle imbalance."""
    classes = np.unique(y_train)
    weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(weights)}
    print(f"✓ Class weights computed: {class_weight_dict}\n")
    return class_weight_dict


def build_model(input_shape=(224, 224, 3)):
    """Build CNN model with MobileNetV2 transfer learning."""
    print("BUILDING MODEL")
    print("-" * 70)
    
    # Load pretrained MobileNetV2
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Build custom top layers
    model = models.Sequential([
        layers.Input(shape=input_shape),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu', name='dense1'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', name='dense2'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid', name='output')  # Binary classification
    ])
    
    print(f"✓ Model built with MobileNetV2 backbone")
    print(f"✓ Total parameters: {model.count_params():,}")
    print("-" * 70 + "\n")
    
    return model


def compile_model(model, learning_rate=LEARNING_RATE):
    """Compile model with optimizer and loss function."""
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("✓ Model compiled with Adam optimizer and binary crossentropy loss\n")


def create_model_directory():
    """Create model directory if it doesn't exist."""
    model_dir = os.path.join(os.getcwd(), "model")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def train_model(model, X_train, X_val, y_train, y_val, 
                class_weights, model_dir, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Train the model with callbacks."""
    print("TRAINING MODEL")
    print("-" * 70)
    
    # Create callbacks
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_dir, "best_model.weights.h5"),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    tensorboard = TensorBoard(
        log_dir=os.path.join(model_dir, "logs"),
        histogram_freq=0
    )
    
    callbacks = [checkpoint, early_stop, reduce_lr, tensorboard]
    
    # Data augmentation (optional)
    if USE_AUGMENTATION:
        datagen = create_augmentation_pipeline()
        print("✓ Data augmentation enabled\n")
    else:
        datagen = None
        print("✓ Data augmentation disabled\n")
    
    # Record training start time
    start_time = time.time()
    
    # Train model
    if datagen:
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            steps_per_epoch=len(X_train) // batch_size,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
    
    training_time = time.time() - start_time
    print(f"\n✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print("-" * 70 + "\n")
    
    return history, training_time


def evaluate_model(model, X_val, y_val):
    """Evaluate model on validation set."""
    print("EVALUATING MODEL")
    print("-" * 70)
    
    # Get predictions
    y_pred_proba = model.predict(X_val, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    y_val_int = y_val.astype(int)
    
    # Compute metrics
    acc = accuracy_score(y_val_int, y_pred)
    prec = precision_score(y_val_int, y_pred, zero_division=0)
    rec = recall_score(y_val_int, y_pred, zero_division=0)
    f1 = f1_score(y_val_int, y_pred, zero_division=0)
    auc = roc_auc_score(y_val_int, y_pred_proba)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print("-" * 70 + "\n")
    
    metrics = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(auc)
    }
    
    return metrics, y_pred, y_pred_proba


def save_training_history(history, model_dir):
    """Save training history to CSV."""
    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(model_dir, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"✓ Training history saved to {history_path}")


def save_confusion_matrix_plot(y_val, y_pred, model_dir):
    """Compute and save confusion matrix plot."""
    cm = confusion_matrix(y_val.astype(int), y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=16)
    plt.colorbar()
    
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['No Tumor', 'Tumor'])
    plt.yticks(tick_marks, ['No Tumor', 'Tumor'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    cm_path = os.path.join(model_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {cm_path}")


def save_metrics_json(metrics, training_time, model_dir):
    """Save evaluation metrics to JSON."""
    metrics['training_time_seconds'] = training_time
    metrics['timestamp'] = datetime.now().isoformat()
    
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"✓ Metrics saved to {metrics_path}")


def save_model(model, model_dir):
    """Save the trained model."""
    model_path = os.path.join(model_dir, "brain_tumor_classifier")
    model.save(model_path)
    print(f"✓ Model saved to {model_path}")
    return model_path


def predict_image(image_path, model_dir="model"):
    """Load saved model and predict on a single image."""
    try:
        print("\n" + "="*70)
        print("INFERENCE MODE")
        print("="*70)
        
        # Check if model exists
        model_path = os.path.join(model_dir, "brain_tumor_classifier.keras")
        model.save(model_path)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"✗ Model not found at {model_path}. Train the model first.")
        
        # Check if image exists
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"✗ Image not found at {image_path}")
        
        print(f"Loading model from {model_path}...")
        model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully\n")
        
        # Load and preprocess image
        print(f"Loading image from {image_path}...")
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError("✗ Could not read image. Ensure it's a valid image file.")
        
        # Convert to 3 channels if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Resize and normalize
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        print(f"✓ Image loaded and preprocessed\n")
        
        # Make prediction
        prediction_proba = model.predict(img, verbose=0)[0][0]
        prediction = "yes" if prediction_proba > 0.5 else "no"
        
        result = {
            "prediction": prediction,
            "probability": float(prediction_proba),
            "confidence": float(max(prediction_proba, 1 - prediction_proba))
        }
        
        print("PREDICTION RESULT:")
        print("-" * 70)
        print(f"Classification: {result['prediction'].upper()} (tumor: {'present' if result['prediction'] == 'yes' else 'not detected'})")
        print(f"Raw Probability: {result['probability']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("="*70 + "\n")
        
        print("⚠ DISCLAIMER: This prediction is for research purposes only.")
        print("Medical AI models require clinical validation before diagnostic use.\n")
        
        return result
    
    except FileNotFoundError as e:
        print(f"✗ Error: {str(e)}\n")
        return None
    except Exception as e:
        print(f"✗ Unexpected error during inference: {str(e)}\n")
        return None


# ============================================================================
# ============================= MAIN EXECUTION ==============================
# ============================================================================

def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("BRAIN MRI TUMOR CLASSIFICATION - CNN TRAINING PIPELINE")
    print("="*70 + "\n")
    
    try:
        # Setup
        setup_random_seeds()
        detect_gpu()
        
        # Validate dataset
        yes_dir, no_dir, yes_count, no_count = validate_dataset_structure(DATASET_ROOT)
        
        # Prepare data
        X_train, X_val, y_train, y_val = prepare_dataset(yes_dir, no_dir)
        
        # Compute class weights
        class_weights = compute_class_weights(y_train)
        
        # Build model
        model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        compile_model(model)
        
        # Create model directory
        model_dir = create_model_directory()
        
        # Train model
        history, training_time = train_model(
            model, X_train, X_val, y_train, y_val,
            class_weights, model_dir, epochs=EPOCHS, batch_size=BATCH_SIZE
        )
        
        # Evaluate model
        metrics, y_pred, y_pred_proba = evaluate_model(model, X_val, y_val)
        
        # Save artifacts
        save_training_history(history, model_dir)
        save_confusion_matrix_plot(y_val, y_pred, model_dir)
        save_metrics_json(metrics, training_time, model_dir)
        save_model(model, model_dir)
        
        print("\n" + "="*70)
        print("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Model and artifacts saved to: {model_dir}")
        print(f"Training time: {training_time:.2f} seconds")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
    
    # ========================================================================
    # OPTIONAL: RUN INFERENCE ON A TEST IMAGE
    # ========================================================================
    # Uncomment the line below to test inference on an image
    # Make sure to set TEST_IMAGE_PATH above first
    
    # predict_image(TEST_IMAGE_PATH)