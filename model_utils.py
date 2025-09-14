import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

def get_project_root():
    """Get the project root directory."""
    current = Path(__file__).parent
    while True:
        if (current / "data").exists() or (current / ".git").exists():
            return current
        if current == current.parent:
            raise FileNotFoundError("Project root not found!")
        current = current.parent

def load_trained_model():
    """Load the trained model."""
    PROJECT_ROOT = get_project_root()
    model_path = PROJECT_ROOT / "models" / "federated_covid_model_proven.h5"
    print(f"Loading model from: {model_path}")
    return keras.models.load_model(model_path)

def preprocess_image(image_path):
    """Preprocess image exactly like during training."""
    # Load image
    img = tf.io.read_file(str(image_path))
    img = tf.image.decode_jpeg(img, channels=3)
    
    # Resize to match training size (128x128)
    img = tf.image.resize(img, [128, 128])
    
    # Convert to float32 (same as training)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    return img

def implement_test_time_augmentation(model, image, n_augmentations=5):
    """Apply test time augmentation to improve prediction confidence."""
    augmentations = []
    
    # Original image
    augmentations.append(image)
    
    # Horizontal flip
    augmentations.append(tf.image.flip_left_right(image))
    
    # Small rotations
    augmentations.append(tf.image.rot90(image, k=1))
    augmentations.append(tf.image.rot90(image, k=3))
    
    # Brightness adjustments
    augmentations.append(tf.image.adjust_brightness(image, delta=0.1))
    augmentations.append(tf.image.adjust_brightness(image, delta=-0.1))
    
    # Ensure we don't exceed requested number of augmentations
    augmentations = augmentations[:n_augmentations]
    
    # Get predictions for all augmentations
    predictions = []
    for aug_img in augmentations:
        # Add batch dimension
        aug_img = tf.expand_dims(aug_img, axis=0)
        pred = model.predict(aug_img, verbose=0)
        predictions.append(pred)
    
    # Average the predictions
    avg_prediction = np.mean(predictions, axis=0)
    return avg_prediction

def predict_with_confidence(model, image_path, confidence_threshold=0.7):
    """Make prediction with enhanced confidence using TTA."""
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Use test time augmentation
    prediction = implement_test_time_augmentation(model, processed_image)
    
    # Get confidence and class
    confidence = np.max(prediction)
    class_idx = np.argmax(prediction)
    class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
    
    return {
        "class": class_names[class_idx],
        "confidence": float(confidence),
        "is_confident": confidence >= confidence_threshold,
        "all_predictions": prediction.tolist()
    }