import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tqdm import tqdm

# ====================== CONFIGURATION ======================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Model locations (priority order)
MODEL_LOCATIONS = [
    os.path.join(PROJECT_ROOT, 'models', 'federated_covid_model.h5'),
    os.path.join(PROJECT_ROOT, 'federated_covid_model.h5'),
    r'C:\Users\laiba\projects\fl-covid-diagnosis\models\federated_covid_model.h5'
]

# Data locations (priority order)
DATA_LOCATIONS = [
    os.path.join(PROJECT_ROOT, 'data'),
    os.path.join(PROJECT_ROOT, 'FL-COVID-DIAGNOSIS', 'data')
]

CLASSES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
IMG_SIZE = (224, 224)
MAX_SAMPLES = 1000  # Safety limit
TTA_NUM = 5  # Number of test-time augmentations

# ====================== TTA GENERATOR ======================
def create_tta_generator():
    """Create generator for test-time augmentations"""
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

# ====================== TTA PREDICTION ======================
def predict_with_tta(model, img_path, tta_generator, n_aug=TTA_NUM):
    """Make predictions using test-time augmentation"""
    try:
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        preds = []
        for _ in range(n_aug):
            aug_img = tta_generator.random_transform(img_array[0])
            preds.append(model.predict(np.expand_dims(aug_img, axis=0), verbose=0))
        
        avg_pred = np.mean(preds, axis=0)
        pred_class = CLASSES[np.argmax(avg_pred)]
        confidence = float(np.max(avg_pred))
        
        return pred_class, confidence
    
    except Exception as e:
        raise ValueError(f"TTA failed for {img_path}: {str(e)}")

# ====================== PATH RESOLUTION ======================
def find_resource(locations, resource_type):
    """Find the first existing resource from possible locations"""
    for path in locations:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            print(f"Found {resource_type} at: {abs_path}")
            return abs_path
    print(f"\nERROR: Could not find {resource_type}. Checked:")
    for path in locations:
        print(f"- {os.path.abspath(path)}")
    return None

# ====================== DATA LOADING ======================
def load_test_data(partitions_dir, image_dir):
    """Safely load test data with thorough validation"""
    test_file = os.path.join(partitions_dir, 'test.txt')
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test partition file not found at {test_file}")

    # Read test file with error handling
    try:
        with open(test_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        raise IOError(f"Failed to read test file: {str(e)}")

    if not lines:
        raise ValueError("Test file is empty")

    data = []
    missing_files = []
    duplicate_files = set()
    processed_count = 0

    for line in lines:
        if processed_count >= MAX_SAMPLES:
            print(f"\nWarning: Reached maximum sample limit of {MAX_SAMPLES}")
            break

        # Clean and validate the line
        parts = line.split('\t')
        if not parts:
            continue

        img_path = parts[0].strip()
        if not img_path:
            continue

        # Check for duplicates
        if img_path in duplicate_files:
            continue
        duplicate_files.add(img_path)

        # Find the image in class subdirectories
        found = False
        for class_name in CLASSES:
            class_image_dir = os.path.join(image_dir, class_name, 'images')
            potential_path = os.path.join(class_image_dir, os.path.basename(img_path))
            
            if os.path.exists(potential_path):
                data.append({
                    'filename': potential_path,
                    'class': class_name
                })
                found = True
                processed_count += 1
                break

        if not found:
            missing_files.append(img_path)

    # Provide feedback about missing files
    if missing_files:
        print(f"\nWarning: Could not find {len(missing_files)} image files")
        if len(missing_files) > 5:
            print("Examples:")
            for f in missing_files[:5]:
                print(f"- {f}")
            print(f"... and {len(missing_files)-5} more")
        else:
            for f in missing_files:
                print(f"- {f}")

    if not data:
        raise ValueError("No valid test images found")

    return pd.DataFrame(data)

# ====================== PREDICTION ======================
def predict_images(model, test_df, use_tta=True):
    """Safe prediction with progress tracking and optional TTA"""
    results = []
    successful = 0
    failed = 0
    
    tta_generator = create_tta_generator() if use_tta else None

    print(f"\nMaking predictions ({'with TTA' if use_tta else 'without TTA'})...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
        try:
            if use_tta:
                pred_class, confidence = predict_with_tta(model, row['filename'], tta_generator)
            else:
                img = load_img(row['filename'], target_size=IMG_SIZE)
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                preds = model.predict(img_array, verbose=0)
                pred_class = CLASSES[np.argmax(preds)]
                confidence = float(np.max(preds))
            
            results.append({
                'image': os.path.basename(row['filename']),
                'true_class': row['class'],
                'predicted_class': pred_class,
                'confidence': round(confidence, 4),
                'correct': row['class'] == pred_class,
                'method': 'TTA' if use_tta else 'standard'
            })
            successful += 1
        except Exception as e:
            print(f"\nError processing {row['filename']}: {str(e)}")
            failed += 1
            continue

    print(f"\nCompleted: {successful} successful, {failed} failed")
    return pd.DataFrame(results)

# ====================== MAIN EXECUTION ======================
def main():
    print("=== COVID-19 Diagnosis Prediction with TTA ===")
    
    try:
        # 1. Resolve all paths
        model_path = find_resource(MODEL_LOCATIONS, "model file")
        data_root = find_resource(DATA_LOCATIONS, "data directory")
        
        if not model_path or not data_root:
            sys.exit(1)

        image_dir = os.path.join(data_root, 'Covid19 Radiography Database')
        partitions_dir = os.path.join(data_root, 'partitions')

        # 2. Validate directories
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(partitions_dir):
            raise FileNotFoundError(f"Partitions directory not found: {partitions_dir}")

        # 3. Load model
        print("\nLoading model...")
        model = load_model(model_path)
        print("Model loaded successfully!")

        # 4. Load test data
        print("\nLoading test data...")
        test_df = load_test_data(partitions_dir, image_dir)
        print(f"Found {len(test_df)} valid test images")

        # 5. Make predictions (with TTA)
        results_df = predict_images(model, test_df, use_tta=True)

        # 6. Save and display results
        if not results_df.empty:
            print("\nPrediction results (sample):")
            print(results_df.head())
            
            results_file = os.path.join(PROJECT_ROOT, 'prediction_results_with_tta.csv')
            results_df.to_csv(results_file, index=False)
            print(f"\nResults saved to: {results_file}")
            
            accuracy = results_df['correct'].mean() * 100
            print(f"\nModel Accuracy (with TTA): {accuracy:.2f}%")
            
            # Compare with standard predictions if needed
            if len(results_df) > 100:  # Only compare for larger datasets
                std_results = predict_images(model, test_df.sample(100), use_tta=False)
                std_accuracy = std_results['correct'].mean() * 100
                print(f"Standard Accuracy (subset): {std_accuracy:.2f}%")
                print(f"TTA Improvement: {accuracy - std_accuracy:.2f}%")
        else:
            print("\nNo successful predictions were made")

    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()