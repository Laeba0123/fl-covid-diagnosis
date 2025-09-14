import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (classification_report, 
                            confusion_matrix,
                            roc_auc_score,
                            balanced_accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuration
MODEL_PATH = r'C:\Users\laiba\projects\fl-covid-diagnosis\models\federated_covid_model.h5'
DATA_DIR = r'C:\Users\laiba\projects\fl-covid-diagnosis\data'
TEST_PARTITION = os.path.join(DATA_DIR, 'partitions', 'test.txt')
IMAGE_DIR = os.path.join(DATA_DIR, 'Covid19 Radiography Database')
CLASSES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_test_data():
    """Load and validate test data with comprehensive checks"""
    if not os.path.exists(TEST_PARTITION):
        raise FileNotFoundError(f"Test partition not found at {TEST_PARTITION}")

    with open(TEST_PARTITION, 'r') as f:
        lines = [line.strip().split('\t')[0] for line in f if line.strip()]

    data = []
    missing = []
    for img_path in tqdm(lines, desc="Validating test files"):
        found = False
        for class_name in CLASSES:
            full_path = os.path.join(IMAGE_DIR, class_name, 'images', os.path.basename(img_path))
            if os.path.exists(full_path):
                data.append({'filename': full_path, 'class': class_name})
                found = True
                break
        if not found:
            missing.append(img_path)

    if missing:
        print(f"\nWarning: {len(missing)} images missing. Examples:")
        for f in missing[:3]:
            print(f"- {f}")
        if len(missing) > 3:
            print(f"... and {len(missing)-3} more")

    if not data:
        raise ValueError("No valid test images found")
    
    return pd.DataFrame(data)

def evaluate_model(model, test_df):
    """Comprehensive model evaluation with multiple metrics"""
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filename',
        y_col='class',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # 1. Basic Evaluation
    print("\n[1/4] Running basic evaluation...")
    loss, acc = model.evaluate(test_gen)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")

    # 2. Detailed Metrics
    print("\n[2/4] Calculating detailed metrics...")
    y_true = test_gen.classes
    y_pred = model.predict(test_gen).argmax(axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    
    print("\nBalanced Accuracy:", balanced_accuracy_score(y_true, y_pred))
    
    try:
        y_proba = model.predict(test_gen)
        print("\nROC AUC (OvR):", roc_auc_score(
            tf.keras.utils.to_categorical(y_true),
            y_proba,
            multi_class='ovr'
        ))
    except Exception as e:
        print(f"\nROC AUC calculation skipped: {str(e)}")

    # 3. Confusion Matrix Visualization
    print("\n[3/4] Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")

    # 4. Error Analysis
    print("\n[4/4] Performing error analysis...")
    test_df['prediction'] = [CLASSES[i] for i in y_pred]
    test_df['correct'] = test_df['class'] == test_df['prediction']
    
    errors = test_df[~test_df['correct']]
    if not errors.empty:
        print("\nMost common misclassifications:")
        print(errors.groupby(['class', 'prediction']).size().sort_values(ascending=False).head(10))
        
        error_samples = errors.sample(min(5, len(errors)))
        print("\nSample misclassified images:")
        print(error_samples[['filename', 'class', 'prediction']])
    else:
        print("\nNo misclassifications found!")

def main():
    print("=== COVID-19 Model Benchmark ===")
    
    # 1. Load Model
    try:
        print("\nLoading model...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        model = load_model(MODEL_PATH)
        print(f"Loaded {model.name} with input shape {model.input_shape}")
    except Exception as e:
        print(f"\nModel loading failed: {str(e)}")
        return

    # 2. Load Data
    try:
        print("\nLoading test data...")
        test_df = load_test_data()
        print(f"Found {len(test_df)} valid test images")
        print("\nClass distribution:")
        print(test_df['class'].value_counts())
    except Exception as e:
        print(f"\nData loading failed: {str(e)}")
        return

    # 3. Run Evaluation
    try:
        evaluate_model(model, test_df)
    except Exception as e:
        print(f"\nEvaluation failed: {str(e)}")
        return

    print("\nBenchmark completed successfully!")

if __name__ == '__main__':
    main()