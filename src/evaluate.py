import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
from pathlib import Path

# --------------------------
# Auto-configure paths (MUST MATCH TRAINING)
# --------------------------
def get_project_root():
    current = Path(__file__).parent
    while True:
        if (current / "data").exists() or (current / ".git").exists():
            return current
        if current == current.parent:
            raise FileNotFoundError("Project root not found!")
        current = current.parent

# --------------------------
# Dataset Loading (MUST MATCH TRAINING EXACTLY)
# --------------------------
def load_test_data():
    """Load test data using the SAME method as training"""
    PROJECT_ROOT = get_project_root()
    test_file = PROJECT_ROOT / "data" / "partitions" / "test.txt"
    
    with open(test_file, "r") as f:
        test_files = [line.strip().split("\t") for line in f]
    
    images = []
    labels = []
    for img_path, label in test_files:
        full_img_path = PROJECT_ROOT / img_path
        img = tf.io.read_file(str(full_img_path))
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [128, 128])  # MUST MATCH TRAINING SIZE
        img = tf.image.convert_image_dtype(img, tf.float32)
        images.append(img)
        labels.append(int(label))
    
    test_images = tf.stack(images)
    test_labels = tf.one_hot(labels, depth=4)
    return test_images, test_labels, test_files

# --------------------------
# Comprehensive Evaluation
# --------------------------
def comprehensive_evaluation(model, test_images, test_labels, test_files):
    """Complete evaluation that will match training accuracy"""
    # Get predictions
    PROJECT_ROOT = Path(__file__).parent.parent  # ADD THIS LINE
    y_probs = model.predict(test_images, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = np.argmax(test_labels.numpy(), axis=1)
    
    print("🔬 COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 60)
    print(f"📊 Test Set Size: {len(y_true)} samples")
    
    # Show class distribution
    print("\n📈 Test Set Class Distribution:")
    for i, class_name in enumerate(["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]):
        count = np.sum(y_true == i)
        print(f"   {class_name}: {count} samples ({count/len(y_true)*100:.1f}%)")
    
    # 1. Basic metrics (MUST MATCH TRAINING)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"\n✅ Overall Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"✅ Loss: {test_loss:.4f}")
    
    # 2. Detailed classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred,
          target_names=["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"],
          digits=4))
    
    # 3. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"],
                yticklabels=["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"])
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save outputs
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()
    print("✅ Confusion matrix saved to outputs/confusion_matrix.png")
    
    # 4. Prediction analysis
    print("\n🎯 Prediction Distribution:")
    unique, counts = np.unique(y_pred, return_counts=True)
    for class_idx, count in zip(unique, counts):
        class_name = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"][class_idx]
        print(f"   {class_name}: {count} predictions ({count/len(y_pred)*100:.1f}%)")
    
    # 5. Class-wise accuracy
    print("\n🎯 Class-wise Performance:")
    for i, class_name in enumerate(["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]):
        class_mask = y_true == i
        if np.any(class_mask):
            class_accuracy = np.mean(y_pred[class_mask] == i)
            print(f"   {class_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    
    return test_acc, test_loss

# --------------------------
# Load Training History (If available)
# --------------------------
def load_training_history():
    """Load training history to show progression"""
    PROJECT_ROOT = get_project_root()
    history_path = PROJECT_ROOT / "training_history1.csv"
    
    if history_path.exists():
        history = pd.read_csv(history_path)
        print("\n📈 Training Progress:")
        for i, (round_num, acc, loss) in enumerate(zip(history['round'], history['accuracy'], history['loss'])):
            print(f"   Round {round_num}: Accuracy={acc:.4f}, Loss={loss:.4f}")
        
        # Plot training progress
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['round'], history['accuracy'], marker='o', linewidth=2)
        plt.title('Accuracy per Communication Round')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['round'], history['loss'], marker='s', color='red', linewidth=2)
        plt.title('Loss per Communication Round')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_dir = PROJECT_ROOT / "outputs"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'training_progress1.png')
        plt.close()
        print("✅ Training progress plot saved to outputs/training_progress1.png")

# --------------------------
# Main Evaluation
# --------------------------
def main():
    PROJECT_ROOT = get_project_root()
    
    # Load model (MUST BE THE SAME AS TRAINING)
    model_path = PROJECT_ROOT / "models" / "federated_covid_model_proven.h5"
    print(f"Loading model from: {model_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = keras.models.load_model(model_path)
    
    # Load test data (EXACT SAME AS TRAINING)
    test_images, test_labels, test_files = load_test_data()
    print(f"Loaded test data: {test_images.shape[0]} samples")
    
    # Comprehensive evaluation
    accuracy, loss = comprehensive_evaluation(model, test_images, test_labels, test_files)
    
    # Load training history if available
    load_training_history()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Final Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Final Loss: {loss:.4f}")
    
    if accuracy >= 0.8:
        print("✅ TARGET ACHIEVED: Accuracy >=80%!")
        print("🎉 Your federated learning model is successful!")
    else:
        print("⚠️  Target not reached. Check training-test consistency")
    
    print(f"\n📊 Visualizations saved to: {PROJECT_ROOT / 'outputs'}")

if __name__ == "__main__":
    main()