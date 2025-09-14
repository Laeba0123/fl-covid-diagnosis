import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from pathlib import Path

# Set style for professional plots
plt.style.use('default')
sns.set_palette("colorblind")

def get_project_root():
    """Get the project root directory."""
    current = Path(__file__).parent
    while True:
        if (current / "data").exists() or (current / ".git").exists():
            return current
        if current == current.parent:
            raise FileNotFoundError("Project root not found!")
        current = current.parent

def load_data_and_model():
    """Load the test data and the trained model. MUST BE IDENTICAL TO evaluate.py."""
    PROJECT_ROOT = get_project_root()
    
    # 1. Load the correct, proven model
    model_path = PROJECT_ROOT / "models" / "federated_covid_model_proven.h5"
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    # 2. Load test data THE EXACT SAME WAY as training/evaluation
    test_file = PROJECT_ROOT / "data" / "partitions" / "test.txt"
    with open(test_file, "r") as f:
        test_files = [line.strip().split("\t") for line in f]
    
    images = []
    labels = []
    for img_path, label in test_files:
        full_img_path = PROJECT_ROOT / img_path
        img = tf.io.read_file(str(full_img_path))
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [128, 128])  # CRITICAL: Must match training size (128, 128)
        img = tf.image.convert_image_dtype(img, tf.float32)
        images.append(img)
        labels.append(int(label))
    
    test_images = tf.stack(images)
    test_labels = tf.one_hot(labels, depth=4) # CRITICAL: depth must match classes (4)
    
    return model, test_images, test_labels, test_files

def create_comprehensive_visualizations(model, test_images, test_labels):
    """Generate accurate and professional visualizations."""
    PROJECT_ROOT = get_project_root()
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Get predictions and true labels
    y_probs = model.predict(test_images, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = np.argmax(test_labels.numpy(), axis=1)
    class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
    
    # Calculate overall accuracy (should be ~81.57%)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"✅ Model Accuracy for Visualization: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # --- Figure 1: Confusion Matrix ---
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    # Normalize the matrix to show percentages, which are easier to interpret
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                cbar_kws={'label': 'Percentage of True Class'},
                xticklabels=class_names, 
                yticklabels=class_names)
    
    plt.title(f'Normalized Confusion Matrix\n(Overall Accuracy: {test_acc*100:.2f}%)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_accurate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Figure 2: Class-wise Accuracy Bar Plot ---
    class_accuracy = []
    for i in range(len(class_names)):
        class_mask = y_true == i
        if np.any(class_mask):
            acc = np.mean(y_pred[class_mask] == i)
            class_accuracy.append(acc)
        else:
            class_accuracy.append(0)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_accuracy, color=['#4c72b0', '#55a868', '#c44e52', '#8172b3'])
    plt.axhline(y=test_acc, color='r', linestyle='--', label=f'Overall Accuracy ({test_acc*100:.2f}%)')
    plt.title('Accuracy per Class', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.ylim(0, 1)
    plt.legend()
    
    # Add value labels on top of each bar
    for bar, acc in zip(bars, class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_per_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Figure 3: Prediction Confidence Distribution (for CORRECT predictions) ---
    correct_mask = (y_pred == y_true)
    correct_confidences = np.max(y_probs, axis=1)[correct_mask]
    
    plt.figure(figsize=(10, 6))
    plt.hist(correct_confidences, bins=20, color='green', alpha=0.7, edgecolor='black')
    plt.title('Prediction Confidence Distribution for CORRECT Predictions', fontsize=14, fontweight='bold')
    plt.xlabel('Model Confidence', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distribution_correct.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Figure 4: Test Set Class Distribution ---
    unique, counts = np.unique(y_true, return_counts=True)
    plt.figure(figsize=(10, 6))
    bars = plt.bar([class_names[i] for i in unique], counts, color='skyblue', edgecolor='black')
    plt.title('Test Set Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontweight='bold')
    plt.ylabel('Number of Samples', fontweight='bold')
    
    # Add count labels on top of each bar
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_set_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Print a detailed report to console as well ---
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT:")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    print(f"\nVisualizations saved to: {output_dir}/")
    print("Files created:")
    print("  - confusion_matrix_accurate.png")
    print("  - accuracy_per_class.png")
    print("  - confidence_distribution_correct.png")
    print("  - test_set_distribution.png")

def main():
    """Main function to run the visualization."""
    print("🚀 Generating Accurate Visualizations for Proven Model (81.57%)")
    print("="*70)
    model, test_images, test_labels, test_files = load_data_and_model()
    create_comprehensive_visualizations(model, test_images, test_labels)
    print("\n✅ Visualization complete! The graphs now reflect your true model accuracy.")

if __name__ == "__main__":
    main()