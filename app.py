# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import tensorflow as tf
import numpy as np
from pathlib import Path
import traceback

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Global model variable
model = None
class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

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
    try:
        PROJECT_ROOT = get_project_root()
        model_path = PROJECT_ROOT / "models" / "federated_covid_model_proven.h5"
        print(f"Loading model from: {model_path}")
        
        # Check if model file exists
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(traceback.format_exc())
        return None

def preprocess_image(image_path):
    """Preprocess image exactly like during training."""
    try:
        # Load image
        img = tf.io.read_file(str(image_path))
        img = tf.image.decode_jpeg(img, channels=3)
        
        # Resize to match training size (128x128)
        img = tf.image.resize(img, [128, 128])
        
        # Convert to float32 (same as training)
        img = tf.image.convert_image_dtype(img, tf.float32)
        
        return img
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        raise

def simple_predict(image_path):
    """Simple prediction without augmentation for debugging."""
    global model
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_path)
        
        # Add batch dimension
        processed_image = tf.expand_dims(processed_image, axis=0)
        
        # Get prediction
        prediction = model.predict(processed_image, verbose=0)
        
        # Get confidence and class
        confidence = np.max(prediction)
        class_idx = np.argmax(prediction)
        
        return {
            "class": class_names[class_idx],
            "confidence": float(confidence),
            "is_confident": confidence >= 0.7,
            "all_predictions": prediction.tolist()
        }
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        raise

def predict_with_confidence(image_path, confidence_threshold=0.7):
    """Make prediction with enhanced confidence using TTA."""
    global model
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_path)
        
        # Use test time augmentation
        augmentations = []
        
        # Original image
        augmentations.append(processed_image)
        
        # Horizontal flip
        augmentations.append(tf.image.flip_left_right(processed_image))
        
        # Small rotations
        augmentations.append(tf.image.rot90(processed_image, k=1))
        augmentations.append(tf.image.rot90(processed_image, k=3))
        
        # Get predictions for all augmentations
        predictions = []
        for aug_img in augmentations:
            # Add batch dimension
            aug_img = tf.expand_dims(aug_img, axis=0)
            pred = model.predict(aug_img, verbose=0)
            predictions.append(pred)
        
        # Average the predictions
        avg_prediction = np.mean(predictions, axis=0)
        
        # Get confidence and class
        confidence = np.max(avg_prediction)
        class_idx = np.argmax(avg_prediction)
        
        return {
            "class": class_names[class_idx],
            "confidence": float(confidence),
            "is_confident": confidence >= confidence_threshold,
            "all_predictions": avg_prediction.tolist()
        }
    except Exception as e:
        print(f"Error in TTA prediction: {str(e)}")
        print(traceback.format_exc())
        # Fall back to simple prediction
        return simple_predict(image_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the model when the app starts
print("Loading trained model...")
model = load_trained_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If no file is selected
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        # If file is allowed
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            
            # Add timestamp to make filename unique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{filename}"
            
            # Save the file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            file.save(filepath)
            
            try:
                # Check if model is loaded
                if model is None:
                    flash('Model not loaded. Please restart the server.', 'error')
                    return redirect(request.url)
                
                # Get prediction with enhanced confidence
                result = predict_with_confidence(filepath)
                
                # Format confidence as percentage
                confidence_percent = round(result['confidence'] * 100, 2)
                
                # Prepare result message based on confidence
                if result['is_confident']:
                    message = f"High confidence prediction. This is likely {result['class']}."
                else:
                    message = "Low confidence prediction. Please consult a healthcare professional for accurate diagnosis."
                
                return render_template('result.html', 
                                     prediction=result['class'],
                                     confidence=confidence_percent,
                                     message=message,
                                     image_url=filepath)
            
            except Exception as e:
                error_msg = f'Error processing image: {str(e)}'
                print(error_msg)
                print(traceback.format_exc())
                flash(error_msg, 'error')
                return redirect(request.url)
        
        else:
            flash('Allowed image types are: png, jpg, jpeg, bmp, tiff', 'error')
            return redirect(request.url)
    
    return render_template('diagnose.html')

# Add error handler
@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error=str(error)), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error=str(error)), 404

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Check if model loaded successfully
    if model is None:
        print("WARNING: Model failed to load. Server will start but predictions will fail.")
    
    # Run the server
    print("Starting server...")
    print("Open http://127.0.0.1:5000 in your web browser")
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)