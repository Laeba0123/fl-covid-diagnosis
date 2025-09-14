import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from tensorflow import keras
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import base64
import traceback
from skimage import exposure
from skimage.restoration import denoise_tv_chambolle

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')  # Better secret key handling

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/images', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Global variables
model = None
class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
DEMO_MODE = False

# Load your trained model
def load_model():
    """Load the trained COVID-19 diagnosis model"""
    global DEMO_MODE
    model_path = 'models/federated_covid_model_proven.h5'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        DEMO_MODE = True
        return None
        
    try:
        model = keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        DEMO_MODE = False
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        DEMO_MODE = True
        return None

# Initialize model
model = load_model()

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Enhanced preprocessing for better predictions"""
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if needed (some images might be grayscale or RGBA)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to match training size (128x128)
        img = img.resize((128, 128))
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # **ENHANCEMENT: Contrast stretching**
        p2, p98 = np.percentile(img_array, (2, 98))
        img_array = exposure.rescale_intensity(img_array, in_range=(p2, p98))
        
        # **ENHANCEMENT: Denoising**
        img_array = denoise_tv_chambolle(img_array, weight=0.1)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
            
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def check_image_quality(image_path):
    """Check if image is suitable for analysis"""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Check contrast
        contrast = np.std(img_array)
        if contrast < 25:  # Low contrast
            return False, "Image has low contrast - please upload a clearer X-ray"
        
        # Check brightness
        brightness = np.mean(img_array)
        if brightness < 30 or brightness > 220:  # Too dark or too bright
            return False, "Image brightness is not optimal"
        
        return True, "Image quality OK"
    except Exception as e:
        return False, f"Error checking image quality: {str(e)}"

def refine_prediction(predictions, temperature=0.5):
    """
    Apply temperature scaling to sharpen confidence scores
    Lower temperature (0.1-0.5) = more confident predictions
    Higher temperature (1.0-2.0) = more conservative predictions
    """
    # Apply temperature scaling
    scaled_predictions = predictions / temperature
    # Softmax to get new probabilities
    exp_preds = np.exp(scaled_predictions - np.max(scaled_predictions))
    refined_preds = exp_preds / np.sum(exp_preds)
    
    return refined_preds

def ensemble_predictions(models, image_array):
    """Combine predictions from multiple models"""
    all_predictions = []
    
    for model in models:
        pred = model.predict(image_array, verbose=0)
        all_predictions.append(pred[0])  # Get the prediction array
    
    # Average the predictions
    avg_prediction = np.mean(all_predictions, axis=0)
    return np.expand_dims(avg_prediction, axis=0)

def plot_to_base64():
    """Convert matplotlib plot to base64 for HTML display"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return image_base64

def generate_demo_predictions():
    """Generate realistic demo predictions when model is not available"""
    # Create random but plausible predictions
    np.random.seed(hash('demo') % 1000)
    preds = np.abs(np.random.randn(4))
    preds = preds / np.sum(preds)  # Normalize to sum to 1
    
    # Make one class more prominent
    dominant_class = np.random.randint(0, 4)
    preds[dominant_class] *= 2
    preds = preds / np.sum(preds)
    
    return preds, dominant_class

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html', demo_mode=DEMO_MODE)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        flash('No file selected!', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected!', 'error')
        return redirect(request.url)
    
    if not allowed_file(file.filename):
        flash('Invalid file type! Please upload PNG, JPG, or JPEG.', 'error')
        return redirect(request.url)
    
    try:
        # Secure the filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Check image quality
        quality_ok, quality_msg = check_image_quality(filepath)
        if not quality_ok:
            flash(f'Image quality issue: {quality_msg}', 'warning')
            # Continue processing but show warning
        
        # Preprocess the image
        processed_image = preprocess_image(filepath)
        
        if processed_image is None:
            flash('Error processing image! Please try another image.', 'error')
            return redirect(request.url)
        
        # Make prediction (or use demo mode)
        if DEMO_MODE:
            predictions, predicted_class_idx = generate_demo_predictions()
            confidence = predictions[predicted_class_idx]
            diagnosis = class_names[predicted_class_idx]
            flash('⚠️ Running in demo mode - using simulated predictions', 'warning')
        else:
            # Use model for prediction
            models_list = [model]  # You can add more models here for ensemble
            if len(models_list) > 1:
                predictions = ensemble_predictions(models_list, processed_image)
            else:
                predictions = model.predict(processed_image, verbose=0)
            
            # Refine predictions
            predictions = refine_prediction(predictions, temperature=0.3)
            
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_class_idx]
            diagnosis = class_names[predicted_class_idx]
        
        # Get confidence percentage
        confidence_percent = confidence * 100
        
        # Determine confidence level
        if confidence_percent > 70:
            confidence_level = "high"
            show_warning = False
        elif confidence_percent > 40:
            confidence_level = "moderate"
            show_warning = True
        else:
            confidence_level = "low"
            show_warning = True

        # Create a visualization
        plt.figure(figsize=(10, 6))
        colors = ['#ff6b6b', '#ff9e6b', '#4ecdc4', '#6b5ce7']
        bars = plt.bar(range(len(class_names)), predictions[0], color=colors)
        
        plt.title('Prediction Confidence Scores', fontsize=16, fontweight='bold')
        plt.xlabel('Diagnosis Classes', fontsize=12)
        plt.ylabel('Confidence Score', fontsize=12)
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, predictions[0]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Convert plot to base64 for HTML
        plot_url = f"data:image/png;base64,{plot_to_base64()}"
        
        # Pass to template
        return render_template('results.html', 
                             diagnosis=diagnosis,
                             confidence=confidence_percent,
                             confidence_level=confidence_level,
                             show_warning=show_warning,
                             plot_url=plot_url,
                             filename=filename,
                             demo_mode=DEMO_MODE,
                             all_predictions=predictions[0],
                             class_names=class_names)
            
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error during prediction: {error_trace}")
        flash(f'Error during prediction: {str(e)}', 'error')
        return redirect(request.url)

@app.route('/about')
def about():
    """About page with project information"""
    # Check if visualization files exist
    viz_files = {}
    viz_paths = {
        'confusion_matrix': 'outputs/confusion_matrix_accurate.png',
        'accuracy': 'outputs/accuracy_per_class.png',
        'confidence': 'outputs/confidence_distribution_correct.png'
    }
    
    for name, path in viz_paths.items():
        if os.path.exists(path):
            try:
                with open(path, "rb") as image_file:
                    viz_files[name] = f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
            except Exception as e:
                print(f"Error loading visualization {name}: {e}")
    
    return render_template('about.html', viz_files=viz_files, demo_mode=DEMO_MODE)

@app.route('/enhance', methods=['POST'])
def enhance_image():
    """Endpoint to enhance image before prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save the file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{filename}")
        file.save(temp_path)
        
        # Apply enhancement techniques here
        # This is a placeholder - implement actual enhancement logic
        enhanced_img = Image.open(temp_path)
        
        # Simple enhancement: increase contrast
        enhanced_img = enhanced_img.convert('L')  # Convert to grayscale
        enhanced_img = Image.eval(enhanced_img, lambda x: x * 1.2)  # Increase contrast
        
        # Save enhanced image
        enhanced_path = os.path.join(app.config['UPLOAD_FOLDER'], f"enhanced_{filename}")
        enhanced_img.save(enhanced_path)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return jsonify({
            'message': 'Image enhanced successfully', 
            'enhanced_image': f"/static/uploads/enhanced_{filename}"
        })
        
    except Exception as e:
        return jsonify({'error': f'Enhancement failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File too large! Maximum size is 16MB.', 'error')
    return redirect(request.url)

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    flash('An internal error occurred. Please try again later.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Run the Flask app
    print("🚀 Starting Flask server...")
    if DEMO_MODE:
        print("⚠️  Running in DEMO MODE - no actual model loaded")
    print("👉 Open http://localhost:5000 in your browser")
    
    # Set TensorFlow logging level to reduce verbose output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    app.run(debug=True, host='0.0.0.0', port=5000)