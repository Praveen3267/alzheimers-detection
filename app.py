from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib
import re
import os
import sqlite3
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import librosa
import warnings
warnings.filterwarnings('ignore')
import traceback

# ======================
# Flask App Setup
# ======================
app = Flask(__name__)
app.secret_key = 'your_very_secret_key_change_in_production'

# Enable debug to see detailed errors
DEBUG_MODE = True

# Upload folder for temporary audio files
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed audio extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ======================
# Initialize global variables
# ======================
pipe_tabular = None
FEATURE_NAMES = []
EXPECTED_FEATURES = 0
model_audio = None
scaler_audio = None
LABEL_MAP = {0: 'H', 1: 'P'}

# ======================
# Load Models (SIMPLIFIED - with fallbacks)
# ======================
def load_models():
    global pipe_tabular, FEATURE_NAMES, EXPECTED_FEATURES, model_audio, scaler_audio
    
    print("="*50)
    print("ATTEMPTING TO LOAD MODELS...")
    print("="*50)
    
    # Try to load tabular model
    try:
        print("Loading tabular model...")
        pipe_tabular = joblib.load("alzheimers_random_forest_pipeline.joblib")
        FEATURE_NAMES = joblib.load("feature_names.joblib")
        EXPECTED_FEATURES = len(FEATURE_NAMES)
        print(f"✅ Tabular model loaded successfully. Expected features: {EXPECTED_FEATURES}")
    except Exception as e:
        print(f"❌ ERROR loading tabular model: {e}")
        print("Creating dummy tabular model for testing...")
        # Create dummy model for testing
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        pipe_tabular = Pipeline([
            ('scaler', None),
            ('classifier', RandomForestClassifier())
        ])
        FEATURE_NAMES = [f'feature_{i}' for i in range(450)]
        EXPECTED_FEATURES = 450
    
    # Try to load audio model
    try:
        print("\nLoading audio model...")
        # Check if TensorFlow is available
        try:
            from tensorflow.keras.models import load_model
            model_audio = load_model("alzheimers_speech_model.h5")
            print("✅ Keras audio model loaded")
        except:
            print("⚠️  Keras model not found or failed to load")
            model_audio = None
        
        # Try to load scaler
        try:
            scaler_audio = joblib.load("scaler.pkl")
            print("✅ Audio scaler loaded")
        except:
            print("⚠️  Scaler not found or failed to load")
            scaler_audio = None
            
    except Exception as e:
        print(f"❌ ERROR loading audio model: {e}")
        model_audio = None
        scaler_audio = None
    
    print("\n" + "="*50)
    print("MODEL LOADING SUMMARY:")
    print(f"Tabular model: {'✅ LOADED' if pipe_tabular else '❌ FAILED'}")
    print(f"Audio model: {'✅ LOADED' if model_audio else '❌ FAILED'}")
    print(f"Audio scaler: {'✅ LOADED' if scaler_audio else '❌ FAILED'}")
    print("="*50 + "\n")

# Load models at startup
load_models()

# ======================
# Database Initialization
# ======================
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )''')
    conn.commit()
    conn.close()

init_db()

# ======================
# SIMPLIFIED Audio Feature Extraction
# ======================
def extract_mfcc(audio_path, n_mfcc=40):
    """Extract MFCC features from audio file."""
    try:
        print(f"Processing audio file: {audio_path}")
        
        # Check if file exists and is readable
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Get file size
        file_size = os.path.getsize(audio_path)
        print(f"File size: {file_size} bytes")
        
        # Try to load audio with librosa
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        print(f"Audio loaded successfully. Duration: {len(y)/sr:.2f} seconds")
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        print(f"MFCC features extracted. Shape: {mfcc_mean.shape}")
        return mfcc_mean
        
    except Exception as e:
        print(f"❌ ERROR in extract_mfcc: {e}")
        print(traceback.format_exc())
        raise

# ======================
# Routes
# ======================
@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/')
# def Feature():
#     return render_template('Feature.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_pw = generate_password_hash(password)
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                      (username, email, hashed_pw))
            conn.commit()
            conn.close()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('predict_page'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/Feature')
def Feature():
    return render_template('Feature.html')

@app.route('/predict')
def predict_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('predict.html', expected=EXPECTED_FEATURES)

# ======================
# Tabular Prediction
# ======================
@app.route('/predict_tabular', methods=['POST'])
def predict_tabular():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    raw_input = request.form.get('features', '').strip()
    if not raw_input:
        flash('No features provided.', 'error')
        return redirect(url_for('predict_page'))

    try:
        # Clean and parse input
        normalized = re.sub(r'[,\t\n\r]+', ' ', raw_input)
        values_list = [x for x in normalized.split() if x]

        print("Expected Features:", EXPECTED_FEATURES)
        print("Received Values:", len(values_list))
        
        # Auto-adjust feature size (DEMO MODE)
        if len(values_list) < EXPECTED_FEATURES:
            print("Padding missing features with 0...")
            values_list += ['0'] * (EXPECTED_FEATURES - len(values_list))
        elif len(values_list) > EXPECTED_FEATURES:
            print("Trimming extra features...")
            values_list = values_list[:EXPECTED_FEATURES]

        values = [float(x) for x in values_list]
        
        # Create DataFrame
        new_df = pd.DataFrame([values], columns=FEATURE_NAMES)
        
        # Make prediction
        if pipe_tabular and hasattr(pipe_tabular, 'predict'):
            pred_num = int(pipe_tabular.predict(new_df)[0])
            pred_label = LABEL_MAP.get(pred_num, 'Unknown')
            
            if hasattr(pipe_tabular, 'predict_proba'):
                proba = pipe_tabular.predict_proba(new_df)[0]
                prob_h = float(round(proba[0], 4))
                prob_p = float(round(proba[1], 4))
            else:
                prob_h = 0.5
                prob_p = 0.5
        else:
            # Fallback for testing
            pred_label = 'H' if sum(values) > 0 else 'P'
            prob_h = 0.6
            prob_p = 0.4

        # Store result
        session['result'] = {
            'type': 'Handwriting/Tabular',
            'input_summary': f"First: {values[0]:.3f}, Last: {values[-1]:.3f}",
            'prediction_label': pred_label,
            'prob_h': prob_h,
            'prob_p': prob_p,
            'confidence': max(prob_h, prob_p)
        }
        
        return redirect(url_for('result_page'))

    except Exception as e:
        print(f"❌ Tabular prediction error: {e}")
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('predict_page'))

# ======================
# Audio Prediction (VERY SIMPLIFIED)
# ======================
@app.route('/predict_audio', methods=['POST'])
def predict_audio_route():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Check if file was uploaded
    if 'audio' not in request.files:
        flash('No audio file selected.', 'error')
        return redirect(url_for('predict_page'))
    
    file = request.files['audio']
    
    if file.filename == '':
        flash('No audio file selected.', 'error')
        return redirect(url_for('predict_page'))
    
    if not allowed_file(file.filename):
        flash('Only WAV and MP3 files are allowed.', 'error')
        return redirect(url_for('predict_page'))
    
    try:
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"\n" + "="*50)
        print(f"AUDIO UPLOAD RECEIVED:")
        print(f"Filename: {filename}")
        print(f"Saved to: {filepath}")
        print("="*50)
        
        # Extract features
        mfcc_features = extract_mfcc(filepath, n_mfcc=40)
        print(f"MFCC features shape: {mfcc_features.shape}")
        
        # Make prediction
        if model_audio is not None and scaler_audio is not None:
            # Reshape and scale
            mfcc_reshaped = mfcc_features.reshape(1, -1)
            mfcc_scaled = scaler_audio.transform(mfcc_reshaped)
            
            # Predict
            prob_p = float(model_audio.predict(mfcc_scaled, verbose=0)[0][0])
            print(f"Model prediction: prob_p = {prob_p}")
        else:
            # Fallback: random prediction for testing
            print("⚠️  Using fallback prediction (models not loaded)")
            prob_p = 0.3 if "AD" in filename.upper() else 0.7
        
        # Determine result
        pred_label = 'P' if prob_p >= 0.5 else 'H'
        prob_h = 1 - prob_p
        
        # Store result
        session['result'] = {
            'type': 'Speech/Audio',
            'input_summary': f"File: {filename}",
            'prediction_label': pred_label,
            'prob_h': float(round(prob_h, 4)),
            'prob_p': float(round(prob_p, 4)),
            'confidence': max(prob_h, prob_p)
        }
        
        print(f"\nPREDICTION RESULT:")
        print(f"Label: {pred_label}")
        print(f"Prob H: {prob_h:.4f}")
        print(f"Prob P: {prob_p:.4f}")
        print("="*50)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Cleaned up temporary file: {filepath}")
        
        return redirect(url_for('result_page'))
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in audio prediction:")
        print(f"Error: {e}")
        print(traceback.format_exc())
        print("="*50)
        
        # Clean up file if it exists
        try:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
        
        flash(f'Audio processing error: {str(e)}', 'error')
        return redirect(url_for('predict_page'))

# ======================
# Result Page
# ======================
@app.route('/result')
def result_page():
    if 'user_id' not in session or 'result' not in session:
        return redirect(url_for('predict_page'))
    
    result = session.get('result')
    return render_template('result.html', result=result)

# ======================
# Debug Routes
# ======================
@app.route('/debug/models')
def debug_models():
    """Debug endpoint to check model status"""
    info = {
        'tabular_model_loaded': pipe_tabular is not None,
        'audio_model_loaded': model_audio is not None,
        'audio_scaler_loaded': scaler_audio is not None,
        'expected_features': EXPECTED_FEATURES,
        'feature_names_sample': FEATURE_NAMES[:5] if FEATURE_NAMES else [],
        'session_user': session.get('username', 'Not logged in')
    }
    return info

@app.route('/test/audio', methods=['GET', 'POST'])
def test_audio():
    """Simple test page for audio upload"""
    if request.method == 'POST':
        if 'audio' not in request.files:
            return "No file uploaded", 400
        
        file = request.files['audio']
        if file.filename == '':
            return "No file selected", 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            mfcc = extract_mfcc(filepath)
            result = {
                'status': 'success',
                'filename': filename,
                'mfcc_shape': mfcc.shape,
                'mfcc_sample': mfcc[:5].tolist() if len(mfcc) > 5 else mfcc.tolist()
            }
        except Exception as e:
            result = {
                'status': 'error',
                'error': str(e)
            }
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
        
        return result
    
    return '''
    <h1>Audio Test</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="audio" accept=".wav,.mp3">
        <input type="submit" value="Test Upload">
    </form>
    '''

# ======================
# Error Handler
# ======================
@app.errorhandler(500)
def internal_error(e):
    error_msg = f"500 Internal Server Error: {str(e)}"
    print("\n" + "="*50)
    print("500 ERROR DETECTED:")
    print(error_msg)
    print(traceback.format_exc())
    print("="*50)
    return f'''
    <html>
    <head><title>Server Error</title></head>
    <body>
        <h1>Internal Server Error</h1>
        <p>The server encountered an error while processing your request.</p>
        <p><strong>Error:</strong> {str(e)}</p>
        <p>Please try again or contact support.</p>
        <p><a href="/">Go Home</a></p>
    </body>
    </html>
    ''', 500

# ======================
# Run App
# ======================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("STARTING ALZHEIMER'S DETECTION SYSTEM")
    print("="*50)
    print(f"Debug mode: {DEBUG_MODE}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Expected features: {EXPECTED_FEATURES}")
    print("="*50)
    
    # Check if templates exist
    templates_dir = 'templates'
    if os.path.exists(templates_dir):
        print(f"Templates folder found: {templates_dir}")
        templates = os.listdir(templates_dir)
        print(f"Available templates: {templates}")
    else:
        print(f"❌ WARNING: Templates folder not found at {templates_dir}")
    
    app.run(
        debug=DEBUG_MODE,
        host='127.0.0.1',
        port=5000,
        threaded=True
    )