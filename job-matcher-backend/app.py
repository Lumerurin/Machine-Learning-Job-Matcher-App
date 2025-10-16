"""
Flask API for Job Matcher
Place this file as app.py in your backend folder along with:
- mlb.pkl
- ohe_qual.pkl
- le_dict.pkl
- job_role_data.json
- job_matcher_model.keras
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

print("Loading model and encoders...")

# Load the trained model - try multiple formats
model = None
try:
    # Try .keras format first (Keras 3 native format)
    if os.path.exists('job_matcher_model.keras'):
        model = tf.keras.models.load_model('job_matcher_model.keras')
        print("✓ Model loaded successfully (.keras format)!")
    # Try SavedModel format
    elif os.path.exists('saved_model'):
        model = tf.keras.models.load_model('saved_model')
        print("✓ Model loaded successfully (SavedModel format)!")
    # Try H5 format
    elif os.path.exists('keras_model.h5'):
        model = tf.keras.models.load_model('keras_model.h5')
        print("✓ Model loaded successfully (H5 format)!")
    # Rebuild model and load weights as fallback
    elif os.path.exists('model_weights.h5'):
        print("Rebuilding model architecture and loading weights...")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.load_weights('model_weights.h5')
        print("✓ Model rebuilt and weights loaded successfully!")
    else:
        print("✗ No model files found!")
        print("   Looking for: job_matcher_model.keras, saved_model/, keras_model.h5, or model_weights.h5")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    model = None

# Load the encoders
try:
    with open('mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)
    print("✓ MultiLabelBinarizer loaded")
    
    with open('ohe_qual.pkl', 'rb') as f:
        ohe_qual = pickle.load(f)
    print("✓ OneHotEncoder loaded")
    
    with open('le_dict.pkl', 'rb') as f:
        le_dict = pickle.load(f)
    print("✓ LabelEncoder dictionary loaded")
    
    with open('job_role_data.json', 'r') as f:
        job_role_data = json.load(f)
    print("✓ Job role data loaded")
    
    print(f"\nAvailable options:")
    print(f"  - {len(job_role_data['skills'])} skills")
    print(f"  - {len(job_role_data['qualifications'])} qualifications")
    print(f"  - {len(job_role_data['experience_levels'])} experience levels")
    print(f"  - {len(job_role_data['job_roles'])} job roles")
    
except Exception as e:
    print(f"✗ Error loading encoders: {e}")
    import traceback
    traceback.print_exc()
    mlb = None
    ohe_qual = None
    le_dict = None
    job_role_data = None

@app.route('/')
def home():
    """Welcome endpoint"""
    return jsonify({
        'message': 'Job Matcher API',
        'status': 'running',
        'endpoints': {
            'health': '/api/health',
            'options': '/api/options',
            'predict': '/api/predict (POST)'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'api': 'healthy',
        'model_loaded': model is not None,
        'encoders_loaded': all([mlb, ohe_qual, le_dict, job_role_data])
    }
    
    if not all(status.values()):
        return jsonify({'status': 'unhealthy', 'details': status}), 500
    
    return jsonify({'status': 'healthy', 'message': 'All systems operational'})

@app.route('/api/options', methods=['GET'])
def get_options():
    """Return all available options for dropdowns"""
    if not job_role_data:
        return jsonify({'error': 'Job role data not loaded'}), 500
    
    return jsonify({
        'skills': sorted(job_role_data['skills']),
        'qualifications': sorted(job_role_data['qualifications']),
        'experience_levels': job_role_data['experience_levels'],
        'job_roles': sorted(job_role_data['job_roles'])
    })

@app.route('/api/predict', methods=['POST'])
def predict_match():
    """Predict match score for a candidate-job role combination"""
    
    # Check if all components are loaded
    if not all([model, mlb, ohe_qual, le_dict, job_role_data]):
        return jsonify({'error': 'Server not ready. Model or encoders not loaded.'}), 500
    
    try:
        data = request.json
        
        # Extract input data
        selected_skills = data.get('skills', [])
        selected_qualification = data.get('qualification', '')
        selected_experience = data.get('experience_level', '')
        target_job_role = data.get('job_role', '')
        
        # Validate inputs
        if not selected_skills:
            return jsonify({'error': 'Please select at least one skill'}), 400
        if not selected_qualification:
            return jsonify({'error': 'Please select a qualification'}), 400
        if not selected_experience:
            return jsonify({'error': 'Please select an experience level'}), 400
        if not target_job_role:
            return jsonify({'error': 'Please select a target job role'}), 400
        
        # Validate job role exists
        if target_job_role not in job_role_data['job_roles']:
            return jsonify({'error': f'Job role "{target_job_role}" not found in dataset'}), 404
        
        # Validate skills exist
        valid_skills = [s for s in selected_skills if s in mlb.classes_]
        if not valid_skills:
            return jsonify({'error': 'None of the selected skills are valid'}), 400
        
        if len(valid_skills) < len(selected_skills):
            print(f"Warning: Some skills were invalid and ignored")
        
        # --- STEP 1: Encode candidate features ---
        
        # One-hot encode skills using MultiLabelBinarizer
        sample_skills_encoded = mlb.transform([valid_skills])
        
        # One-hot encode qualification
        sample_qualification_encoded = ohe_qual.transform([[selected_qualification]])
        
        # Combine skills and qualification features
        sample_candidate_features = np.concatenate([
            sample_skills_encoded,
            sample_qualification_encoded
        ], axis=1)
        
        # Label encode experience level
        sample_experience_encoded = le_dict['experience_level'].transform([selected_experience])[0]
        
        # --- STEP 2: Get target job role requirements ---
        
        # Get one-hot encoded columns (same order as training)
        one_hot_encoded_columns = list(mlb.classes_) + list(ohe_qual.get_feature_names_out(['qualification']))
        
        # Get job role feature vector
        target_job_features = np.array(
            [job_role_data['requirements'][target_job_role][col] for col in one_hot_encoded_columns]
        ).reshape(1, -1)
        
        # Get job role's expected experience level
        target_job_experience_encoded = job_role_data['experience'][target_job_role]
        
        # --- STEP 3: Calculate similarity scores ---
        
        # Cosine similarity for skills and qualifications
        skill_qual_similarity = cosine_similarity(
            sample_candidate_features + 1e-5,
            target_job_features + 1e-5
        )[0][0]
        
        # Experience match score (normalized difference)
        num_experience_levels = len(le_dict['experience_level'].classes_)
        experience_match_score = 1.0 - abs(
            sample_experience_encoded - target_job_experience_encoded
        ) / num_experience_levels
        
        # --- STEP 4: Predict match score using the model ---
        
        model_input = np.array([[skill_qual_similarity, experience_match_score]])
        predicted_score = float(model.predict(model_input, verbose=0)[0][0])
        
        # Clip score to valid range (1-5)
        predicted_score = np.clip(predicted_score, 1.0, 5.0)
        
        # Calculate percentage for UI
        score_percentage = (predicted_score / 5.0) * 100
        
        # --- STEP 5: Return results ---
        
        return jsonify({
            'success': True,
            'match_score': round(predicted_score, 2),
            'score_percentage': round(score_percentage, 1),
            'skill_qual_similarity': round(float(skill_qual_similarity), 3),
            'experience_match_score': round(float(experience_match_score), 3),
            'job_role': target_job_role,
            'candidate_profile': {
                'skills': valid_skills,
                'qualification': selected_qualification,
                'experience_level': selected_experience
            },
            'max_score': 5.0
        })
    
    except KeyError as e:
        print(f"KeyError: {e}")
        return jsonify({'error': f'Invalid input field: {str(e)}'}), 400
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/job-roles/<job_role>/requirements', methods=['GET'])
def get_job_requirements(job_role):
    """Get requirements for a specific job role"""
    if not job_role_data:
        return jsonify({'error': 'Job role data not loaded'}), 500
    
    if job_role not in job_role_data['job_roles']:
        return jsonify({'error': f'Job role "{job_role}" not found'}), 404
    
    # Get top skills for this role
    role_reqs = job_role_data['requirements'][job_role]
    
    # Get skill columns only (exclude qualification columns)
    skill_scores = {k: v for k, v in role_reqs.items() if k in mlb.classes_}
    
    # Sort by importance
    top_skills = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return jsonify({
        'job_role': job_role,
        'top_skills': [{'skill': s, 'importance': round(i, 3)} for s, i in top_skills],
        'experience_level_code': job_role_data['experience'][job_role]
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Flask API Server...")
    print("="*60)
    print("API will be available at: http://localhost:5000")
    print("Test health check: http://localhost:5000/api/health")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')