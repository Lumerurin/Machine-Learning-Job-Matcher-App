"""
Complete Flask API for Job Matcher with SBERT Interview Scoring
Place this file as app.py in your backend folder along with:

Required files:
- mlb.pkl
- ohe_qual.pkl
- le_dict.pkl
- job_role_data.json
- job_matcher_model.keras

Optional SBERT files (for interview scoring):
- sbert_model/ (folder)
- ideal_embeddings.pkl
- ideal_answers_processed.csv
- candidate_interview_scores.csv
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

# Import SBERT components (will be set to None if not available)
try:
    from sentence_transformers import SentenceTransformer, util
    import pandas as pd
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Interview scoring will be disabled.")
    print("   Install with: pip install sentence-transformers")

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

print("="*60)
print("LOADING MODELS AND ENCODERS")
print("="*60)

# ==================== LOAD STRUCTURED MODEL ====================
print("\n1. Loading structured model and encoders...")

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
except Exception as e:
    print(f"✗ Error loading model: {e}")
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
    
    print(f"\n   Available options:")
    print(f"   - {len(job_role_data['skills'])} skills")
    print(f"   - {len(job_role_data['qualifications'])} qualifications")
    print(f"   - {len(job_role_data['experience_levels'])} experience levels")
    print(f"   - {len(job_role_data['job_roles'])} job roles")
    
except Exception as e:
    print(f"✗ Error loading encoders: {e}")
    mlb = None
    ohe_qual = None
    le_dict = None
    job_role_data = None

# ==================== LOAD SBERT COMPONENTS ====================
print("\n2. Loading SBERT interview scoring components...")

sbert_model = None
ideal_embeddings = None
df_ideal_answers = None
df_interview_scores = None

if SBERT_AVAILABLE:
    try:
        # Load SBERT model
        if os.path.exists('sbert_model'):
            sbert_model = SentenceTransformer('sbert_model')
            print("✓ SBERT model loaded")
        else:
            print("! SBERT model not found (optional - for interview scoring)")
        
        # Load ideal embeddings
        if os.path.exists('ideal_embeddings.pkl'):
            with open('ideal_embeddings.pkl', 'rb') as f:
                ideal_embeddings = pickle.load(f)
            print("✓ Ideal embeddings loaded")
        else:
            print("! Ideal embeddings not found (optional)")
        
        # Load ideal answers
        if os.path.exists('ideal_answers_processed.csv'):
            df_ideal_answers = pd.read_csv('ideal_answers_processed.csv')
            print("✓ Ideal answers loaded")
        else:
            print("! Ideal answers not found (optional)")
        
        # Load pre-computed interview scores (optional)
        if os.path.exists('candidate_interview_scores.csv'):
            df_interview_scores = pd.read_csv('candidate_interview_scores.csv')
            print("✓ Pre-computed interview scores loaded")
        else:
            print("! Pre-computed interview scores not found (optional)")
        
    except Exception as e:
        print(f"✗ Error loading SBERT components: {e}")
        sbert_model = None
        ideal_embeddings = None
        df_ideal_answers = None
        df_interview_scores = None
else:
    print("! SBERT library not available - interview scoring disabled")

print("\n" + "="*60)
print("LOADING COMPLETE")
print("="*60)
print(f"Structured Matching: {'✓ Ready' if model else '✗ Not Available'}")
print(f"Interview Scoring: {'✓ Ready' if (sbert_model and ideal_embeddings) else '✗ Not Available'}")
print("="*60 + "\n")

# ==================== ROUTES ====================

@app.route('/')
def home():
    """Welcome endpoint"""
    return jsonify({
        'message': 'Job Matcher API',
        'status': 'running',
        'features': {
            'structured_matching': model is not None,
            'interview_scoring': sbert_model is not None and ideal_embeddings is not None
        },
        'endpoints': {
            'health': '/api/health',
            'options': '/api/options',
            'predict': '/api/predict (POST)',
            'predict_combined': '/api/predict-combined (POST)',
            'interview_questions': '/api/interview-questions',
            'score_interview': '/api/score-interview (POST)'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'api': 'healthy',
        'model_loaded': model is not None,
        'encoders_loaded': all([mlb, ohe_qual, le_dict, job_role_data]),
        'sbert_loaded': sbert_model is not None and ideal_embeddings is not None
    }
    
    if not status['model_loaded'] or not status['encoders_loaded']:
        return jsonify({'status': 'unhealthy', 'details': status}), 500
    
    return jsonify({'status': 'healthy', 'message': 'All systems operational', 'details': status})

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
    """Predict match score for a candidate-job role combination (structured data only)"""
    
    if not all([model, mlb, ohe_qual, le_dict, job_role_data]):
        return jsonify({'error': 'Server not ready. Model or encoders not loaded.'}), 500
    
    try:
        data = request.json
        
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
        
        if target_job_role not in job_role_data['job_roles']:
            return jsonify({'error': f'Job role "{target_job_role}" not found'}), 404
        
        valid_skills = [s for s in selected_skills if s in mlb.classes_]
        if not valid_skills:
            return jsonify({'error': 'None of the selected skills are valid'}), 400
        
        # Encode features
        sample_skills_encoded = mlb.transform([valid_skills])
        sample_qualification_encoded = ohe_qual.transform([[selected_qualification]])
        sample_candidate_features = np.concatenate([
            sample_skills_encoded,
            sample_qualification_encoded
        ], axis=1)
        sample_experience_encoded = le_dict['experience_level'].transform([selected_experience])[0]
        
        # Get job requirements
        one_hot_encoded_columns = list(mlb.classes_) + list(ohe_qual.get_feature_names_out(['qualification']))
        target_job_features = np.array(
            [job_role_data['requirements'][target_job_role][col] for col in one_hot_encoded_columns]
        ).reshape(1, -1)
        target_job_experience_encoded = job_role_data['experience'][target_job_role]
        
        # Calculate similarities
        skill_qual_similarity = cosine_similarity(
            sample_candidate_features + 1e-5,
            target_job_features + 1e-5
        )[0][0]
        
        num_experience_levels = len(le_dict['experience_level'].classes_)
        experience_match_score = 1.0 - abs(
            sample_experience_encoded - target_job_experience_encoded
        ) / num_experience_levels
        
        # Predict
        model_input = np.array([[skill_qual_similarity, experience_match_score]])
        predicted_score = float(model.predict(model_input, verbose=0)[0][0])
        predicted_score = np.clip(predicted_score, 1.0, 5.0)
        
        score_percentage = (predicted_score / 5.0) * 100
        
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
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/interview-questions', methods=['GET'])
def get_interview_questions():
    """Get list of available interview questions"""
    
    if df_ideal_answers is None:
        return jsonify({'error': 'Interview questions not loaded', 'available': False}), 404
    
    questions = df_ideal_answers[['question_id', 'question']].to_dict(orient='records')
    
    return jsonify({
        'questions': questions,
        'total': len(questions),
        'available': True
    })

@app.route('/api/score-interview', methods=['POST'])
def score_interview():
    """Score candidate interview responses using SBERT"""
    
    if not all([sbert_model, ideal_embeddings]):
        return jsonify({'error': 'SBERT model not loaded. Interview scoring unavailable.'}), 500
    
    try:
        data = request.json
        responses = data.get('responses', [])
        
        if not responses:
            return jsonify({'error': 'No responses provided'}), 400
        
        scores = []
        
        for resp in responses:
            question_id = resp.get('question_id')
            response_text = resp.get('response', '')
            
            if not response_text or question_id not in ideal_embeddings:
                continue
            
            response_embedding = sbert_model.encode(response_text, convert_to_tensor=True)
            ideal_embedding = ideal_embeddings[question_id]
            similarity = util.cos_sim(response_embedding, ideal_embedding).item()
            
            scores.append({
                'question_id': question_id,
                'similarity_score': round(similarity, 4)
            })
        
        if not scores:
            return jsonify({'error': 'No valid responses to score'}), 400
        
        avg_score = sum(s['similarity_score'] for s in scores) / len(scores)
        
        return jsonify({
            'success': True,
            'average_interview_score': round(avg_score, 4),
            'question_scores': scores,
            'questions_answered': len(scores)
        })
    
    except Exception as e:
        print(f"Error scoring interview: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/interview-score/<candidate_id>', methods=['GET'])
def get_interview_score(candidate_id):
    """Get pre-computed interview score for a candidate"""
    
    if df_interview_scores is None:
        return jsonify({'error': 'Interview scores not available'}), 404
    
    candidate_score = df_interview_scores[
        df_interview_scores['candidate_id'] == candidate_id
    ]
    
    if candidate_score.empty:
        return jsonify({'error': f'No interview score found for {candidate_id}'}), 404
    
    return jsonify({
        'candidate_id': candidate_id,
        'avg_interview_score': float(candidate_score['avg_interview_score'].iloc[0]),
        'min_question_score': float(candidate_score['min_question_score'].iloc[0]),
        'max_question_score': float(candidate_score['max_question_score'].iloc[0]),
        'questions_answered': int(candidate_score['questions_answered'].iloc[0])
    })

@app.route('/api/predict-combined', methods=['POST'])
def predict_combined():
    """Combined prediction using both structured data and interview responses"""
    
    if not all([model, mlb, ohe_qual, le_dict, job_role_data]):
        return jsonify({'error': 'Structured model not ready'}), 500
    
    try:
        data = request.json
        
        selected_skills = data.get('skills', [])
        selected_qualification = data.get('qualification', '')
        selected_experience = data.get('experience_level', '')
        target_job_role = data.get('job_role', '')
        interview_responses = data.get('interview_responses', [])
        candidate_id = data.get('candidate_id', None)
        
        # Validate structured data
        if not all([selected_skills, selected_qualification, selected_experience, target_job_role]):
            return jsonify({'error': 'Missing structured data fields'}), 400
        
        # === PART 1: Structured Score ===
        valid_skills = [s for s in selected_skills if s in mlb.classes_]
        if not valid_skills:
            return jsonify({'error': 'No valid skills selected'}), 400
        
        sample_skills_encoded = mlb.transform([valid_skills])
        sample_qualification_encoded = ohe_qual.transform([[selected_qualification]])
        sample_candidate_features = np.concatenate([
            sample_skills_encoded,
            sample_qualification_encoded
        ], axis=1)
        sample_experience_encoded = le_dict['experience_level'].transform([selected_experience])[0]
        
        one_hot_encoded_columns = list(mlb.classes_) + list(ohe_qual.get_feature_names_out(['qualification']))
        target_job_features = np.array(
            [job_role_data['requirements'][target_job_role][col] for col in one_hot_encoded_columns]
        ).reshape(1, -1)
        target_job_experience_encoded = job_role_data['experience'][target_job_role]
        
        skill_qual_similarity = cosine_similarity(
            sample_candidate_features + 1e-5,
            target_job_features + 1e-5
        )[0][0]
        
        num_experience_levels = len(le_dict['experience_level'].classes_)
        experience_match_score = 1.0 - abs(
            sample_experience_encoded - target_job_experience_encoded
        ) / num_experience_levels
        
        model_input = np.array([[skill_qual_similarity, experience_match_score]])
        structured_score = float(model.predict(model_input, verbose=0)[0][0])
        structured_score = np.clip(structured_score, 1.0, 5.0)
        
        # === PART 2: Interview Score ===
        interview_score = None
        interview_details = None
        
        if candidate_id and df_interview_scores is not None:
            candidate_interview = df_interview_scores[
                df_interview_scores['candidate_id'] == candidate_id
            ]
            if not candidate_interview.empty:
                interview_score = float(candidate_interview['avg_interview_score'].iloc[0])
                interview_details = {
                    'source': 'pre-computed',
                    'questions_answered': int(candidate_interview['questions_answered'].iloc[0])
                }
        
        elif interview_responses and sbert_model and ideal_embeddings:
            scores = []
            for resp in interview_responses:
                question_id = resp.get('question_id')
                response_text = resp.get('response', '')
                
                if question_id in ideal_embeddings and response_text:
                    response_embedding = sbert_model.encode(response_text, convert_to_tensor=True)
                    ideal_embedding = ideal_embeddings[question_id]
                    similarity = util.cos_sim(response_embedding, ideal_embedding).item()
                    scores.append(similarity)
            
            if scores:
                interview_score = sum(scores) / len(scores)
                interview_details = {
                    'source': 'real-time',
                    'questions_answered': len(scores)
                }
        
        # === PART 3: Combined Score ===
        if interview_score is not None:
            interview_score_scaled = 1 + (interview_score * 4)
            combined_score = (0.6 * structured_score) + (0.4 * interview_score_scaled)
            combined_score = np.clip(combined_score, 1.0, 5.0)
            
            return jsonify({
                'success': True,
                'combined_score': round(combined_score, 2),
                'structured_score': round(structured_score, 2),
                'interview_score': round(interview_score, 4),
                'interview_score_scaled': round(interview_score_scaled, 2),
                'interview_details': interview_details,
                'skill_qual_similarity': round(float(skill_qual_similarity), 3),
                'experience_match_score': round(float(experience_match_score), 3),
                'job_role': target_job_role,
                'scoring_method': 'combined'
            })
        else:
            return jsonify({
                'success': True,
                'combined_score': round(structured_score, 2),
                'structured_score': round(structured_score, 2),
                'interview_score': None,
                'skill_qual_similarity': round(float(skill_qual_similarity), 3),
                'experience_match_score': round(float(experience_match_score), 3),
                'job_role': target_job_role,
                'scoring_method': 'structured_only',
                'note': 'No interview data available'
            })
    
    except Exception as e:
        print(f"Error in combined prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/job-roles/<job_role>/requirements', methods=['GET'])
def get_job_requirements(job_role):
    """Get requirements for a specific job role"""
    if not job_role_data:
        return jsonify({'error': 'Job role data not loaded'}), 500
    
    if job_role not in job_role_data['job_roles']:
        return jsonify({'error': f'Job role "{job_role}" not found'}), 404
    
    role_reqs = job_role_data['requirements'][job_role]
    skill_scores = {k: v for k, v in role_reqs.items() if k in mlb.classes_}
    top_skills = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return jsonify({
        'job_role': job_role,
        'top_skills': [{'skill': s, 'importance': round(i, 3)} for s, i in top_skills],
        'experience_level_code': job_role_data['experience'][job_role]
    })

# ==================== START SERVER ====================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("STARTING FLASK API SERVER")
    print("="*60)
    print("API will be available at: http://localhost:5000")
    print("Test health check: http://localhost:5000/api/health")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')