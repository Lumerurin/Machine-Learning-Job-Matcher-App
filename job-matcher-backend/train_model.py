import tensorflow as tf
import numpy as np
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity

print("="*60)
print("RETRAINING MODEL WITH CURRENT KERAS VERSION")
print("="*60)

print("\nLoading encoders and data...")

# Load encoders
with open('mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)
with open('ohe_qual.pkl', 'rb') as f:
    ohe_qual = pickle.load(f)
with open('le_dict.pkl', 'rb') as f:
    le_dict = pickle.load(f)
with open('job_role_data.json', 'r') as f:
    job_role_data = json.load(f)

print("✓ All encoders loaded")

# Generate synthetic training data from job_role_data
print("\nGenerating training data...")
X_train = []
y_train = []

job_roles = job_role_data['job_roles']
all_skills = job_role_data['skills']
qualifications = job_role_data['qualifications']
experience_levels = job_role_data['experience_levels']

# Generate 500 training samples
np.random.seed(42)
for _ in range(500):
    # Random candidate profile
    num_skills = np.random.randint(2, 8)
    candidate_skills = list(np.random.choice(all_skills, num_skills, replace=False))
    candidate_qual = np.random.choice(qualifications)
    candidate_exp = np.random.choice(experience_levels)
    
    # Random job role
    target_role = np.random.choice(job_roles)
    
    # Encode candidate
    skills_encoded = mlb.transform([candidate_skills])
    qual_encoded = ohe_qual.transform([[candidate_qual]])
    candidate_features = np.concatenate([skills_encoded, qual_encoded], axis=1)
    candidate_exp_encoded = le_dict['experience_level'].transform([candidate_exp])[0]
    
    # Get job requirements
    one_hot_cols = list(mlb.classes_) + list(ohe_qual.get_feature_names_out(['qualification']))
    job_features = np.array([job_role_data['requirements'][target_role][col] for col in one_hot_cols]).reshape(1, -1)
    job_exp_encoded = job_role_data['experience'][target_role]
    
    # Calculate similarity scores
    skill_sim = cosine_similarity(candidate_features + 1e-5, job_features + 1e-5)[0][0]
    exp_match = 1.0 - abs(candidate_exp_encoded - job_exp_encoded) / len(le_dict['experience_level'].classes_)
    
    # Generate target score (weighted combination with noise)
    target_score = (0.6 * skill_sim + 0.4 * exp_match) * 5
    target_score += np.random.normal(0, 0.3)
    target_score = np.clip(target_score, 1, 5)
    
    X_train.append([skill_sim, exp_match])
    y_train.append(target_score)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"✓ Generated {len(X_train)} training samples")

# Build and train model
print("\nBuilding model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("✓ Model built")

print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    verbose=0
)

print(f"✓ Training complete! Final MAE: {history.history['mae'][-1]:.4f}")

# Save model
model.save('job_matcher_model.keras')
print("✓ Model saved to 'job_matcher_model.keras'")

# Test prediction
test_input = np.array([[0.8, 0.9]])
prediction = model.predict(test_input, verbose=0)
print(f"\nTest prediction: {prediction[0][0]:.2f}/5.0")

print("\n" + "="*60)
print("SUCCESS! Model retrained and saved.")
print("Now run: python app.py")
print("="*60)