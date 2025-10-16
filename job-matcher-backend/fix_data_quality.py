"""
Run this to fix data quality issues in your existing files
Place in your backend folder and run: python fix_data_quality.py
"""

import json
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("="*60)
print("FIXING DATA QUALITY ISSUES")
print("="*60)

# Load the data
with open('job_role_data.json', 'r') as f:
    job_role_data = json.load(f)

with open('le_dict.pkl', 'rb') as f:
    le_dict = pickle.load(f)

# Clean experience levels
print("\n1. Cleaning experience levels...")
original_exp = job_role_data['experience_levels'].copy()
job_role_data['experience_levels'] = [exp.strip().rstrip(',') for exp in job_role_data['experience_levels']]
job_role_data['experience_levels'] = list(set(job_role_data['experience_levels']))  # Remove duplicates
job_role_data['experience_levels'].sort()  # Sort for consistency

print(f"   Before: {original_exp}")
print(f"   After:  {job_role_data['experience_levels']}")

# Recreate the label encoder with clean data
print("\n2. Recreating experience level encoder...")
le_exp = LabelEncoder()
le_exp.fit(job_role_data['experience_levels'])
le_dict['experience_level'] = le_exp

print(f"   New encoding:")
for i, exp in enumerate(le_exp.classes_):
    print(f"     {i}: {exp}")

# Clean skills (remove leading/trailing whitespace)
print("\n3. Cleaning skills...")
original_skill_count = len(job_role_data['skills'])
job_role_data['skills'] = [skill.strip() for skill in job_role_data['skills']]
job_role_data['skills'] = list(set(job_role_data['skills']))  # Remove duplicates
job_role_data['skills'].sort()

print(f"   Original: {original_skill_count} skills")
print(f"   After:    {len(job_role_data['skills'])} skills")

# Clean qualifications
print("\n4. Cleaning qualifications...")
original_qual_count = len(job_role_data['qualifications'])
job_role_data['qualifications'] = [qual.strip() for qual in job_role_data['qualifications']]
job_role_data['qualifications'] = list(set(job_role_data['qualifications']))
job_role_data['qualifications'].sort()

print(f"   Original: {original_qual_count} qualifications")
print(f"   After:    {len(job_role_data['qualifications'])} qualifications")

# Clean job roles
print("\n5. Cleaning job roles...")
original_role_count = len(job_role_data['job_roles'])
job_role_data['job_roles'] = [role.strip() for role in job_role_data['job_roles']]
job_role_data['job_roles'] = list(set(job_role_data['job_roles']))
job_role_data['job_roles'].sort()

print(f"   Original: {original_role_count} job roles")
print(f"   After:    {len(job_role_data['job_roles'])} job roles")

# Update experience encoding in job_role_data
print("\n6. Updating experience encoding in requirements...")
for role, exp_encoded in job_role_data['experience'].items():
    # Map old encoding to new encoding
    # This assumes the order might have changed
    old_classes = ['Entry', 'Entry,', 'Mid', 'Senior']  # Example - adjust based on your data
    if exp_encoded < len(old_classes):
        old_exp = old_classes[int(exp_encoded)]
        cleaned_exp = old_exp.strip().rstrip(',')
        if cleaned_exp in le_exp.classes_:
            new_encoded = int(le_exp.transform([cleaned_exp])[0])
            job_role_data['experience'][role] = new_encoded

print("   Experience mappings updated")

# Save cleaned data
print("\n7. Saving cleaned files...")

with open('job_role_data.json', 'w') as f:
    json.dump(job_role_data, f, indent=2)
print("   ✓ job_role_data.json saved")

with open('le_dict.pkl', 'wb') as f:
    pickle.dump(le_dict, f)
print("   ✓ le_dict.pkl saved")

print("\n" + "="*60)
print("✓ DATA CLEANING COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Restart your Flask server: python app.py")
print("2. The duplicate 'Entry' and 'Entry,' should now be fixed")
print("3. Test the application to verify")
print("\n" + "="*60)