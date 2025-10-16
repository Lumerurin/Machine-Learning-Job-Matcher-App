"""
Run this script to check and fix data quality issues
Place in your backend folder and run: python check_data_quality.py
"""

import json
import pickle

print("="*60)
print("DATA QUALITY CHECK")
print("="*60)

# Load the data
with open('job_role_data.json', 'r') as f:
    job_role_data = json.load(f)

with open('le_dict.pkl', 'rb') as f:
    le_dict = pickle.load(f)

print("\n1. EXPERIENCE LEVELS CHECK:")
print("-" * 40)
experience_levels = job_role_data['experience_levels']
print(f"Total experience levels: {len(experience_levels)}")
print("Raw values:")
for i, exp in enumerate(experience_levels):
    print(f"  [{i}] '{exp}' (length: {len(exp)})")

# Check for whitespace or special characters
issues = []
for exp in experience_levels:
    if exp != exp.strip():
        issues.append(f"'{exp}' has leading/trailing whitespace")
    if ',' in exp:
        issues.append(f"'{exp}' contains comma")

if issues:
    print("\n⚠️ ISSUES FOUND:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("\n✓ No whitespace or comma issues found")

print("\n2. LABEL ENCODER CLASSES:")
print("-" * 40)
le_classes = le_dict['experience_level'].classes_
print(f"Total classes in encoder: {len(le_classes)}")
for i, cls in enumerate(le_classes):
    print(f"  [{i}] '{cls}'")

print("\n3. SKILLS CHECK:")
print("-" * 40)
skills = job_role_data['skills']
print(f"Total skills: {len(skills)}")
skill_issues = []
for skill in skills:
    if skill != skill.strip():
        skill_issues.append(f"'{skill}' has whitespace")

if skill_issues:
    print(f"⚠️ Found {len(skill_issues)} skills with whitespace issues")
    print("First 5 examples:")
    for issue in skill_issues[:5]:
        print(f"  - {issue}")
else:
    print("✓ No skill whitespace issues")

print("\n4. QUALIFICATIONS CHECK:")
print("-" * 40)
qualifications = job_role_data['qualifications']
print(f"Total qualifications: {len(qualifications)}")
qual_issues = []
for qual in qualifications:
    if qual != qual.strip():
        qual_issues.append(f"'{qual}' has whitespace")

if qual_issues:
    print(f"⚠️ Found {len(qual_issues)} qualifications with whitespace issues")
    for issue in qual_issues:
        print(f"  - {issue}")
else:
    print("✓ No qualification whitespace issues")

print("\n5. JOB ROLES CHECK:")
print("-" * 40)
job_roles = job_role_data['job_roles']
print(f"Total job roles: {len(job_roles)}")
role_issues = []
for role in job_roles:
    if role != role.strip():
        role_issues.append(f"'{role}' has whitespace")

if role_issues:
    print(f"⚠️ Found {len(role_issues)} job roles with whitespace issues")
    for issue in role_issues:
        print(f"  - {issue}")
else:
    print("✓ No job role whitespace issues")

print("\n" + "="*60)
print("EXPERIENCE MATCHING LOGIC CHECK")
print("="*60)

# Show how experience matching works
print("\nCurrent experience level mapping:")
for i, exp in enumerate(le_classes):
    print(f"  {exp} → {i}")

print("\nExample experience match calculations:")
print("(Lower score difference = better match)\n")

examples = [
    ("Entry", "Entry", "Perfect match"),
    ("Entry", "Mid", "One level difference"),
    ("Entry", "Senior", "Two levels difference"),
    ("Mid", "Senior", "One level difference"),
]

for cand_exp, job_exp, desc in examples:
    if cand_exp in le_classes and job_exp in le_classes:
        cand_encoded = list(le_classes).index(cand_exp)
        job_encoded = list(le_classes).index(job_exp)
        match_score = 1.0 - abs(cand_encoded - job_encoded) / len(le_classes)
        print(f"{cand_exp:10s} vs {job_exp:10s}: {match_score:.3f} ({desc})")

print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

if issues or skill_issues or qual_issues or role_issues:
    print("\n⚠️ Data quality issues detected!")
    print("\nTo fix these issues:")
    print("1. Re-run the Colab notebook with data cleaning")
    print("2. Or run the fix script below")
else:
    print("\n✓ Data quality looks good!")

print("\n" + "="*60)