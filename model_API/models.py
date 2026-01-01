#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Global model variables
gpa_model = None
risk_model = None
feature_columns = None

def load_models(model_path):
    """Load both ML models into memory"""
    global gpa_model, risk_model, feature_columns
    
    try:
        # These paths will now be absolute based on the BASE_DIR from main.py
        gpa_model_path = os.path.join(model_path, 'gpa_predictor_final.pkl')
        gpa_model = joblib.load(gpa_model_path)
        
        risk_model_path = os.path.join(model_path, 'student_risk_classifier.pkl')
        risk_model = joblib.load(risk_model_path)
        
        # Define expected feature columns (update with your actual features)
        feature_columns = [
            'attendance_rate', 'previous_gpa', 'quiz_avg', 'assignment_avg',
            'mid_sem_score', 'study_time_hours', 'dashboard_time_hours',
            'current_gpa', 'assessment_avg', 'activity_index',
            'attendance_x_assignment', 'extracurricular_hours'
        ]
        
        print("✅ Models loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        raise e

def predict_gpa(input_data):
    """Predict student GPA"""
    features = prepare_features(input_data)
    prediction = gpa_model.predict(features)[0]
    
    return {
        'student_id': input_data.get('student_id', 'unknown'),
        'predicted_gpa': round(float(prediction), 2),
        'confidence': 'high',
        'timestamp': datetime.now().isoformat(),
        'model_version': '1.0'
    }

def predict_risk(input_data):
    """Predict student risk category"""
    features = prepare_features(input_data)
    
    # Get prediction and probabilities
    prediction = risk_model.predict(features)[0]
    probabilities = risk_model.predict_proba(features)[0]
    
    # Map to meaningful labels
    class_mapping = {0: 'At Risk', 1: 'Average', 2: 'Excellent'}
    risk_label = class_mapping.get(prediction, 'Unknown')
    confidence = float(probabilities[prediction])
    
    return {
        'student_id': input_data.get('student_id', 'unknown'),
        'risk_category': risk_label,
        'confidence': round(confidence, 3),
        'all_probabilities': {
            'At Risk': round(float(probabilities[0]), 3),
            'Average': round(float(probabilities[1]), 3),
            'Excellent': round(float(probabilities[2]), 3)
        },
        'timestamp': datetime.now().isoformat(),
        'model_version': '1.0',
        'recommended_actions': get_recommended_actions(risk_label, confidence)
    }

def predict_comprehensive(input_data):
    """Get both GPA and risk predictions"""
    gpa_result = predict_gpa(input_data)
    risk_result = predict_risk(input_data)
    
    return {
        'student_id': input_data.get('student_id', 'unknown'),
        'gpa_prediction': gpa_result,
        'risk_prediction': risk_result,
        'timestamp': datetime.now().isoformat()
    }

def prepare_features(input_data):
    """Convert input data to model-ready format"""
    # Create DataFrame with correct feature order
    feature_dict = {col: input_data.get(col, 0) for col in feature_columns}
    return pd.DataFrame([feature_dict], columns=feature_columns)

def get_recommended_actions(risk_category, confidence):
    """Generate recommended actions based on risk category"""
    # Your existing implementation
    actions = {
        'At Risk': ['Schedule academic advising', 'Enroll in tutoring'],
        'Average': ['Maintain study habits', 'Set improvement goals'],
        'Excellent': ['Explore honors projects', 'Research opportunities']
    }
    return actions.get(risk_category, [])

