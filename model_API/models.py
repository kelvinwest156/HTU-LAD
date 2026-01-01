#!/usr/bin/env python
# coding: utf-8

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

# Global model variables
gpa_model = None
risk_model = None
feature_columns = None

def load_models(model_path):
    """Load ML models into memory using absolute paths"""
    global gpa_model, risk_model, feature_columns
    
    try:
        gpa_path = os.path.join(model_path, 'gpa_predictor_final.pkl')
        risk_path = os.path.join(model_path, 'student_risk_classifier.pkl')


        
        # Verify files exist
        if not os.path.exists(gpa_path):
            raise FileNotFoundError(f"Missing GPA model at {gpa_path}")
        if not os.path.exists(risk_path):
            raise FileNotFoundError(f"Missing Risk model at {risk_path}")

        # Load models
        gpa_model = joblib.load(gpa_path)
        risk_model = joblib.load(risk_path)
        
        # Define expected feature columns
        feature_columns = [
            'attendance_rate', 'previous_gpa', 'quiz_avg', 'assignment_avg',
            'mid_sem_score', 'study_time_hours', 'dashboard_time_hours',
            'current_gpa', 'assessment_avg', 'activity_index',
            'attendance_x_assignment', 'extracurricular_hours'
        ]
        
        logger.info("✅ Models loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        raise e

def predict_gpa(input_data):
    """Predict student GPA"""
    features = prepare_features(input_data)
    prediction = gpa_model.predict(features)[0]
    
    return {
        'student_id': input_data.get('student_id', 'unknown'),
        'predicted_gpa': round(float(prediction), 2),
        'timestamp': datetime.now().isoformat()
    }

def predict_risk(input_data):
    """Predict student risk category"""
    features = prepare_features(input_data)
    
    prediction = risk_model.predict(features)[0]
    probabilities = risk_model.predict_proba(features)[0]
    
    class_mapping = {0: 'At Risk', 1: 'Average', 2: 'Excellent'}
    risk_label = class_mapping.get(prediction, 'Unknown')
    
    return {
        'student_id': input_data.get('student_id', 'unknown'),
        'risk_category': risk_label,
        'confidence': round(float(probabilities[prediction]), 3),
        'timestamp': datetime.now().isoformat()
    }

def predict_comprehensive(input_data):
    """Get both GPA and risk predictions"""
    return {
        'student_id': input_data.get('student_id', 'unknown'),
        'gpa_prediction': predict_gpa(input_data),
        'risk_prediction': predict_risk(input_data),
        'timestamp': datetime.now().isoformat()
    }

def prepare_features(input_data):
    """Convert input data to model-ready DataFrame"""
    feature_dict = {col: input_data.get(col, 0) for col in feature_columns}
    return pd.DataFrame([feature_dict], columns=feature_columns)
