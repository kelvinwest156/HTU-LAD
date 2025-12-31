#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def validate_input_data(data):
    """Validate input data for predictions"""
    errors = []
    
    required_features = [
        'attendance_rate', 'previous_gpa', 'quiz_avg', 'assignment_avg',
        'mid_sem_score', 'study_time_hours', 'dashboard_time_hours',
        'current_gpa', 'assessment_avg', 'activity_index',
        'attendance_x_assignment', 'extracurricular_hours'
    ]
    
    # Check required features
    missing = set(required_features) - set(data.keys())
    if missing:
        errors.append(f"Missing required features: {list(missing)}")
    
    # Validate numeric ranges
    for feature in required_features:
        if feature in data:
            try:
                value = float(data[feature])
                if feature.endswith('_rate') and not (0 <= value <= 1):
                    errors.append(f"{feature} must be between 0 and 1")
                elif feature.endswith('_gpa') and not (0 <= value <= 4):
                    errors.append(f"{feature} must be between 0 and 4")
            except (ValueError, TypeError):
                errors.append(f"{feature} must be a numeric value")
    
    return errors

