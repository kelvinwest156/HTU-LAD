#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import os

# Standardized imports (remove the dot if models.py is in the same folder)
from models import load_models, predict_gpa, predict_risk, predict_comprehensive
from validation import validate_input_data

def create_app():
    app = Flask(__name__)
    CORS(app) # Essential for frontend team access
    
    # Configuration
    # Uses Environment Variable or defaults to './models/' folder
    app.config['MODEL_PATH'] = os.getenv('MODEL_PATH', './models/')
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load models immediately during app creation
    # @app.before_first_request is deprecated in Flask 3.0+
    with app.app_context():
        try:
            load_models(app.config['MODEL_PATH'])
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            # Do not raise here if you want the app to start even if loading fails
    
    # Health check endpoint for Render monitoring
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    
    # GPA prediction endpoint
    @app.route('/predict/gpa', methods=['POST'])
    def predict_gpa_endpoint():
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            errors = validate_input_data(data)
            if errors:
                return jsonify({'errors': errors}), 400
            
            result = predict_gpa(data)
            return jsonify(result)
        except Exception as e:
            logger.error(f"GPA prediction error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500
    
    # Risk category prediction endpoint
    @app.route('/predict/risk-category', methods=['POST'])
    def predict_risk_endpoint():
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            errors = validate_input_data(data)
            if errors:
                return jsonify({'errors': errors}), 400
            
            result = predict_risk(data)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Risk prediction error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500
    
    # Comprehensive prediction endpoint
    @app.route('/predict/comprehensive', methods=['POST'])
    def predict_comprehensive_endpoint():
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            errors = validate_input_data(data)
            if errors:
                return jsonify({'errors': errors}), 400
            
            result = predict_comprehensive(data)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Comprehensive prediction error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500
    
    return app

# CRITICAL: Create the app instance at the top level for Gunicorn
app = create_app()

if __name__ == '__main__':
    # Local development settings
    app.run(host='0.0.0.0', port=5000, debug=True)
