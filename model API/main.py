#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import os

# Import your custom logic
# Using standard imports instead of relative dots for easier deployment
from models import load_models, predict_gpa, predict_risk, predict_comprehensive
from validation import validate_input_data

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # SETUP LOGGING
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # DYNAMIC PATH RESOLUTION
    # This finds the absolute path of the directory containing main.py
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # If the models are in the same folder as main.py, use BASE_DIR
    # We allow an Environment Variable 'MODEL_PATH' to override this if needed
    model_folder = os.getenv('MODEL_PATH', BASE_DIR)
    app.config['MODEL_PATH'] = model_folder
    
    # LOAD MODELS AT STARTUP
    # This runs once when the server starts
    try:
        load_models(app.config['MODEL_PATH'])
        logger.info(f"✅ All models loaded successfully from: {app.config['MODEL_PATH']}")
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        # We don't 'raise' here so the /health endpoint can still tell us what's wrong
    
    # ROUTES
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Used by Render and your team to verify the server status"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_directory': app.config['MODEL_PATH'],
            'version': '1.0.1'
        })
    
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
            return jsonify({'error': str(e)}), 500
    
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
            return jsonify({'error': str(e)}), 500
    
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
            return jsonify({'error': str(e)}), 500
    
    return app

# CRITICAL FOR RENDER/GUNICORN:
# Create the app object at the top level of the file
app = create_app()

if __name__ == '__main__':
    # Local testing settings
    app.run(host='0.0.0.0', port=5000, debug=True)
