#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import os

# Use standard imports (no dots) for reliability on Render
from models import load_models, predict_gpa, predict_risk, predict_comprehensive
from validation import validate_input_data

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # DYNAMIC PATH RESOLUTION
    # This finds the absolute path of the directory containing main.py
    # This solves the [Errno 2] issue by pointing exactly to the folder
# Reliable base directory: folder where main.py lives
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    app.config['MODEL_PATH'] = BASE_DIR

    # Load models immediately during app startup
    try:
        load_models(app.config['MODEL_PATH'])
        logger.info(f"✅ Models loaded successfully from: {app.config['MODEL_PATH']}")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {str(e)}")

    @app.route('/health', methods=['GET'])
    def health_check():
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
            return jsonify({'error': 'Internal server error'}), 500
    
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

# Define 'app' for Gunicorn
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
