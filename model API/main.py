from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import os

# Changed: Removed the dots for standard imports
from models import load_models, predict_gpa, predict_risk, predict_comprehensive
from validation import validate_input_data

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # SETUP LOGGING
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # FIXED: Calculate the ABSOLUTE path to the current folder
    # This ensures the server finds the files regardless of where it starts
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Use BASE_DIR as the default if no environment variable is set
    model_folder = os.getenv('MODEL_PATH', BASE_DIR)
    app.config['MODEL_PATH'] = model_folder
    
    # FIXED: Load models immediately during app creation
    # @app.before_first_request is deprecated and unreliable on some servers
    try:
        load_models(app.config['MODEL_PATH'])
        logger.info(f"✅ Models successfully loaded from: {app.config['MODEL_PATH']}")
    except Exception as e:
        logger.error(f"❌ Critical Error loading models: {str(e)}")

    @app.route('/health', methods=['GET'])
   def health_check():
        return jsonify({
            'status': 'healthy',
            'searching_in': app.config['MODEL_PATH'],
            'timestamp': datetime.now().isoformat()
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
