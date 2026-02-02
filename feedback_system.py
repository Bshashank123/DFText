"""
USER FEEDBACK SYSTEM
Collect user validations to improve the system over time

This implements your idea of letting users validate AI predictions
"""

import os
import json
import datetime
from flask import jsonify

# Feedback storage
FEEDBACK_DIR = 'user_feedback'
os.makedirs(FEEDBACK_DIR, exist_ok=True)

class FeedbackCollector:
    """Collect and store user feedback on predictions"""
    
    @staticmethod
    def save_feedback(image_path, prediction_data, user_validation, user_comments=''):
        """
        Save user feedback for a prediction
        
        Args:
            image_path: Path to analyzed image
            prediction_data: Full prediction output (verdict, scores, etc.)
            user_validation: User's assessment ('correct', 'incorrect', 'unsure')
            user_comments: Optional user comments
        """
        
        timestamp = datetime.datetime.now().isoformat()
        
        feedback_entry = {
            'timestamp': timestamp,
            'image_path': image_path,
            'system_verdict': prediction_data.get('verdict'),
            'system_confidence': prediction_data.get('confidence'),
            'system_scores': prediction_data.get('scores', {}),
            'user_validation': user_validation,
            'user_comments': user_comments,
            'text_length': prediction_data.get('text_length', 0)
        }
        
        # Save to JSON file
        filename = f"feedback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        filepath = os.path.join(FEEDBACK_DIR, filename)
        
        with open(filepath, 'w') as f:
            json.dump(feedback_entry, f, indent=2)
        
        return filepath
    
    @staticmethod
    def load_all_feedback():
        """Load all feedback entries"""
        feedback_list = []
        
        for filename in os.listdir(FEEDBACK_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(FEEDBACK_DIR, filename)
                try:
                    with open(filepath, 'r') as f:
                        feedback_list.append(json.load(f))
                except:
                    continue
        
        return feedback_list
    
    @staticmethod
    def analyze_feedback():
        """
        Analyze collected feedback to find improvement opportunities
        """
        feedback_list = FeedbackCollector.load_all_feedback()
        
        if not feedback_list:
            return {
                'total_feedback': 0,
                'message': 'No feedback collected yet'
            }
        
        # Count validations
        correct_count = sum(1 for f in feedback_list if f['user_validation'] == 'correct')
        incorrect_count = sum(1 for f in feedback_list if f['user_validation'] == 'incorrect')
        unsure_count = sum(1 for f in feedback_list if f['user_validation'] == 'unsure')
        
        # Calculate accuracy
        total_decisive = correct_count + incorrect_count
        if total_decisive > 0:
            user_accuracy = correct_count / total_decisive
        else:
            user_accuracy = 0.0
        
        # Find problematic cases (high confidence but wrong)
        false_positives = []  # System said AI but user said Human
        false_negatives = []  # System said Human but user said AI
        
        for f in feedback_list:
            if f['user_validation'] == 'incorrect':
                if f['system_confidence'] > 70:  # High confidence error
                    if 'AI' in f['system_verdict']:
                        false_positives.append(f)
                    elif 'Human' in f['system_verdict']:
                        false_negatives.append(f)
        
        # Module performance from feedback
        module_issues = {}
        for f in feedback_list:
            if f['user_validation'] == 'incorrect':
                scores = f.get('system_scores', {})
                # Find which module had the strongest signal
                if scores:
                    max_module = max(scores.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
                    module_name = max_module[0]
                    module_issues[module_name] = module_issues.get(module_name, 0) + 1
        
        analysis = {
            'total_feedback': len(feedback_list),
            'correct': correct_count,
            'incorrect': incorrect_count,
            'unsure': unsure_count,
            'user_accuracy': user_accuracy,
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'module_issues': module_issues,
            'high_confidence_errors': len(false_positives) + len(false_negatives)
        }
        
        return analysis
    
    @staticmethod
    def get_retraining_data():
        """
        Get validated samples for retraining/recalibration
        Returns separate lists of confirmed real and confirmed fake
        """
        feedback_list = FeedbackCollector.load_all_feedback()
        
        confirmed_real = []
        confirmed_fake = []
        
        for f in feedback_list:
            if f['user_validation'] != 'correct':
                continue  # Only use validated correct predictions
            
            if 'Human' in f['system_verdict']:
                confirmed_real.append({
                    'image_path': f['image_path'],
                    'scores': f['system_scores']
                })
            elif 'AI' in f['system_verdict']:
                confirmed_fake.append({
                    'image_path': f['image_path'],
                    'scores': f['system_scores']
                })
        
        return confirmed_real, confirmed_fake


# Flask routes for feedback system

def add_feedback_routes(app):
    """Add feedback routes to Flask app"""
    
    @app.route('/api/feedback', methods=['POST'])
    def submit_feedback():
        """
        User submits feedback on a prediction
        
        POST data:
        {
            "image_path": "...",
            "prediction_data": {...},
            "user_validation": "correct/incorrect/unsure",
            "user_comments": "..."
        }
        """
        from flask import request
        
        data = request.json
        
        try:
            filepath = FeedbackCollector.save_feedback(
                image_path=data.get('image_path'),
                prediction_data=data.get('prediction_data', {}),
                user_validation=data.get('user_validation'),
                user_comments=data.get('user_comments', '')
            )
            
            return jsonify({
                'success': True,
                'message': 'Thank you for your feedback!',
                'feedback_id': os.path.basename(filepath)
            })
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/feedback/stats', methods=['GET'])
    def feedback_stats():
        """Get feedback statistics (admin only)"""
        
        try:
            stats = FeedbackCollector.analyze_feedback()
            return jsonify(stats)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/recalibrate', methods=['POST'])
    def trigger_recalibration():
        """
        Trigger system recalibration using feedback data (admin only)
        
        This should be protected with authentication in production
        """
        
        try:
            confirmed_real, confirmed_fake = FeedbackCollector.get_retraining_data()
            
            if len(confirmed_real) < 10 or len(confirmed_fake) < 10:
                return jsonify({
                    'success': False,
                    'message': 'Insufficient validated data for recalibration',
                    'real_samples': len(confirmed_real),
                    'fake_samples': len(confirmed_fake),
                    'required': 10
                })
            
            # TODO: Run calibration script with this data
            # For now, just return the data counts
            
            return jsonify({
                'success': True,
                'message': 'Recalibration triggered',
                'real_samples': len(confirmed_real),
                'fake_samples': len(confirmed_fake)
            })
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500


# Usage example for app.py:
# 
# from feedback_system import add_feedback_routes
# 
# app = Flask(__name__)
# add_feedback_routes(app)