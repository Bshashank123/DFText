"""
AI Handwriting Forensics System - FINAL OPTIMIZED VERSION
All improvements integrated
"""

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image

from pipeline.preprocess import preprocess_image
from pipeline.ocr import extract_text
from pipeline.image_forensics import analyze_image_forensics
from pipeline.advanced_forensics import analyze_advanced_forensics
from pipeline.stylometry import analyze_stylometry
from pipeline.ai_text import analyze_ai_text
from pipeline.decision import make_decision
from pipeline.explain import generate_explanation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Could not read image'}), 400
        
        # Step 1: Preprocess
        normalized_img = preprocess_image(img)
        
        # Step 2: OCR
        raw_text = extract_text(normalized_img)
        
        # Step 3: Parallel forensic analysis
        image_scores = analyze_image_forensics(normalized_img)
        advanced_scores = analyze_advanced_forensics(normalized_img)
        stylometry_score = analyze_stylometry(raw_text)
        ai_text_score = analyze_ai_text(raw_text)
        
        # Merge advanced scores
        image_scores['advanced'] = advanced_scores['advanced_forensics']
        image_scores['cfa'] = advanced_scores['cfa']
        image_scores['jpeg'] = advanced_scores['jpeg']
        image_scores['edge'] = advanced_scores['edge']
        image_scores['pixel'] = advanced_scores['pixel']
        
        # Combine traditional + advanced image forensics
        image_scores['image_forensics'] = (
            0.70 * image_scores['image_forensics'] +
            0.30 * advanced_scores['advanced_forensics']
        )
        
        # Step 4: Decision fusion (using calibrated weights)
        verdict, confidence, all_scores = make_decision(
            image_scores, 
            stylometry_score, 
            ai_text_score,
            text_length=len(raw_text)
        )
        
        # Step 5: Generate explanation
        explanation = generate_explanation(
            verdict,
            confidence,
            all_scores,
            raw_text
        )
        
        # Cleanup
        os.remove(filepath)
        
        return jsonify({
            'verdict': verdict,
            'confidence': confidence,
            'scores': all_scores,
            'explanation': explanation,
            'text_length': len(raw_text),
            'raw_text_preview': raw_text[:200] if raw_text else 'No text detected'
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/debug-analyze', methods=['POST'])
def debug_analyze():
    """Detailed debugging analysis showing all intermediate scores"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        # Save and load image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img = cv2.imread(filepath)
        normalized_img = preprocess_image(img)
        raw_text = extract_text(normalized_img)
        
        # Get detailed breakdowns from NEW stroke module
        from pipeline.stroke_analysis import (
            analyze_ink_texture, analyze_stroke_pressure, analyze_stroke_direction
        )
        from pipeline.image_forensics import (
            extract_noise_residual, noise_variance_map, fft_energy_ratio,
            extract_background_regions, texture_entropy
        )
        
        # NOISE DETAILS
        noise = extract_noise_residual(normalized_img)
        variances = noise_variance_map(noise)
        variance_std = np.std(variances) if len(variances) > 0 else 0.0
        variance_mean = np.mean(variances) if len(variances) > 0 else 0.0
        freq_ratio = fft_energy_ratio(noise)
        variance_cv = variance_std / (variance_mean + 1e-6)
        
        # STROKE DETAILS (NEW MODULE)
        texture_score = analyze_ink_texture(normalized_img)
        pressure_score = analyze_stroke_pressure(normalized_img)
        direction_score = analyze_stroke_direction(normalized_img)
        
        # PAPER DETAILS
        bg_pixels = extract_background_regions(normalized_img)
        paper_entropy = texture_entropy(bg_pixels) if len(bg_pixels) > 0 else 0.0
        paper_variance = np.var(bg_pixels) if len(bg_pixels) > 0 else 0.0
        
        # STYLOMETRY DETAILS
        from pipeline.stylometry import (
            parse_text, sentence_length_stats, lexical_diversity,
            function_word_ratio, punctuation_stats, word_repetition_rate,
            detect_gemini_patterns
        )
        
        sentences, words = parse_text(raw_text)
        mean_len, std_len = sentence_length_stats(sentences)
        lex_div = lexical_diversity(words)
        func_ratio = function_word_ratio(words)
        punct_ratio, cap_ratio = punctuation_stats(raw_text)
        rep_rate = word_repetition_rate(words)
        gemini_score = detect_gemini_patterns(raw_text)
        
        # Run full analysis
        image_scores = analyze_image_forensics(normalized_img)
        advanced_scores = analyze_advanced_forensics(normalized_img)
        stylometry_score = analyze_stylometry(raw_text)
        ai_text_score = analyze_ai_text(raw_text)
        
        # Merge scores
        image_scores['advanced'] = advanced_scores['advanced_forensics']
        image_scores['image_forensics'] = (
            0.70 * image_scores['image_forensics'] +
            0.30 * advanced_scores['advanced_forensics']
        )
        
        verdict, confidence, all_scores = make_decision(
            image_scores, stylometry_score, ai_text_score,
            text_length=len(raw_text)
        )
        
        # Cleanup
        os.remove(filepath)
        
        # Return comprehensive debug info
        return jsonify({
            'verdict': verdict,
            'confidence': confidence,
            'final_score': all_scores['final'],
            
            'noise_details': {
                'variance_std': float(variance_std),
                'variance_mean': float(variance_mean),
                'variance_cv': float(variance_cv),
                'freq_ratio': float(freq_ratio),
                'final_noise_score': float(image_scores['noise']),
                'interpretation': f'Real: var_std>25, freq>1.2, CV>0.8 | Yours: {variance_std:.1f}, {freq_ratio:.2f}, {variance_cv:.2f}'
            },
            
            'stroke_details': {
                'texture_score': float(texture_score),
                'pressure_score': float(pressure_score),
                'direction_score': float(direction_score),
                'final_stroke_score': float(image_scores['stroke']),
                'interpretation': 'NEW MODULE: texture, pressure, direction analysis'
            },
            
            'paper_details': {
                'paper_entropy': float(paper_entropy),
                'paper_variance': float(paper_variance),
                'num_bg_pixels': len(bg_pixels),
                'final_paper_score': float(image_scores['paper']),
                'interpretation': f'Real: entropy 2-3.5, var 60-180 | Yours: {paper_entropy:.2f}, {paper_variance:.1f}'
            },
            
            'advanced_forensics': {
                'cfa_score': float(advanced_scores['cfa']),
                'jpeg_score': float(advanced_scores['jpeg']),
                'edge_score': float(advanced_scores['edge']),
                'pixel_score': float(advanced_scores['pixel']),
                'advanced_total': float(advanced_scores['advanced_forensics']),
                'interpretation': 'Camera artifacts, JPEG patterns, edge coherence'
            },
            
            'stylometry_details': {
                'sentence_count': len(sentences),
                'word_count': len(words),
                'mean_sentence_length': float(mean_len),
                'std_sentence_length': float(std_len),
                'lexical_diversity': float(lex_div),
                'function_word_ratio': float(func_ratio),
                'punct_ratio': float(punct_ratio),
                'repetition_rate': float(rep_rate),
                'gemini_score': float(gemini_score),
                'final_stylometry_score': float(stylometry_score),
                'interpretation': f'Gemini markers: {gemini_score:.2f} | Low std_len + high formality = AI'
            },
            
            'ai_text_details': {
                'text_length': len(raw_text),
                'final_ai_score': float(ai_text_score),
                'interpretation': 'Improved heuristics: formal markers, contractions, sentence structure'
            },
            
            'module_scores': {
                'noise': float(image_scores['noise']),
                'stroke': float(image_scores['stroke']),
                'paper': float(image_scores['paper']),
                'advanced': float(advanced_scores['advanced_forensics']),
                'image_forensics': float(image_scores['image_forensics']),
                'stylometry': float(stylometry_score),
                'ai_text': float(ai_text_score)
            },
            
            'weights_used': all_scores.get('weights_used', {}),
            'thresholds_used': all_scores.get('thresholds_used', {}),
            'module_agreement': all_scores.get('module_agreement', 0.0),
            
            'calibration_notes': {
                'paper_weight': '50% - BEST MODULE (0.145 separation)',
                'noise_weight': '20% - Second best (0.106 separation)',
                'stroke_weight': '15% - NEW MODULE (testing)',
                'style_weight': '10% - Weak on your data (0.014 sep)',
                'ai_weight': '5% - Nearly useless (0.000 sep)',
                'optimal_threshold': '0.460 from calibration'
            },
            
            'raw_text_preview': raw_text[:500]
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/compare', methods=['POST'])
def compare_images():
    """Compare two images side by side"""
    
    if 'real' not in request.files or 'fake' not in request.files:
        return jsonify({'error': 'Need both real and fake images'}), 400
    
    real_file = request.files['real']
    fake_file = request.files['fake']
    
    def analyze_single(file):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img = cv2.imread(filepath)
        normalized_img = preprocess_image(img)
        raw_text = extract_text(normalized_img)
        
        image_scores = analyze_image_forensics(normalized_img)
        advanced_scores = analyze_advanced_forensics(normalized_img)
        stylometry_score = analyze_stylometry(raw_text)
        ai_text_score = analyze_ai_text(raw_text)
        
        image_scores['advanced'] = advanced_scores['advanced_forensics']
        image_scores['image_forensics'] = (
            0.70 * image_scores['image_forensics'] +
            0.30 * advanced_scores['advanced_forensics']
        )
        
        verdict, confidence, all_scores = make_decision(
            image_scores, stylometry_score, ai_text_score,
            text_length=len(raw_text)
        )
        
        os.remove(filepath)
        
        return {
            'verdict': verdict,
            'scores': all_scores
        }
    
    real_result = analyze_single(real_file)
    fake_result = analyze_single(fake_file)
    
    differences = {
        'noise': abs(real_result['scores']['noise'] - fake_result['scores']['noise']),
        'stroke': abs(real_result['scores']['stroke'] - fake_result['scores']['stroke']),
        'paper': abs(real_result['scores']['paper'] - fake_result['scores']['paper']),
        'advanced': abs(real_result['scores'].get('advanced', 0) - fake_result['scores'].get('advanced', 0)),
        'stylometry': abs(real_result['scores']['stylometry'] - fake_result['scores']['stylometry']),
        'ai_text': abs(real_result['scores']['ai_text'] - fake_result['scores']['ai_text']),
        'final': abs(real_result['scores']['final'] - fake_result['scores']['final'])
    }
    
    return jsonify({
        'real': real_result,
        'fake': fake_result,
        'differences': differences,
        'strongest_signal': max(differences.items(), key=lambda x: x[1])[0]
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)