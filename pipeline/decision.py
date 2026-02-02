"""
Decision Fusion Module - CALIBRATED FOR YOUR DATASET
Based on calibration results: optimal threshold 0.460
Paper module is best performer (0.145 separation)
"""

import numpy as np

def calculate_heuristic_adjustments(text_length, image_quality=1.0, ocr_confidence=1.0):
    """Apply penalties for low-quality inputs"""
    penalties = []
    
    if text_length < 50:
        penalties.append(0.3)
    elif text_length < 150:
        penalties.append(0.15)
    
    if image_quality < 0.5:
        penalties.append(0.2)
    
    if ocr_confidence < 0.6:
        penalties.append(0.15)
    
    if penalties:
        return np.mean(penalties)
    
    return 0.0

def make_decision(image_scores, stylometry_score, ai_text_score, 
                  text_length=100, image_quality=1.0, ocr_confidence=1.0):
    """
    CALIBRATED decision engine based on your test results
    
    Key findings from calibration:
    - Paper module: BEST (0.145 separation)
    - Noise module: OK (0.106 separation)
    - Everything else: WEAK (<0.02 separation)
    
    Strategy: Weight paper heavily, use noise as support, minimize others
    """
    
    adjustments = calculate_heuristic_adjustments(
        text_length, image_quality, ocr_confidence
    )
    
    # OPTIMIZED WEIGHTS based on module performance
    # Paper is king (0.145 sep), noise is decent (0.106 sep), rest are weak
    paper_weight = 0.50      # DOUBLED - best module
    noise_weight = 0.20      # Second best
    stroke_weight = 0.15     # New module, give it a chance
    style_weight = 0.10      # Weak on your data (0.014 sep)
    ai_weight = 0.05         # Nearly useless (0.000 sep)
    
    # For very short text, shift weight from text modules to image
    if text_length < 100:
        paper_weight = 0.55
        noise_weight = 0.25
        stroke_weight = 0.15
        style_weight = 0.05
        ai_weight = 0.00
    
    # For poor image quality, shift to text
    if image_quality < 0.6:
        paper_weight = 0.30
        noise_weight = 0.15
        stroke_weight = 0.15
        style_weight = 0.25
        ai_weight = 0.15
    
    # CALCULATE FINAL SCORE
    # Use paper score directly plus weighted average of others
    final_score = (
        paper_weight * image_scores['paper'] +
        noise_weight * image_scores['noise'] +
        stroke_weight * image_scores['stroke'] +
        style_weight * stylometry_score +
        ai_weight * ai_text_score
    )
    
    # Add penalty for low quality
    final_score = final_score + (0.10 * adjustments)
    final_score = max(0.0, min(1.0, final_score))
    
    # CALIBRATED THRESHOLDS
    # Your optimal threshold: 0.460
    # But we'll use slightly more conservative ranges
    HUMAN_THRESHOLD = 0.40      # Below this = likely human
    SUSPICIOUS_THRESHOLD = 0.50  # Above this = likely AI
    
    if final_score < HUMAN_THRESHOLD:
        verdict = "Likely Human"
        verdict_color = "green"
    elif final_score < SUSPICIOUS_THRESHOLD:
        verdict = "Suspicious"
        verdict_color = "yellow"
    else:
        verdict = "AI-Assisted Likely"
        verdict_color = "red"
    
    # MODULE AGREEMENT - only count modules with reasonable separation
    module_votes = []
    
    # Paper vote (strongest signal)
    if image_scores['paper'] > 0.35:  # Calibrated threshold
        module_votes.append('synthetic')
    else:
        module_votes.append('real')
    
    # Noise vote (second strongest)
    if image_scores['noise'] > 0.15:  # Calibrated threshold
        module_votes.append('synthetic')
    else:
        module_votes.append('real')
    
    # Stroke vote (new module)
    if image_scores['stroke'] > 0.5:
        module_votes.append('synthetic')
    else:
        module_votes.append('real')
    
    # Don't count weak modules (stylometry, ai_text) in agreement
    
    synthetic_votes = module_votes.count('synthetic')
    agreement = max(synthetic_votes, 3 - synthetic_votes) / 3.0
    
    # CONFIDENCE CALCULATION
    if verdict == "Likely Human":
        # Distance from human threshold
        distance = HUMAN_THRESHOLD - final_score
        base_confidence = int(min(95, 60 + (distance / HUMAN_THRESHOLD) * 35))
    elif verdict == "Suspicious":
        # In middle zone - lower confidence
        center = (HUMAN_THRESHOLD + SUSPICIOUS_THRESHOLD) / 2
        distance_from_center = abs(final_score - center)
        base_confidence = int(50 + distance_from_center * 40)
    else:  # AI-Assisted Likely
        # Distance from AI threshold
        distance = final_score - SUSPICIOUS_THRESHOLD
        max_distance = 1.0 - SUSPICIOUS_THRESHOLD
        base_confidence = int(min(95, 60 + (distance / max_distance) * 35))
    
    # Adjust by module agreement
    confidence = int(base_confidence * (0.6 + 0.4 * agreement))
    confidence = max(40, min(99, confidence))
    
    # Compile all scores
    all_scores = {
        'final': final_score,
        'stylometry': stylometry_score,
        'ai_text': ai_text_score,
        'image_forensics': image_scores['image_forensics'],
        'noise': image_scores['noise'],
        'stroke': image_scores['stroke'],
        'paper': image_scores['paper'],
        'advanced': image_scores.get('advanced', 0.0),
        'adjustments': adjustments,
        'verdict': verdict,
        'verdict_color': verdict_color,
        'confidence': confidence,
        'weights_used': {
            'paper': paper_weight,
            'noise': noise_weight,
            'stroke': stroke_weight,
            'stylometry': style_weight,
            'ai_text': ai_weight
        },
        'module_agreement': agreement,
        'thresholds_used': {
            'human': HUMAN_THRESHOLD,
            'suspicious': SUSPICIOUS_THRESHOLD
        }
    }
    
    return verdict, confidence, all_scores

def get_threshold_info():
    """Return calibrated threshold information"""
    return {
        'human_threshold': 0.40,
        'suspicious_threshold': 0.50,
        'weights': {
            'paper': 0.50,
            'noise': 0.20,
            'stroke': 0.15,
            'stylometry': 0.10,
            'ai_text': 0.05
        }
    }

def calibrate_thresholds(real_scores, fake_scores):
    """Calibration helper"""
    all_scores = sorted(real_scores + fake_scores)
    
    best_threshold = 0.5
    best_separation = 0
    
    for threshold in np.arange(0.2, 0.8, 0.01):
        real_correct = sum(1 for s in real_scores if s < threshold)
        fake_correct = sum(1 for s in fake_scores if s >= threshold)
        
        total_correct = real_correct + fake_correct
        total_samples = len(real_scores) + len(fake_scores)
        
        accuracy = total_correct / total_samples
        
        if accuracy > best_separation:
            best_separation = accuracy
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Accuracy: {best_separation*100:.1f}%")
    print(f"\nReal scores: mean={np.mean(real_scores):.3f}, std={np.std(real_scores):.3f}")
    print(f"Fake scores: mean={np.mean(fake_scores):.3f}, std={np.std(fake_scores):.3f}")
    
    return best_threshold