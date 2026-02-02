"""
CALIBRATION SCRIPT
Run this to test your system and find optimal parameters

Usage:
    python test_and_calibrate.py --real-dir ./real_images --fake-dir ./fake_images
"""

import os
import cv2
import numpy as np
import json
import sys

# Add pipeline to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.preprocess import preprocess_image
from pipeline.ocr import extract_text
from pipeline.image_forensics import analyze_image_forensics
from pipeline.stylometry import analyze_stylometry
from pipeline.ai_text import analyze_ai_text
from pipeline.decision import make_decision

def analyze_directory(directory):
    """Analyze all images in a directory"""
    results = []
    
    print(f"\nProcessing directory: {directory}")
    print("-" * 60)
    
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not files:
        print(f"WARNING: No image files found in {directory}")
        return results
    
    for idx, filename in enumerate(files, 1):
        filepath = os.path.join(directory, filename)
        print(f"[{idx}/{len(files)}] {filename}...", end=" ")
        
        try:
            img = cv2.imread(filepath)
            if img is None:
                print("SKIP (unreadable)")
                continue
                
            normalized_img = preprocess_image(img)
            raw_text = extract_text(normalized_img)
            
            image_scores = analyze_image_forensics(normalized_img)
            stylometry_score = analyze_stylometry(raw_text)
            ai_text_score = analyze_ai_text(raw_text)
            
            verdict, confidence, all_scores = make_decision(
                image_scores, stylometry_score, ai_text_score,
                text_length=len(raw_text)
            )
            
            results.append({
                'filename': filename,
                'final_score': all_scores['final'],
                'noise': all_scores['noise'],
                'stroke': all_scores['stroke'],
                'paper': all_scores['paper'],
                'image_forensics': all_scores['image_forensics'],
                'stylometry': all_scores['stylometry'],
                'ai_text': all_scores['ai_text'],
                'verdict': verdict,
                'confidence': confidence,
                'text_length': len(raw_text)
            })
            
            print(f"OK (score: {all_scores['final']:.3f})")
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    return results

def print_statistics(real_results, fake_results):
    """Print detailed statistics and recommendations"""
    
    if not real_results or not fake_results:
        print("\n" + "="*60)
        print("ERROR: Need both real and fake samples to calibrate!")
        print("="*60)
        return None
    
    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)
    
    # Extract scores
    real_finals = [r['final_score'] for r in real_results]
    fake_finals = [r['final_score'] for r in fake_results]
    
    print(f"\n📊 DATASET OVERVIEW")
    print(f"  Real images: {len(real_results)}")
    print(f"  Fake images: {len(fake_results)}")
    
    print(f"\n📈 FINAL SCORE DISTRIBUTION")
    print(f"\n  REAL IMAGES:")
    print(f"    Mean:  {np.mean(real_finals):.3f}")
    print(f"    Std:   {np.std(real_finals):.3f}")
    print(f"    Range: [{np.min(real_finals):.3f}, {np.max(real_finals):.3f}]")
    
    print(f"\n  FAKE IMAGES:")
    print(f"    Mean:  {np.mean(fake_finals):.3f}")
    print(f"    Std:   {np.std(fake_finals):.3f}")
    print(f"    Range: [{np.min(fake_finals):.3f}, {np.max(fake_finals):.3f}]")
    
    # Separation analysis
    separation = abs(np.mean(fake_finals) - np.mean(real_finals))
    print(f"\n🎯 SEPARATION: {separation:.3f}")
    
    if separation < 0.15:
        print("  ❌ CRITICAL: Very poor separation!")
        print("  → Scores overlap heavily - system cannot discriminate")
        print("  → ACTION: Check if images are truly different, adjust weights")
    elif separation < 0.25:
        print("  ⚠️  WARNING: Moderate separation")
        print("  → May have classification errors")
        print("  → ACTION: Consider adjusting thresholds or weights")
    elif separation < 0.40:
        print("  ✓ Good separation")
    else:
        print("  ✅ Excellent separation!")
    
    # Module-wise comparison
    print("\n" + "-"*60)
    print("🔬 MODULE PERFORMANCE ANALYSIS")
    print("-"*60)
    
    module_performance = {}
    
    for module in ['noise', 'stroke', 'paper', 'image_forensics', 'stylometry', 'ai_text']:
        real_mod = [r[module] for r in real_results]
        fake_mod = [r[module] for r in fake_results]
        mod_sep = abs(np.mean(fake_mod) - np.mean(real_mod))
        
        module_performance[module] = {
            'real_mean': np.mean(real_mod),
            'fake_mean': np.mean(fake_mod),
            'separation': mod_sep
        }
        
        print(f"\n  {module.upper()}")
        print(f"    Real: {np.mean(real_mod):.3f} ± {np.std(real_mod):.3f}")
        print(f"    Fake: {np.mean(fake_mod):.3f} ± {np.std(fake_mod):.3f}")
        print(f"    Separation: {mod_sep:.3f}", end="")
        
        if mod_sep > 0.3:
            print(" ✅ STRONG - This module works well!")
        elif mod_sep > 0.15:
            print(" ✓ Moderate")
        else:
            print(" ❌ Weak - This module needs improvement")
    
    # Find best performing module
    best_module = max(module_performance.items(), key=lambda x: x[1]['separation'])
    print(f"\n  🏆 BEST MODULE: {best_module[0]} (separation: {best_module[1]['separation']:.3f})")
    
    # Find optimal threshold
    print("\n" + "-"*60)
    print("🎚️  THRESHOLD OPTIMIZATION")
    print("-"*60)
    
    best_threshold = 0.5
    best_accuracy = 0
    
    for threshold in np.arange(0.2, 0.8, 0.01):
        real_correct = sum(1 for s in real_finals if s < threshold)
        fake_correct = sum(1 for s in fake_finals if s >= threshold)
        accuracy = (real_correct + fake_correct) / (len(real_finals) + len(fake_finals))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    print(f"\n  Optimal threshold: {best_threshold:.3f}")
    print(f"  Accuracy at optimal: {best_accuracy*100:.1f}%")
    
    # Test current thresholds
    current_thresholds = [0.35, 0.70]
    print(f"\n  Current thresholds: {current_thresholds}")
    
    # Classification with current thresholds
    real_classified = []
    for s in real_finals:
        if s < 0.35:
            real_classified.append('human')
        elif s < 0.70:
            real_classified.append('suspicious')
        else:
            real_classified.append('ai')
    
    fake_classified = []
    for s in fake_finals:
        if s < 0.35:
            fake_classified.append('human')
        elif s < 0.70:
            fake_classified.append('suspicious')
        else:
            fake_classified.append('ai')
    
    print(f"\n  📊 REAL IMAGES CLASSIFIED AS:")
    print(f"    Human:      {real_classified.count('human'):2d} ({real_classified.count('human')/len(real_classified)*100:5.1f}%) ✓ Should be high")
    print(f"    Suspicious: {real_classified.count('suspicious'):2d} ({real_classified.count('suspicious')/len(real_classified)*100:5.1f}%)")
    print(f"    AI:         {real_classified.count('ai'):2d} ({real_classified.count('ai')/len(real_classified)*100:5.1f}%) ✗ Should be low")
    
    print(f"\n  📊 FAKE IMAGES CLASSIFIED AS:")
    print(f"    Human:      {fake_classified.count('human'):2d} ({fake_classified.count('human')/len(fake_classified)*100:5.1f}%) ✗ Should be low")
    print(f"    Suspicious: {fake_classified.count('suspicious'):2d} ({fake_classified.count('suspicious')/len(fake_classified)*100:5.1f}%)")
    print(f"    AI:         {fake_classified.count('ai'):2d} ({fake_classified.count('ai')/len(fake_classified)*100:5.1f}%) ✓ Should be high")
    
    # Overall accuracy
    real_correct = real_classified.count('human') + real_classified.count('suspicious')
    fake_correct = fake_classified.count('ai') + fake_classified.count('suspicious')
    overall_accuracy = (real_correct + fake_correct) / (len(real_classified) + len(fake_classified))
    
    print(f"\n  Overall accuracy: {overall_accuracy*100:.1f}%")
    
    # RECOMMENDATIONS
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS")
    print("="*60)
    
    if separation < 0.15:
        print("\n  ⚠️  CRITICAL ACTIONS NEEDED:")
        print("  1. Verify your test images are actually different (real vs AI)")
        print("  2. Increase weight on best-performing module")
        print(f"     → Best module: {best_module[0]} with {best_module[1]['separation']:.3f} separation")
        print("  3. Consider collecting more diverse samples")
    
    if best_module[0] in ['stylometry', 'ai_text']:
        print("\n  📝 Text-based modules work best:")
        print("  → Increase stylometry weight to 0.45")
        print("  → Increase ai_text weight to 0.35")
        print("  → Reduce image_forensics to 0.15")
    
    if best_module[0] in ['noise', 'stroke', 'paper']:
        print("\n  🖼️  Image-based modules work best:")
        print("  → Increase image_forensics weight to 0.40")
        print("  → Reduce stylometry/ai_text weights")
    
    if abs(best_threshold - 0.35) > 0.10:
        print(f"\n  🎚️  Consider adjusting human threshold from 0.35 to {best_threshold:.2f}")
    
    print("\n" + "="*60)
    
    return {
        'optimal_threshold': best_threshold,
        'optimal_accuracy': best_accuracy,
        'separation': separation,
        'real_mean': np.mean(real_finals),
        'fake_mean': np.mean(fake_finals),
        'best_module': best_module[0],
        'module_performance': module_performance
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Calibrate AI Handwriting Forensics System')
    parser.add_argument('--real-dir', required=True, help='Directory with real handwriting images')
    parser.add_argument('--fake-dir', required=True, help='Directory with AI-generated images')
    parser.add_argument('--output', default='calibration_results.json', help='Output JSON file')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.isdir(args.real_dir):
        print(f"ERROR: Real directory not found: {args.real_dir}")
        sys.exit(1)
    
    if not os.path.isdir(args.fake_dir):
        print(f"ERROR: Fake directory not found: {args.fake_dir}")
        sys.exit(1)
    
    print("="*60)
    print("AI HANDWRITING FORENSICS - CALIBRATION")
    print("="*60)
    
    print("\n🔍 Analyzing REAL images...")
    real_results = analyze_directory(args.real_dir)
    
    print("\n🔍 Analyzing FAKE images...")
    fake_results = analyze_directory(args.fake_dir)
    
    # Print statistics
    stats = print_statistics(real_results, fake_results)
    
    if stats:
        # Save results
        output = {
            'real_results': real_results,
            'fake_results': fake_results,
            'statistics': stats
        }
        
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to: {args.output}")
    else:
        print("\n❌ Calibration failed - insufficient data")