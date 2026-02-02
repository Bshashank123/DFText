"""
MODULE-SPECIFIC CALIBRATION
Tune each forensic module independently, then optimize fusion weights

This is IDEA 2 - Train individual modules separately
"""

import os
import sys
import cv2
import numpy as np
import json
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.preprocess import preprocess_image
from pipeline.ocr import extract_text
from pipeline.image_forensics import analyze_image_forensics
from pipeline.stylometry import analyze_stylometry
from pipeline.ai_text import analyze_ai_text

class ModuleCalibrator:
    """Calibrate individual forensic modules"""
    
    def __init__(self, real_dir, fake_dir):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.results = {}
    
    def analyze_single_module(self, module_name, score_func):
        """
        Analyze a single module's performance
        
        Args:
            module_name: Name of module
            score_func: Function that takes (img, text) and returns score
        """
        print(f"\n{'='*60}")
        print(f"CALIBRATING: {module_name.upper()}")
        print(f"{'='*60}")
        
        real_scores = []
        fake_scores = []
        
        # Process real images
        print("\nProcessing REAL images...")
        for filename in os.listdir(self.real_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            filepath = os.path.join(self.real_dir, filename)
            try:
                img = cv2.imread(filepath)
                normalized = preprocess_image(img)
                text = extract_text(normalized)
                
                score = score_func(normalized, text)
                real_scores.append(score)
                print(f"  {filename}: {score:.3f}")
            except Exception as e:
                print(f"  {filename}: ERROR - {e}")
        
        # Process fake images
        print("\nProcessing FAKE images...")
        for filename in os.listdir(self.fake_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            filepath = os.path.join(self.fake_dir, filename)
            try:
                img = cv2.imread(filepath)
                normalized = preprocess_image(img)
                text = extract_text(normalized)
                
                score = score_func(normalized, text)
                fake_scores.append(score)
                print(f"  {filename}: {score:.3f}")
            except Exception as e:
                print(f"  {filename}: ERROR - {e}")
        
        # Calculate statistics
        if not real_scores or not fake_scores:
            print(f"\n⚠️  WARNING: Insufficient data for {module_name}")
            return None
        
        real_mean = np.mean(real_scores)
        fake_mean = np.mean(fake_scores)
        separation = abs(fake_mean - real_mean)
        
        # Find optimal threshold using ROC
        labels = [0] * len(real_scores) + [1] * len(fake_scores)
        scores = real_scores + fake_scores
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        # Optimal threshold: maximize (TPR - FPR)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        
        # Calculate accuracy at optimal threshold
        real_correct = sum(1 for s in real_scores if s < optimal_threshold)
        fake_correct = sum(1 for s in fake_scores if s >= optimal_threshold)
        accuracy = (real_correct + fake_correct) / (len(real_scores) + len(fake_scores))
        
        print(f"\n{'='*60}")
        print(f"RESULTS FOR {module_name.upper()}")
        print(f"{'='*60}")
        print(f"\nSeparation: {separation:.3f}")
        print(f"ROC AUC: {roc_auc:.3f}")
        print(f"Optimal Threshold: {optimal_threshold:.3f}")
        print(f"Accuracy at Optimal: {accuracy*100:.1f}%")
        print(f"TPR: {optimal_tpr:.3f} | FPR: {optimal_fpr:.3f}")
        print(f"\nReal: {real_mean:.3f} ± {np.std(real_scores):.3f}")
        print(f"Fake: {fake_mean:.3f} ± {np.std(fake_scores):.3f}")
        
        # Performance rating
        if separation > 0.4:
            rating = "EXCELLENT ✅"
        elif separation > 0.25:
            rating = "GOOD ✓"
        elif separation > 0.15:
            rating = "MODERATE ⚠️"
        else:
            rating = "WEAK ❌"
        
        print(f"\nPerformance: {rating}")
        
        result = {
            'module': module_name,
            'separation': separation,
            'roc_auc': roc_auc,
            'optimal_threshold': optimal_threshold,
            'optimal_accuracy': accuracy,
            'optimal_tpr': optimal_tpr,
            'optimal_fpr': optimal_fpr,
            'real_mean': real_mean,
            'real_std': np.std(real_scores),
            'fake_mean': fake_mean,
            'fake_std': np.std(fake_scores),
            'real_scores': real_scores,
            'fake_scores': fake_scores,
            'rating': rating
        }
        
        return result
    
    def calibrate_all(self):
        """Calibrate all modules"""
        
        # Define module score functions
        modules = {
            'noise': lambda img, text: analyze_image_forensics(img)['noise'],
            'stroke': lambda img, text: analyze_image_forensics(img)['stroke'],
            'paper': lambda img, text: analyze_image_forensics(img)['paper'],
            'image_forensics': lambda img, text: analyze_image_forensics(img)['image_forensics'],
            'stylometry': lambda img, text: analyze_stylometry(text),
            'ai_text': lambda img, text: analyze_ai_text(text)
        }
        
        results = {}
        for name, func in modules.items():
            result = self.analyze_single_module(name, func)
            if result:
                results[name] = result
        
        self.results = results
        return results
    
    def recommend_weights(self):
        """Recommend optimal fusion weights based on performance"""
        
        if not self.results:
            print("No results to analyze. Run calibrate_all() first.")
            return
        
        print(f"\n{'='*60}")
        print("WEIGHT RECOMMENDATIONS")
        print(f"{'='*60}")
        
        # Sort by separation
        sorted_modules = sorted(
            self.results.items(),
            key=lambda x: x[1]['separation'],
            reverse=True
        )
        
        print("\nModule Performance Ranking:")
        for i, (name, data) in enumerate(sorted_modules, 1):
            print(f"  {i}. {name.upper()}: {data['separation']:.3f} separation")
        
        # Calculate weights proportional to separation
        total_separation = sum(data['separation'] for _, data in sorted_modules)
        
        if total_separation == 0:
            print("\n⚠️  WARNING: No module shows separation!")
            return
        
        print("\n📊 RECOMMENDED WEIGHTS:")
        recommended_weights = {}
        
        for name, data in sorted_modules:
            weight = data['separation'] / total_separation
            recommended_weights[name] = weight
            print(f"  {name}: {weight:.2f} ({weight*100:.0f}%)")
        
        # Group into categories
        image_modules = ['noise', 'stroke', 'paper']
        text_modules = ['stylometry', 'ai_text']
        
        image_weight = sum(recommended_weights.get(m, 0) for m in image_modules)
        text_weight = sum(recommended_weights.get(m, 0) for m in text_modules)
        
        print(f"\n📈 CATEGORY WEIGHTS:")
        print(f"  Image-based: {image_weight:.2f} ({image_weight*100:.0f}%)")
        print(f"  Text-based:  {text_weight:.2f} ({text_weight*100:.0f}%)")
        
        return recommended_weights
    
    def save_results(self, output_file='module_calibration.json'):
        """Save calibration results"""
        
        if not self.results:
            print("No results to save.")
            return
        
        # Convert numpy types to native Python
        save_data = {}
        for name, data in self.results.items():
            save_data[name] = {
                'separation': float(data['separation']),
                'roc_auc': float(data['roc_auc']),
                'optimal_threshold': float(data['optimal_threshold']),
                'optimal_accuracy': float(data['optimal_accuracy']),
                'real_mean': float(data['real_mean']),
                'fake_mean': float(data['fake_mean']),
                'rating': data['rating']
            }
        
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def plot_distributions(self, output_dir='plots'):
        """Plot score distributions for each module"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, data in self.results.items():
            plt.figure(figsize=(10, 6))
            
            # Plot histograms
            plt.hist(data['real_scores'], bins=20, alpha=0.6, label='Real', color='green')
            plt.hist(data['fake_scores'], bins=20, alpha=0.6, label='Fake', color='red')
            
            # Plot optimal threshold
            plt.axvline(data['optimal_threshold'], color='blue', linestyle='--', 
                       label=f"Optimal Threshold: {data['optimal_threshold']:.3f}")
            
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.title(f'{name.upper()} - Score Distribution\nSeparation: {data["separation"]:.3f} | AUC: {data["roc_auc"]:.3f}')
            plt.legend()
            plt.grid(alpha=0.3)
            
            output_path = os.path.join(output_dir, f'{name}_distribution.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Calibrate individual forensic modules')
    parser.add_argument('--real-dir', required=True, help='Directory with real images')
    parser.add_argument('--fake-dir', required=True, help='Directory with fake images')
    parser.add_argument('--output', default='module_calibration.json', help='Output file')
    parser.add_argument('--plots', action='store_true', help='Generate distribution plots')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.real_dir):
        print(f"ERROR: Real directory not found: {args.real_dir}")
        sys.exit(1)
    
    if not os.path.isdir(args.fake_dir):
        print(f"ERROR: Fake directory not found: {args.fake_dir}")
        sys.exit(1)
    
    print("="*60)
    print("MODULE-SPECIFIC CALIBRATION")
    print("="*60)
    print(f"\nReal images: {args.real_dir}")
    print(f"Fake images: {args.fake_dir}")
    
    # Run calibration
    calibrator = ModuleCalibrator(args.real_dir, args.fake_dir)
    calibrator.calibrate_all()
    
    # Recommendations
    weights = calibrator.recommend_weights()
    
    # Save results
    calibrator.save_results(args.output)
    
    # Generate plots if requested
    if args.plots:
        print("\n📊 Generating distribution plots...")
        calibrator.plot_distributions()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Update decision.py with recommended weights")
    print("2. Update thresholds for each module")
    print("3. Test on validation set")
    print("="*60)


if __name__ == '__main__':
    main()