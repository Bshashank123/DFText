"""
Quick test script for single image analysis
Shows detailed breakdown of all scores

Usage:
    python quick_test.py image.jpg
"""

import sys
import os
import cv2
import json

from pipeline.preprocess import preprocess_image
from pipeline.ocr import extract_text
from pipeline.image_forensics import (
    analyze_image_forensics,
    extract_noise_residual, noise_variance_map, fft_energy_ratio,
    get_edges, estimate_stroke_width, stroke_statistics,
    extract_background_regions, texture_entropy
)
from pipeline.stylometry import (
    analyze_stylometry,
    parse_text, sentence_length_stats, lexical_diversity,
    function_word_ratio, punctuation_stats, word_repetition_rate
)
from pipeline.ai_text import analyze_ai_text
from pipeline.decision import make_decision
import numpy as np

def analyze_single_image(filepath):
    """Analyze a single image with detailed output"""
    
    print("="*70)
    print(f"ANALYZING: {os.path.basename(filepath)}")
    print("="*70)
    
    # Load and preprocess
    print("\n[1/7] Loading and preprocessing...")
    img = cv2.imread(filepath)
    if img is None:
        print("ERROR: Could not read image")
        return
    
    normalized_img = preprocess_image(img)
    print(f"  ✓ Image shape: {normalized_img.shape}")
    
    # OCR
    print("\n[2/7] Running OCR...")
    raw_text = extract_text(normalized_img)
    print(f"  ✓ Extracted {len(raw_text)} characters")
    if len(raw_text) > 0:
        print(f"  Preview: {raw_text[:100]}...")
    
    # Image forensics details
    print("\n[3/7] Analyzing noise patterns...")
    noise = extract_noise_residual(normalized_img)
    variances = noise_variance_map(noise)
    variance_std = np.std(variances)
    freq_ratio = fft_energy_ratio(noise)
    
    print(f"  Variance Std: {variance_std:.3f} (Real: >25, AI: <8)")
    print(f"  Freq Ratio:   {freq_ratio:.3f} (Real: >1.2, AI: <0.6)")
    
    print("\n[4/7] Analyzing stroke textures...")
    edges = get_edges(normalized_img)
    stroke_pixels = estimate_stroke_width(edges)
    if len(stroke_pixels) > 0:
        stroke_stat = stroke_statistics(stroke_pixels)
        print(f"  Stroke stat:  {stroke_stat:.3f} (Real: >10, AI: <5)")
        print(f"  Stroke pixels: {len(stroke_pixels)}")
    else:
        print("  ⚠️  No strokes detected")
    
    print("\n[5/7] Analyzing paper texture...")
    bg_pixels = extract_background_regions(normalized_img)
    if len(bg_pixels) > 500:
        paper_entropy = texture_entropy(bg_pixels)
        paper_variance = np.var(bg_pixels)
        print(f"  Paper entropy: {paper_entropy:.3f} (Real: 2-3.5)")
        print(f"  Paper variance: {paper_variance:.3f} (Real: 60-180)")
    else:
        print("  ⚠️  Insufficient background detected")
    
    # Run full analysis
    print("\n[6/7] Running full forensic analysis...")
    image_scores = analyze_image_forensics(normalized_img)
    stylometry_score = analyze_stylometry(raw_text)
    ai_text_score = analyze_ai_text(raw_text)
    
    print(f"  Noise score:       {image_scores['noise']:.3f}")
    print(f"  Stroke score:      {image_scores['stroke']:.3f}")
    print(f"  Paper score:       {image_scores['paper']:.3f}")
    print(f"  → Image forensics: {image_scores['image_forensics']:.3f}")
    print(f"  Stylometry:        {stylometry_score:.3f}")
    print(f"  AI text:           {ai_text_score:.3f}")
    
    # Stylometry details
    if len(raw_text) > 20:
        print("\n  📝 Stylometry breakdown:")
        sentences, words = parse_text(raw_text)
        mean_len, std_len = sentence_length_stats(sentences)
        lex_div = lexical_diversity(words)
        func_ratio = function_word_ratio(words)
        
        print(f"    Sentences: {len(sentences)}")
        print(f"    Words: {len(words)}")
        print(f"    Avg sentence length: {mean_len:.1f} words")
        print(f"    Sentence length std: {std_len:.2f} (Real: >3, AI: <3)")
        print(f"    Lexical diversity: {lex_div:.3f} (Real: <0.7, AI: >0.7)")
        print(f"    Function word ratio: {func_ratio:.3f} (Real: <0.08, AI: >0.1)")
    
    # Decision
    print("\n[7/7] Making final decision...")
    verdict, confidence, all_scores = make_decision(
        image_scores, stylometry_score, ai_text_score,
        text_length=len(raw_text)
    )
    
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    print(f"\n  🎯 {verdict}")
    print(f"  📊 Confidence: {confidence}%")
    print(f"  🔢 Final Score: {all_scores['final']:.3f}")
    
    print(f"\n  Weights used:")
    for k, v in all_scores['weights_used'].items():
        print(f"    {k}: {v:.2f}")
    
    print(f"\n  Module agreement: {all_scores['module_agreement']:.1%}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if verdict == "Likely Human":
        print("\n  ✅ This document shows characteristics of authentic handwriting")
        print("  • Scores consistently below synthetic thresholds")
        print("  • Natural variation in multiple forensic domains")
    elif verdict == "Suspicious":
        print("\n  ⚠️  This document shows mixed signals")
        print("  • Some modules indicate synthetic content")
        print("  • Recommend additional verification")
    else:
        print("\n  ❌ This document likely involves AI assistance")
        print("  • Multiple forensic modules show synthetic indicators")
        print("  • Scores exceed thresholds for authentic handwriting")
    
    print("\n" + "="*70)
    
    return all_scores

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python quick_test.py <image_path>")
        print("\nExample:")
        print("  python quick_test.py real_handwriting.jpg")
        print("  python quick_test.py fake_handwriting.jpg")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)
    
    analyze_single_image(filepath)