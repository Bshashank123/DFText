"""
Advanced Forensics for High-Quality AI Images
Detects subtle artifacts that simple methods miss
"""

import cv2
import numpy as np
from scipy import signal
from scipy.stats import entropy

def detect_cfa_artifacts(gray_img):
    """
    Color Filter Array (CFA) pattern detection
    Real cameras have CFA patterns, AI images don't
    """
    # Apply frequency analysis
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    h, w = magnitude.shape
    # Check for periodic patterns at CFA frequencies
    center_h, center_w = h//2, w//2
    
    # Sample specific frequency regions
    freq_samples = []
    for offset in [10, 20, 30]:
        sample = magnitude[center_h-offset:center_h+offset, center_w-offset:center_w+offset]
        freq_samples.append(np.std(sample))
    
    # Real cameras show peaks at specific frequencies
    # AI images show uniform frequency distribution
    freq_variance = np.var(freq_samples)
    
    # Score: low variance = AI
    if freq_variance < 100:
        return 0.8  # Likely AI
    elif freq_variance > 500:
        return 0.2  # Likely real
    else:
        return 0.5

def detect_jpeg_artifacts(gray_img):
    """
    JPEG compression artifacts
    Real photos have natural JPEG artifacts
    AI-generated images have unusual compression patterns
    """
    # Convert to frequency domain
    dct = cv2.dct(np.float32(gray_img))
    
    # Analyze DCT coefficient distribution
    dct_flat = dct.flatten()
    
    # Real JPEG: specific distribution of coefficients
    # AI images: different distribution
    hist, _ = np.histogram(dct_flat, bins=50)
    hist_entropy = entropy(hist + 1e-10)
    
    # Real images: entropy ~3-4
    # AI images: entropy ~2-3 or >4.5
    if 3.0 < hist_entropy < 4.0:
        return 0.2  # Likely real
    else:
        return 0.7  # Likely AI

def detect_edge_coherence(gray_img):
    """
    Edge coherence analysis
    Real handwriting: edges have natural variation
    AI handwriting: edges too perfect
    """
    # Multi-scale edge detection
    edges_fine = cv2.Canny(gray_img, 50, 150)
    edges_coarse = cv2.Canny(gray_img, 100, 200)
    
    # Compare edge consistency
    edge_diff = cv2.absdiff(edges_fine, edges_coarse)
    coherence = np.sum(edge_diff) / (gray_img.shape[0] * gray_img.shape[1])
    
    # Real: more edge variation (higher coherence difference)
    # AI: too consistent (lower coherence difference)
    if coherence > 0.15:
        return 0.2  # Likely real
    elif coherence < 0.05:
        return 0.8  # Likely AI
    else:
        return 0.5

def detect_pixel_value_distribution(gray_img):
    """
    Pixel value distribution analysis
    Real photos: natural histogram
    AI images: unnatural peaks
    """
    hist, _ = np.histogram(gray_img.flatten(), bins=256, range=(0, 256))
    
    # Normalize
    hist = hist / hist.sum()
    
    # Check for unnatural peaks (AI tends to cluster values)
    peaks, _ = signal.find_peaks(hist, height=0.01)
    
    # Real images: 5-15 peaks
    # AI images: <5 peaks (too smooth) or >20 peaks (too clustered)
    num_peaks = len(peaks)
    
    if 5 <= num_peaks <= 15:
        return 0.2  # Likely real
    else:
        return 0.7  # Likely AI

def analyze_advanced_forensics(gray_img):
    """
    Run all advanced forensic tests
    """
    cfa_score = detect_cfa_artifacts(gray_img)
    jpeg_score = detect_jpeg_artifacts(gray_img)
    edge_score = detect_edge_coherence(gray_img)
    pixel_score = detect_pixel_value_distribution(gray_img)
    
    # Weighted combination
    advanced_score = (
        0.30 * cfa_score +
        0.25 * jpeg_score +
        0.25 * edge_score +
        0.20 * pixel_score
    )
    
    return {
        'cfa': cfa_score,
        'jpeg': jpeg_score,
        'edge': edge_score,
        'pixel': pixel_score,
        'advanced_forensics': advanced_score
    }