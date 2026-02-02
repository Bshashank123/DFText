"""
Image Forensics Module - UPDATED WITH NEW STROKE ANALYSIS
Analyzes pixel-level behavior: noise, strokes, paper
"""

import cv2
import numpy as np
from scipy.stats import entropy

# Import the new stroke analysis
from pipeline.stroke_analysis import analyze_stroke_naturalness

# ==================== NOISE ANALYSIS ====================

def extract_noise_residual(gray_img):
    """Extract noise by subtracting smoothed version"""
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    noise = gray_img.astype(np.float32) - blurred.astype(np.float32)
    return noise

def noise_variance_map(noise, block_size=32):
    """Measure noise variance across spatial blocks"""
    h, w = noise.shape
    variances = []
    
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = noise[y:y+block_size, x:x+block_size]
            variances.append(np.var(block))
    
    return np.array(variances)

def fft_energy_ratio(noise):
    """Analyze frequency spectrum of noise"""
    fft = np.fft.fft2(noise)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    h, w = magnitude.shape
    center = magnitude[h//4:3*h//4, w//4:3*w//4]
    
    high_freq = magnitude.sum() - center.sum()
    low_freq = center.sum()
    
    return high_freq / (low_freq + 1e-6)

def noise_authenticity_score(gray_img):
    """
    Improved noise analysis
    Higher = more synthetic-like
    """
    noise = extract_noise_residual(gray_img)
    variances = noise_variance_map(noise)
    
    if len(variances) == 0:
        return 0.5
    
    variance_std = np.std(variances)
    variance_mean = np.mean(variances)
    freq_ratio = fft_energy_ratio(noise)
    
    # Add: Check noise distribution uniformity
    variance_cv = variance_std / (variance_mean + 1e-6)
    
    # Real photos: variance_std ~15-50, freq_ratio ~0.9-2.0, CV ~0.5-1.5
    # AI images: variance_std ~2-15, freq_ratio ~0.3-0.9, CV ~0.2-0.5
    
    # Variance scoring
    if variance_std > 25:
        var_score = 0.0
    elif variance_std < 8:
        var_score = 1.0
    else:
        var_score = (25 - variance_std) / 17.0
    
    # Frequency ratio scoring
    if freq_ratio > 1.2:
        freq_score = 0.0
    elif freq_ratio < 0.6:
        freq_score = 1.0
    else:
        freq_score = (1.2 - freq_ratio) / 0.6
    
    # CV scoring (uniformity check)
    if variance_cv > 0.8:
        cv_score = 0.0  # Real - non-uniform
    elif variance_cv < 0.3:
        cv_score = 1.0  # AI - too uniform
    else:
        cv_score = (0.8 - variance_cv) / 0.5
    
    # Weighted combination
    score = 0.45 * var_score + 0.35 * freq_score + 0.20 * cv_score
    
    return max(0.0, min(1.0, score))

# ==================== PAPER TEXTURE ====================

def extract_background_regions(gray_img, threshold=180):
    """Sample ink-free (paper) regions"""
    background_mask = gray_img > threshold
    background_pixels = gray_img[background_mask]
    return background_pixels

def texture_entropy(pixels):
    """Measure randomness in paper texture"""
    if len(pixels) < 100:
        return 0.0
    
    hist, _ = np.histogram(pixels, bins=30, density=True)
    return entropy(hist + 1e-6)

def analyze_paper_grain(gray_img):
    """
    Analyze paper grain patterns
    Real paper: visible fiber structure
    AI paper: too clean or artificial
    """
    bg_pixels = extract_background_regions(gray_img, threshold=180)
    
    if len(bg_pixels) < 500:
        return 0.5
    
    # 1. Entropy
    tex_entropy = texture_entropy(bg_pixels)
    
    # 2. Overall variance
    tex_variance = np.var(bg_pixels)
    
    # 3. Local texture roughness
    if len(bg_pixels) > 1000:
        sample = bg_pixels[:1000]
        local_vars = []
        for i in range(0, len(sample)-10, 10):
            local_vars.append(np.var(sample[i:i+10]))
        texture_roughness = np.std(local_vars)
    else:
        texture_roughness = 0.0
    
    # 4. Check for unnatural uniformity
    # Count how many pixels are in narrow range
    median_val = np.median(bg_pixels)
    narrow_range = np.sum(np.abs(bg_pixels - median_val) < 5)
    uniformity_ratio = narrow_range / len(bg_pixels)
    
    # Real paper: entropy ~2-3.5, variance ~50-200, roughness ~10-30, uniformity ~0.2-0.4
    # AI paper: entropy ~1-2 or >4, variance ~10-50 or >300, roughness ~2-10, uniformity <0.2 or >0.5
    
    # Entropy scoring (both extremes are suspicious)
    if 2.0 < tex_entropy < 3.5:
        entropy_score = 0.0
    else:
        entropy_score = min(1.0, abs(tex_entropy - 2.75) / 1.5)
    
    # Variance scoring
    if 60 < tex_variance < 180:
        var_score = 0.0
    elif tex_variance < 30:
        var_score = 1.0
    elif tex_variance > 250:
        var_score = 0.8
    else:
        var_score = 0.4
    
    # Roughness scoring
    if texture_roughness > 12:
        rough_score = 0.0
    elif texture_roughness < 5:
        rough_score = 1.0
    else:
        rough_score = (12 - texture_roughness) / 7.0
    
    # Uniformity scoring
    if 0.25 < uniformity_ratio < 0.45:
        uniform_score = 0.0  # Natural
    elif uniformity_ratio < 0.15 or uniformity_ratio > 0.6:
        uniform_score = 1.0  # Suspicious
    else:
        uniform_score = 0.5
    
    score = 0.35 * entropy_score + 0.30 * var_score + 0.20 * rough_score + 0.15 * uniform_score
    
    return max(0.0, min(1.0, score))

# ==================== FUSION ====================

def analyze_image_forensics(gray_img):
    """
    Run all image forensics modules
    Returns dict of scores
    """
    noise_score = noise_authenticity_score(gray_img)
    stroke_score = analyze_stroke_naturalness(gray_img)  # NEW STROKE MODULE
    paper_score = analyze_paper_grain(gray_img)
    
    # Weighted fusion - Paper module weighted highest (best performer)
    # Stroke module weighted second (new implementation)
    image_forensics_score = (
        0.20 * noise_score +      # Reduced (weakest)
        0.40 * stroke_score +     # Increased (new module)
        0.40 * paper_score        # Highest (best performer)
    )
    
    return {
        'noise': noise_score,
        'stroke': stroke_score,
        'paper': paper_score,
        'image_forensics': image_forensics_score
    }