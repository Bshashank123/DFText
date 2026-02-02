"""
Image Preprocessing Module
Standardizes images without enhancing them
Preserves generation artifacts while removing camera/lighting bias
"""

import cv2
import numpy as np

def preprocess_image(img):
    """
    Standardize image for analysis
    - Convert to grayscale
    - Resize to fixed width
    - Normalize contrast
    - Light denoising
    
    We do NOT beautify. We standardize.
    """
    
    # Step 1: Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Step 2: Resize to fixed width (1200px) maintaining aspect ratio
    target_width = 1200
    h, w = gray.shape
    if w > target_width:
        aspect_ratio = h / w
        new_h = int(target_width * aspect_ratio)
        gray = cv2.resize(gray, (target_width, new_h), interpolation=cv2.INTER_AREA)
    
    # Step 3: Contrast normalization (CLAHE)
    # This removes lighting bias while keeping noise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    
    # Step 4: Very light denoising (preserve structure, reduce only extreme outliers)
    # We use bilateral filter - preserves edges but smooths minor noise
    denoised = cv2.bilateralFilter(normalized, d=3, sigmaColor=10, sigmaSpace=10)
    
    return denoised

def get_grayscale(img):
    """Helper: just get grayscale version"""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()