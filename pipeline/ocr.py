"""
OCR Module
Extracts raw text from image
NO cleanup, NO correction - OCR noise is signal
"""

import pytesseract
from PIL import Image
import cv2
import numpy as np

def extract_text(gray_img):
    """
    Extract raw text using Tesseract OCR
    
    Args:
        gray_img: Grayscale numpy array
    
    Returns:
        raw_text: String (exactly as OCR sees it)
    """
    
    # Convert numpy array to PIL Image for pytesseract
    pil_img = Image.fromarray(gray_img)
    
    # OCR configuration
    # --psm 6: Assume uniform block of text
    # --oem 3: Use default OCR Engine Mode
    custom_config = r'--psm 6 --oem 3'
    
    try:
        # Extract text - NO post-processing
        raw_text = pytesseract.image_to_string(pil_img, config=custom_config)
        
        # Only strip leading/trailing whitespace (not internal)
        raw_text = raw_text.strip()
        
        return raw_text
    
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def get_ocr_confidence(gray_img):
    """
    Get OCR confidence scores (useful for penalties in decision module)
    """
    pil_img = Image.fromarray(gray_img)
    
    try:
        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
        confidences = [int(conf) for conf in data['conf'] if conf != '-1']
        
        if confidences:
            return np.mean(confidences)
        return 0.0
    
    except:
        return 0.0