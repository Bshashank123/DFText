"""
AI Text Detection Module - IMPROVED
Uses better heuristics since model-based detection isn't working well
"""

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

# Global model cache
_model = None
_tokenizer = None

def load_model():
    """Load pretrained AI text detector"""
    global _model, _tokenizer
    
    if _model is None:
        try:
            model_name = "roberta-base-openai-detector"
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModelForSequenceClassification.from_pretrained(model_name)
            _model.eval()
        except Exception as e:
            print(f"Warning: Could not load AI detection model: {e}")
            _model = "fallback"
            _tokenizer = "fallback"
    
    return _model, _tokenizer

def improved_heuristic_detection(text):
    """
    IMPROVED fallback heuristic-based AI detection
    More accurate patterns for handwritten text
    """
    if not text or len(text.strip()) < 20:
        return 0.5
    
    text_lower = text.lower()
    words = text_lower.split()
    
    if len(words) < 5:
        return 0.5
    
    # === FEATURE 1: AI Formal Markers ===
    ai_markers = {
        'furthermore', 'moreover', 'additionally', 'consequently',
        'therefore', 'thus', 'hence', 'nevertheless', 'nonetheless',
        'delve', 'utilize', 'leverage', 'facilitate', 'optimize',
        'ensure', 'enhance', 'implement', 'establish', 'demonstrate'
    }
    marker_count = sum(1 for w in words if w in ai_markers)
    marker_ratio = marker_count / len(words)
    
    # === FEATURE 2: Lack of Casual Language ===
    casual_words = {
        "yeah", "nah", "gonna", "wanna", "kinda", "sorta", "gotta",
        "yep", "nope", "ok", "okay", "um", "uh", "hmm", "lol", "haha"
    }
    has_casual = any(w in words for w in casual_words)
    casual_score = 0.0 if has_casual else 0.3  # No casual = AI likely
    
    # === FEATURE 3: Contraction Usage ===
    contractions = ["don't", "can't", "won't", "it's", "i'm", "you're", 
                   "we're", "they're", "isn't", "aren't", "wasn't", 
                   "weren't", "hasn't", "haven't"]
    has_contractions = any(c in text_lower for c in contractions)
    contraction_score = 0.0 if has_contractions else 0.25  # No contractions = AI
    
    # === FEATURE 4: Sentence Structure ===
    sentences = [s.strip() for s in re.split('[.!?]', text) if s.strip()]
    
    if len(sentences) > 1:
        sentence_lengths = [len(s.split()) for s in sentences]
        sent_std = np.std(sentence_lengths)
        sent_mean = np.mean(sentence_lengths)
        
        # AI tends to have very consistent sentence lengths
        sent_cv = sent_std / (sent_mean + 1e-6)
        
        # Real: CV ~0.4-0.8, AI: CV ~0.1-0.3
        if sent_cv > 0.5:
            structure_score = 0.0  # Real
        elif sent_cv < 0.2:
            structure_score = 1.0  # AI
        else:
            structure_score = (0.5 - sent_cv) / 0.3
    else:
        structure_score = 0.5
    
    # === FEATURE 5: Word Complexity ===
    avg_word_len = np.mean([len(w) for w in words])
    
    # AI tends toward longer words
    # Real handwriting: 4-5 chars, AI: 5.5-7 chars
    if avg_word_len > 6:
        complexity_score = 1.0
    elif avg_word_len < 4.5:
        complexity_score = 0.0
    else:
        complexity_score = (avg_word_len - 4.5) / 1.5
    
    # === FEATURE 6: Punctuation Perfection ===
    # Count sentences that start with capital and end with punctuation
    proper_sentences = 0
    for sent in sentences:
        if sent and sent[0].isupper() and text.count(sent) > 0:
            # Check if followed by punctuation
            idx = text.find(sent)
            if idx + len(sent) < len(text):
                next_char = text[idx + len(sent)]
                if next_char in '.!?':
                    proper_sentences += 1
    
    if len(sentences) > 0:
        punctuation_perfection = proper_sentences / len(sentences)
    else:
        punctuation_perfection = 0.5
    
    # Very high perfection = AI
    if punctuation_perfection > 0.9:
        punct_score = 1.0
    elif punctuation_perfection < 0.6:
        punct_score = 0.0
    else:
        punct_score = (punctuation_perfection - 0.6) / 0.3
    
    # === FEATURE 7: Repetition Patterns ===
    # Humans repeat words, AI avoids it
    word_counts = {}
    for w in words:
        if len(w) > 3:  # Only count substantial words
            word_counts[w] = word_counts.get(w, 0) + 1
    
    repeated_words = sum(1 for count in word_counts.values() if count > 1)
    repetition_ratio = repeated_words / (len(word_counts) + 1)
    
    # Real: 0.3-0.5, AI: 0.1-0.2
    if repetition_ratio > 0.35:
        rep_score = 0.0  # Real
    elif repetition_ratio < 0.15:
        rep_score = 1.0  # AI
    else:
        rep_score = (0.35 - repetition_ratio) / 0.2
    
    # === WEIGHTED COMBINATION ===
    final_score = (
        0.20 * marker_ratio * 10.0 +      # Formal markers
        0.15 * casual_score +               # Lack of casual language
        0.15 * contraction_score +          # Lack of contractions
        0.15 * structure_score +            # Sentence consistency
        0.10 * complexity_score +           # Word complexity
        0.15 * punct_score +                # Punctuation perfection
        0.10 * rep_score                    # Low repetition
    )
    
    return max(0.0, min(1.0, final_score))

def model_based_detection(text, model, tokenizer):
    """Use pretrained model for AI detection"""
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            ai_prob = probs[0][1].item()
        
        return ai_prob
    
    except Exception as e:
        print(f"Model inference failed: {e}")
        return improved_heuristic_detection(text)

def analyze_ai_text(text):
    """
    Main function: detect AI text probability
    Returns score [0,1]
    Higher = more likely AI-generated/edited
    
    UPDATED: Now uses improved heuristics primarily
    """
    if not text or len(text.strip()) < 20:
        return 0.5  # Neutral for short text
    
    # Always use improved heuristics (more reliable for handwritten OCR text)
    heuristic_score = improved_heuristic_detection(text)
    
    # Try model if available, but weight heuristics higher
    model, tokenizer = load_model()
    
    if model != "fallback":
        try:
            model_score = model_based_detection(text, model, tokenizer)
            # Weight: 70% heuristic, 30% model (heuristics work better for OCR)
            final_score = 0.70 * heuristic_score + 0.30 * model_score
        except:
            final_score = heuristic_score
    else:
        final_score = heuristic_score
    
    return max(0.0, min(1.0, final_score))

def get_ai_confidence(text):
    """Get confidence level for AI detection"""
    score = analyze_ai_text(text)
    
    if score < 0.3:
        return "Low AI probability"
    elif score < 0.6:
        return "Moderate AI probability"
    else:
        return "High AI probability"