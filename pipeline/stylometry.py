"""
Stylometry Module - UPDATED WITH GEMINI DETECTION
Analyzes writing behavior, NOT handwriting shape
Strongest signal against AI-edited text
"""

import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Function words commonly overused by AI
FUNCTION_WORDS = {
    'and', 'but', 'or', 'so', 'because',
    'if', 'then', 'however', 'therefore',
    'moreover', 'furthermore', 'additionally',
    'thus', 'hence', 'consequently'
}

def parse_text(text):
    """Parse into sentences and words"""
    if not text or len(text.strip()) < 10:
        return [], []
    
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    return sentences, words

def sentence_length_stats(sentences):
    """
    Measure sentence length variation
    AI tends to normalize sentence length
    Humans are inconsistent
    """
    if not sentences:
        return 0.0, 0.0
    
    lengths = [len(word_tokenize(s)) for s in sentences]
    
    if not lengths:
        return 0.0, 0.0
    
    return np.mean(lengths), np.std(lengths)

def lexical_diversity(words):
    """
    Vocabulary richness
    AI edits often increase diversity unnaturally
    Type-Token Ratio (TTR)
    """
    if not words or len(words) < 10:
        return 0.0
    
    unique_words = len(set(words))
    total_words = len(words)
    
    return unique_words / total_words

def function_word_ratio(words):
    """
    Ratio of function words (connectors)
    AI tends to overuse these for 'better flow'
    """
    if not words:
        return 0.0
    
    func_count = sum(1 for w in words if w in FUNCTION_WORDS)
    return func_count / len(words)

def punctuation_stats(text):
    """
    Punctuation usage patterns
    Humans: inconsistent
    AI: tidy, normalized
    """
    if not text:
        return 0.0, 0.0
    
    punct_marks = ',.;:!?'
    punct_count = sum(1 for c in text if c in punct_marks)
    cap_count = sum(1 for c in text if c.isupper())
    
    text_len = len(text)
    
    punct_ratio = punct_count / text_len if text_len > 0 else 0.0
    cap_ratio = cap_count / text_len if text_len > 0 else 0.0
    
    return punct_ratio, cap_ratio

def word_repetition_rate(words):
    """
    How often words repeat
    Humans repeat common words naturally
    AI tries to vary vocabulary
    """
    if not words or len(words) < 20:
        return 0.0
    
    word_counts = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1
    
    # Average repetition
    repetitions = sum(count for count in word_counts.values() if count > 1)
    
    return repetitions / len(words)

def detect_gemini_patterns(text):
    """
    NEW: Specific patterns that Gemini-generated handwriting shows
    Gemini tends to create very clean, formal text
    """
    if not text or len(text) < 20:
        return 0.5
    
    words = text.lower().split()
    
    # Gemini tends to:
    # 1. Use very consistent spacing
    # 2. Have perfect spelling
    # 3. Use formal language
    # 4. Avoid contractions
    
    # Check for contractions (humans use them, AI avoids)
    contractions = ["don't", "can't", "won't", "it's", "i'm", "you're", "we're", 
                   "they're", "isn't", "aren't", "wasn't", "weren't", "hasn't", 
                   "haven't", "hadn't", "doesn't", "didn't", "couldn't", "shouldn't",
                   "wouldn't", "mightn't", "mustn't"]
    has_contractions = any(c in words for c in contractions)
    
    # Check for informal words
    informal = ["yeah", "nah", "gonna", "wanna", "kinda", "sorta", "gotta", 
               "lemme", "dunno", "yep", "nope", "ok", "okay"]
    has_informal = any(i in words for i in informal)
    
    # Check for common human errors/quirks
    casual_markers = ["lol", "haha", "hehe", "omg", "btw", "tbh", "imo", "fyi"]
    has_casual = any(c in words for c in casual_markers)
    
    # Check spacing consistency
    sentences = text.split('.')
    sentence_lengths = [len(s.strip()) for s in sentences if s.strip()]
    
    if len(sentence_lengths) > 2:
        spacing_std = np.std(sentence_lengths)
        # AI has very consistent spacing
        spacing_score = 1.0 if spacing_std < 20 else 0.3
    else:
        spacing_score = 0.5
    
    # Check for overly perfect capitalization
    lines = text.split('\n')
    proper_start = sum(1 for line in lines if line.strip() and line.strip()[0].isupper())
    cap_perfection = proper_start / len(lines) if lines else 0.0
    
    # Very high capitalization perfection = AI
    cap_score = 1.0 if cap_perfection > 0.9 else 0.2
    
    # Combine signals
    gemini_score = 0.0
    
    if not has_contractions:
        gemini_score += 0.25  # No contractions = formal = AI
    if not has_informal:
        gemini_score += 0.20  # No informal words = AI
    if not has_casual:
        gemini_score += 0.15  # No casual markers = AI
    
    gemini_score += 0.25 * spacing_score  # Spacing consistency
    gemini_score += 0.15 * cap_score  # Capitalization perfection
    
    return min(1.0, gemini_score)

def analyze_stylometry(text):
    """
    Complete stylometric analysis - UPDATED WITH GEMINI DETECTION
    Returns deviation score [0,1]
    Higher = more AI-normalized behavior
    """
    if not text or len(text.strip()) < 20:
        return 0.5  # Neutral for very short text
    
    sentences, words = parse_text(text)
    
    # Feature extraction
    mean_len, std_len = sentence_length_stats(sentences)
    lex_div = lexical_diversity(words)
    func_ratio = function_word_ratio(words)
    punct_ratio, cap_ratio = punctuation_stats(text)
    rep_rate = word_repetition_rate(words)
    
    # Scoring logic:
    # - Low sentence variance = AI normalization
    # - High lexical diversity = AI trying to sound smart
    # - High function word ratio = AI connectors
    # - Very regular punctuation = AI tidiness
    # - Low repetition = AI avoiding redundancy
    
    # Normalize variance (human ~3-8, AI ~1-3)
    variance_score = 1.0 - min(1.0, std_len / 5.0)
    
    # High diversity suspicious (human ~0.5-0.7, AI ~0.7-0.9)
    diversity_score = max(0.0, lex_div - 0.6) * 2.0
    
    # Function word overuse (human ~0.05-0.1, AI ~0.1-0.15)
    func_score = max(0.0, func_ratio - 0.08) * 10.0
    
    # Punctuation regularity (measure deviation from natural)
    punct_score = min(1.0, punct_ratio * 50.0)
    
    # Low repetition = AI (human ~0.3-0.5, AI ~0.1-0.3)
    rep_score = 1.0 - min(1.0, rep_rate / 0.4)
    
    # Base stylometry score
    stylometry_score = (
        0.30 * variance_score +
        0.25 * diversity_score +
        0.20 * func_score +
        0.15 * punct_score +
        0.10 * rep_score
    )
    
    # Add Gemini-specific detection
    gemini_score = detect_gemini_patterns(text)
    
    # Weighted combination (60% traditional stylometry, 40% Gemini-specific)
    final_score = 0.60 * stylometry_score + 0.40 * gemini_score
    
    return max(0.0, min(1.0, final_score))