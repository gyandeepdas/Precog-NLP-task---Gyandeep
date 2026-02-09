"""
Linguistic Feature Extractor for XGBoost Model
==============================================

Extracts the same features that the XGBoost model was trained on:
- TTR (Type-Token Ratio)
- Readability scores
- POS tag ratios (noun, verb, adj, adv, etc.)
"""

import numpy as np
import spacy
from textstat import flesch_reading_ease
from collections import Counter

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading spaCy model...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class LinguisticFeatureExtractor:
    """Extract linguistic features from text."""
    
    def __init__(self):
        self.feature_names = [
            'ttr', 'readability', 'noun_ratio', 'verb_ratio', 
            'adj_ratio', 'adv_ratio', 'pron_ratio', 'det_ratio', 
            'adp_ratio', 'aux_ratio', 'sconj_ratio', 'cconj_ratio', 
            'part_ratio', 'intj_ratio'
        ]
    
    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract all linguistic features from text.
        
        Args:
            text: Input text string
            
        Returns:
            numpy array with 14 features
        """
        # Process text with spaCy
        doc = nlp(text)
        
        # 1. Type-Token Ratio (TTR)
        tokens = [token.text.lower() for token in doc if token.is_alpha]
        if len(tokens) == 0:
            ttr = 0.0
        else:
            ttr = len(set(tokens)) / len(tokens)
        
        # 2. Readability (Flesch Reading Ease)
        try:
            readability = flesch_reading_ease(text)
        except:
            readability = 50.0  # Default neutral score
        
        # 3-14. POS tag ratios
        total_tokens = len(doc)
        if total_tokens == 0:
            pos_ratios = [0.0] * 12
        else:
            pos_counts = Counter([token.pos_ for token in doc])
            
            pos_ratios = [
                pos_counts.get('NOUN', 0) / total_tokens,   # noun_ratio
                pos_counts.get('VERB', 0) / total_tokens,   # verb_ratio
                pos_counts.get('ADJ', 0) / total_tokens,    # adj_ratio
                pos_counts.get('ADV', 0) / total_tokens,    # adv_ratio
                pos_counts.get('PRON', 0) / total_tokens,   # pron_ratio
                pos_counts.get('DET', 0) / total_tokens,    # det_ratio
                pos_counts.get('ADP', 0) / total_tokens,    # adp_ratio
                pos_counts.get('AUX', 0) / total_tokens,    # aux_ratio
                pos_counts.get('SCONJ', 0) / total_tokens,  # sconj_ratio
                pos_counts.get('CCONJ', 0) / total_tokens,  # cconj_ratio
                pos_counts.get('PART', 0) / total_tokens,   # part_ratio
                pos_counts.get('INTJ', 0) / total_tokens,   # intj_ratio
            ]
        
        features = [ttr, readability] + pos_ratios
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self):
        """Return list of feature names."""
        return self.feature_names
