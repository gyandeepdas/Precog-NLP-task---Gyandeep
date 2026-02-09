"""
STACKING ENSEMBLE FOR AI-TEXT DETECTION
=========================================

ASSUMPTIONS:
1. XGBoost model saved in JSON format (xgboost_ai_detector.json)
2. Semantic FFNN expects 100-dimensional GloVe embeddings as input
3. DistilBERT model expects tokenized text (HuggingFace format)
4. All models output probabilities for AI-generated text (binary classification)
5. Meta-model will be trained on a held-out validation set to prevent data leakage

ARCHITECTURE:
- Three base models (frozen, pre-trained)
- Disagreement-based features (std_p, max_gap)
- Logistic Regression meta-learner
- Confidence-based shortcut for high-agreement cases

AUTHOR: AI-Text Detection Ensemble System
DATE: February 2026
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Model-specific imports
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")


class GloVeEmbedder:
    """
    Handles text-to-embedding conversion using pre-trained GloVe vectors.
    Averages word embeddings to create sentence-level representations.
    """
    
    def __init__(self, glove_path: str):
        """
        Load GloVe embeddings from file.
        
        Args:
            glove_path: Path to glove.6B.100d.txt file
        """
        print("Loading GloVe embeddings...")
        self.embeddings = {}
        self.embedding_dim = 100
        
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                self.embeddings[word] = vector
        
        print(f"Loaded {len(self.embeddings)} word vectors")
    
    def get_sentence_embedding(self, text: str) -> np.ndarray:
        """
        Convert text to averaged GloVe embedding.
        
        Args:
            text: Input text string
            
        Returns:
            100-dimensional numpy array
        """
        words = text.lower().split()
        word_vectors = []
        
        for word in words:
            if word in self.embeddings:
                word_vectors.append(self.embeddings[word])
        
        if len(word_vectors) == 0:
            # Return zero vector if no words found
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Average all word vectors
        return np.mean(word_vectors, axis=0).astype(np.float32)


class XGBoostPredictor:
    """
    Wrapper for XGBoost model with linguistic features.
    Assumes model was trained on linguistic features (TTR, readability, POS ratios).
    """
    
    def __init__(self, model_path: str, feature_extractor=None):
        """
        Load XGBoost model from JSON file.
        
        Args:
            model_path: Path to xgboost_ai_detector.json
            feature_extractor: LinguisticFeatureExtractor instance
        """
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        # Feature extractor (will be set during initialization)
        self.feature_extractor = feature_extractor
        print("XGBoost model loaded")
    
    def set_feature_extractor(self, feature_extractor):
        """Set the feature extractor."""
        self.feature_extractor = feature_extractor
    
    def predict(self, text: str) -> float:
        """
        Predict probability of AI-generated text.
        
        Args:
            text: Input text string
            
        Returns:
            Probability between 0 and 1
        """
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not set. Call set_feature_extractor() first.")
        
        # Extract linguistic features
        features = self.feature_extractor.extract_features(text)
        features = features.reshape(1, -1)
        
        # Create DMatrix with feature names
        dmatrix = xgb.DMatrix(
            features,
            feature_names=self.feature_extractor.get_feature_names()
        )
        
        # Get probability
        prob = self.model.predict(dmatrix)[0]
        return float(prob)


class FFNNPredictor:
    """
    Wrapper for 3-layer Feedforward Neural Network.
    Uses GloVe embeddings as input features.
    """
    
    def __init__(self, model_path: str, glove_embedder: GloVeEmbedder):
        """
        Load Keras FFNN model.
        
        Args:
            model_path: Path to model.weights.h5 directory
            glove_embedder: GloVeEmbedder instance for text encoding
        """
        # Load model architecture from config
        config_path = Path(model_path) / "config.json"
        weights_path = Path(model_path) / "model.weights.h5"
        
        # Reconstruct model from config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Build model: Input(100) -> Dense(64, relu) -> Dense(32, relu) -> Dense(1, sigmoid)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(100,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Load weights
        self.model.load_weights(str(weights_path))
        self.glove_embedder = glove_embedder
        
        print("FFNN model loaded")
    
    def predict(self, text: str) -> float:
        """
        Predict probability of AI-generated text.
        
        Args:
            text: Input text string
            
        Returns:
            Probability between 0 and 1
        """
        # Convert text to GloVe embedding
        embedding = self.glove_embedder.get_sentence_embedding(text)
        embedding = np.expand_dims(embedding, axis=0)  # Add batch dimension
        
        # Get probability
        prob = self.model.predict(embedding, verbose=0)[0][0]
        return float(prob)


class DistilBERTPredictor:
    """
    Wrapper for DistilBERT sequence classifier.
    Uses HuggingFace transformers for tokenization and inference.
    """
    
    def __init__(self, model_path: str):
        """
        Load DistilBERT model and tokenizer.
        
        Args:
            model_path: Path to DistilBERT model directory
        """
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()  # Set to evaluation mode
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"DistilBERT model loaded (device: {self.device})")
    
    def predict(self, text: str) -> float:
        """
        Predict probability of AI-generated text.
        
        Args:
            text: Input text string
            
        Returns:
            Probability between 0 and 1
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            
            # Assuming class 1 is AI-generated
            prob = probs[0][1].item()
        
        return float(prob)


class DisagreementFeatures:
    """
    Computes disagreement-based features from base model predictions.
    These features capture uncertainty and model disagreement.
    """
    
    @staticmethod
    def compute(predictions: List[float]) -> Dict[str, float]:
        """
        Compute disagreement features.
        
        Args:
            predictions: List of probabilities [p_xgb, p_ffnn, p_bert]
            
        Returns:
            Dictionary with mean_p, std_p, max_gap
        """
        preds = np.array(predictions)
        
        # Mean probability (ensemble average)
        mean_p = float(np.mean(preds))
        
        # Standard deviation (measure of disagreement)
        std_p = float(np.std(preds))
        
        # Maximum pairwise gap
        max_gap = 0.0
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                gap = abs(preds[i] - preds[j])
                if gap > max_gap:
                    max_gap = gap
        
        return {
            'mean_p': mean_p,
            'std_p': std_p,
            'max_gap': float(max_gap)
        }


class StackingEnsemble:
    """
    Main stacking ensemble that combines three base models with a meta-learner.
    
    Architecture:
        Base Models -> [p_xgb, p_ffnn, p_bert] -> 
        Disagreement Features -> [std_p, max_gap] ->
        Meta-Model (Logistic Regression) -> Final Probability
    """
    
    def __init__(
        self,
        base_dir: str = ".",
        confidence_threshold: float = 0.05
    ):
        """
        Initialize the stacking ensemble.
        
        Args:
            base_dir: Base directory containing all model files
            confidence_threshold: If std_p < this, use mean shortcut
        """
        self.base_dir = Path(base_dir)
        self.confidence_threshold = confidence_threshold
        
        # Initialize GloVe embedder (shared by FFNN)
        print("\n" + "="*60)
        print("INITIALIZING STACKING ENSEMBLE")
        print("="*60)
        
        glove_path = self.base_dir / "glove.6B.100d.txt"
        self.glove_embedder = GloVeEmbedder(str(glove_path))
        
        # Initialize linguistic feature extractor for XGBoost
        from linguistic_features import LinguisticFeatureExtractor
        self.feature_extractor = LinguisticFeatureExtractor()
        
        # Initialize base models
        print("\nLoading base models...")
        self.xgb_model = XGBoostPredictor(
            str(self.base_dir / "Xgboost model" / "xgboost_ai_detector.json"),
            feature_extractor=self.feature_extractor
        )
        
        self.ffnn_model = FFNNPredictor(
            str(self.base_dir / "semantic_nnmodel"),
            self.glove_embedder
        )
        
        self.bert_model = DistilBERTPredictor(
            str(self.base_dir / "Distilbert model")
        )
        
        # Meta-model (will be trained later)
        self.meta_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        self.meta_scaler = StandardScaler()
        self.is_meta_trained = False
        
        print("\n" + "="*60)
        print("ALL MODELS LOADED SUCCESSFULLY")
        print("="*60 + "\n")
    
    def get_base_predictions(self, text: str) -> Dict[str, float]:
        """
        Get predictions from all three base models.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with p_xgb, p_ffnn, p_bert
        """
        p_xgb = self.xgb_model.predict(text)
        p_ffnn = self.ffnn_model.predict(text)
        p_bert = self.bert_model.predict(text)
        
        return {
            'p_xgb': p_xgb,
            'p_ffnn': p_ffnn,
            'p_bert': p_bert
        }
    
    def extract_meta_features(self, base_preds: Dict[str, float]) -> np.ndarray:
        """
        Extract meta-features from base predictions.
        
        Args:
            base_preds: Dictionary with p_xgb, p_ffnn, p_bert
            
        Returns:
            Feature vector [p_xgb, p_ffnn, p_bert, std_p, max_gap]
        """
        predictions = [base_preds['p_xgb'], base_preds['p_ffnn'], base_preds['p_bert']]
        disagreement = DisagreementFeatures.compute(predictions)
        
        features = [
            base_preds['p_xgb'],
            base_preds['p_ffnn'],
            base_preds['p_bert'],
            disagreement['std_p'],
            disagreement['max_gap']
        ]
        
        return np.array(features)
    
    def train_meta_model(
        self,
        texts: List[str],
        labels: List[int]
    ):
        """
        Train the meta-model on validation data.
        
        IMPORTANT: To prevent data leakage, this should be called on:
        - A held-out validation set, OR
        - Out-of-fold predictions from cross-validation
        
        Args:
            texts: List of input texts
            labels: List of true labels (0=human, 1=AI)
        """
        print("\n" + "="*60)
        print("TRAINING META-MODEL")
        print("="*60)
        print(f"Training samples: {len(texts)}")

        
        # Get base model predictions for all samples
        meta_features = []
        
        for i, text in enumerate(texts):
            if (i + 1) % 100 == 0:
                print(f"Processing sample {i + 1}/{len(texts)}...")
            
            try:
                base_preds = self.get_base_predictions(text)
                features = self.extract_meta_features(base_preds)
                meta_features.append(features)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # Use neutral predictions as fallback
                meta_features.append(np.array([0.5, 0.5, 0.5, 0.0, 0.0]))
        
        meta_features = np.array(meta_features)
        
        # Standardize features
        meta_features_scaled = self.meta_scaler.fit_transform(meta_features)
        
        # Train logistic regression
        self.meta_model.fit(meta_features_scaled, labels)
        self.is_meta_trained = True
        
        # Report training accuracy
        train_preds = self.meta_model.predict(meta_features_scaled)
        accuracy = np.mean(train_preds == labels)
        print(f"\nMeta-model training accuracy: {accuracy:.4f}")
        print("="*60 + "\n")
    
    def predict(self, text: str, use_confidence_shortcut: bool = True) -> float:
        """
        Main inference function: returns calibrated probability of AI-generated text.
        
        Pipeline:
        1. Get predictions from all base models
        2. Compute disagreement features
        3. If high agreement (std_p < threshold), return mean
        4. Otherwise, use meta-model for final prediction
        
        Args:
            text: Input text string
            use_confidence_shortcut: If True, use mean when models agree
            
        Returns:
            Probability that text is AI-generated (0 to 1)
        """
        # Get base predictions
        base_preds = self.get_base_predictions(text)
        
        # Compute disagreement
        predictions = [base_preds['p_xgb'], base_preds['p_ffnn'], base_preds['p_bert']]
        disagreement = DisagreementFeatures.compute(predictions)
        
        # Confidence shortcut: if models strongly agree, return mean
        if use_confidence_shortcut and disagreement['std_p'] < self.confidence_threshold:
            return disagreement['mean_p']
        
        # Otherwise, use meta-model
        if not self.is_meta_trained:
            # Fallback to mean if meta-model not trained
            print("Warning: Meta-model not trained. Returning mean prediction.")
            return disagreement['mean_p']
        
        # Extract and scale features
        features = self.extract_meta_features(base_preds)
        features_scaled = self.meta_scaler.transform(features.reshape(1, -1))
        
        # Get meta-model probability
        prob = self.meta_model.predict_proba(features_scaled)[0][1]
        
        return float(prob)
    
    def predict_with_details(self, text: str) -> Dict:
        """
        Return detailed predictions including base model outputs and disagreement.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with all intermediate predictions and final result
        """
        base_preds = self.get_base_predictions(text)
        predictions = [base_preds['p_xgb'], base_preds['p_ffnn'], base_preds['p_bert']]
        disagreement = DisagreementFeatures.compute(predictions)
        
        final_prob = self.predict(text)
        
        return {
            'base_predictions': base_preds,
            'disagreement_features': disagreement,
            'final_probability': final_prob,
            'prediction': 'AI-generated' if final_prob > 0.5 else 'Human-written',
            'confidence': abs(final_prob - 0.5) * 2  # 0 to 1 scale
        }
    
    def save_meta_model(self, save_path: str = "meta_model.pkl"):
        """Save the trained meta-model and scaler."""
        if not self.is_meta_trained:
            print("Warning: Meta-model not trained yet.")
            return
        
        save_data = {
            'meta_model': self.meta_model,
            'meta_scaler': self.meta_scaler,
            'confidence_threshold': self.confidence_threshold
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Meta-model saved to {save_path}")
    
    def load_meta_model(self, load_path: str = "meta_model.pkl"):
        """Load a trained meta-model and scaler."""
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.meta_model = save_data['meta_model']
        self.meta_scaler = save_data['meta_scaler']
        self.confidence_threshold = save_data['confidence_threshold']
        self.is_meta_trained = True
        print(f"Meta-model loaded from {load_path}")


# ==============================================================================
# MAIN EXECUTION & DEMONSTRATION
# ==============================================================================

def main():
    """
    Demonstration of the stacking ensemble system.
    """
    # Initialize ensemble
    ensemble = StackingEnsemble()
    
    # Example: For meta-model training, you would use a validation set
    # This is just a placeholder - replace with your actual validation data
    print("\nNOTE: Meta-model training requires validation data.")
    print("After providing validation data, call:")
    print("  ensemble.train_meta_model(val_texts, val_labels)")
    print("  ensemble.save_meta_model('meta_model.pkl')")
    
    # Example inference (after meta-model is trained)
    example_text = "This is an example text to classify."
    
    print("\n" + "="*60)
    print("EXAMPLE INFERENCE (without trained meta-model)")
    print("="*60)
    
    try:
        result = ensemble.predict_with_details(example_text)
        
        print(f"\nInput text: {example_text[:100]}...")
        print(f"\nBase Model Predictions:")
        print(f"  XGBoost:    {result['base_predictions']['p_xgb']:.4f}")
        print(f"  FFNN:       {result['base_predictions']['p_ffnn']:.4f}")
        print(f"  DistilBERT: {result['base_predictions']['p_bert']:.4f}")
        
        print(f"\nDisagreement Features:")
        print(f"  Mean:       {result['disagreement_features']['mean_p']:.4f}")
        print(f"  Std Dev:    {result['disagreement_features']['std_p']:.4f}")
        print(f"  Max Gap:    {result['disagreement_features']['max_gap']:.4f}")
        
        print(f"\nFinal Prediction: {result['prediction']}")
        print(f"Probability (AI): {result['final_probability']:.4f}")
        print(f"Confidence:       {result['confidence']:.4f}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Make sure all model files are present and TF-IDF vectorizer is set.")


if __name__ == "__main__":
    main()
