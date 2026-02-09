import torch
import numpy as np
import pandas as pd
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, classification_report
import tensorflow as tf
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("AI vs Human Text Classifier - Model Performance Evaluation")
print("="*80)

# Load test data
print("\n[1/6] Loading test dataset...")
df = pd.read_csv('/home/SexyLadGD/Downloads/NLP models/final_dataset_gemma_complete.csv')
test_texts = df['text'].tolist()
test_labels = df['is_ai'].tolist()
print(f"✓ Loaded {len(test_texts)} samples")
print(f"  - AI-written: {sum(test_labels)} ({sum(test_labels)/len(test_labels)*100:.1f}%)")
print(f"  - Human-written: {len(test_labels)-sum(test_labels)} ({(len(test_labels)-sum(test_labels))/len(test_labels)*100:.1f}%)")

# Performance Evaluation Function
def evaluate_model_performance(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """Calculate comprehensive performance metrics"""
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    roc_auc = roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else None
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    return results

def print_performance_report(results):
    """Print detailed performance report"""
    print(f"\n{'='*70}")
    print(f"Performance Report: {results['model_name']}")
    print(f"{'='*70}")
    print(f"Accuracy:      {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Precision:     {results['precision']:.4f}")
    print(f"Recall:        {results['recall']:.4f}")
    print(f"F1-Score:      {results['f1_score']:.4f}")
    print(f"Specificity:   {results['specificity']:.4f}")
    print(f"Sensitivity:   {results['sensitivity']:.4f}")
    
    if results['roc_auc'] is not None:
        print(f"ROC-AUC:       {results['roc_auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                Human      AI")
    print(f"Actual Human    {results['true_negatives']:>6}  {results['false_positives']:>6}")
    print(f"       AI       {results['false_negatives']:>6}  {results['true_positives']:>6}")
    print(f"{'='*70}\n")

# ========== DISTILBERT MODEL EVALUATION ==========
print("\n[2/6] Evaluating DistilBERT model...")
try:
    distilbert_model = AutoModelForSequenceClassification.from_pretrained(
        '/home/SexyLadGD/Downloads/NLP models/Distilbert model'
    )
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Predict in batches to avoid memory issues
    batch_size = 16
    all_predictions = []
    all_probabilities = []
    
    for i in range(0, len(test_texts), batch_size):
        batch_texts = test_texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            probabilities = torch.softmax(outputs.logits, dim=-1)[:, 1]
            
            all_predictions.extend(predictions.numpy())
            all_probabilities.extend(probabilities.numpy())
    
    distilbert_results = evaluate_model_performance(
        test_labels, 
        all_predictions, 
        all_probabilities,
        "DistilBERT"
    )
    print_performance_report(distilbert_results)
    print("✓ DistilBERT evaluation complete")
    
except Exception as e:
    print(f"✗ Error evaluating DistilBERT: {e}")
    distilbert_results = None

# ========== SEMANTIC NN MODEL EVALUATION ==========
print("\n[3/6] Evaluating Semantic NN model...")
try:
    semantic_model = tf.keras.models.load_model(
        '/home/SexyLadGD/Downloads/NLP models/semantic_nnmodel',
        compile=False
    )
    
    # For semantic model, we need to prepare text embeddings
    # Using TF-IDF as a simple feature extractor (adjust based on your training)
    with open('/home/SexyLadGD/Downloads/NLP models/semantic_nnmodel/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Create TF-IDF features (this is a simplification - adjust based on your actual preprocessing)
    vectorizer = TfidfVectorizer(max_features=metadata.get('input_dim', 5000))
    X_test = vectorizer.fit_transform(test_texts).toarray()
    
    # Predict
    predictions_proba = semantic_model.predict(X_test, batch_size=32, verbose=0)
    predictions = (predictions_proba > 0.5).astype(int).flatten()
    
    semantic_results = evaluate_model_performance(
        test_labels,
        predictions,
        predictions_proba.flatten(),
        "Semantic NN"
    )
    print_performance_report(semantic_results)
    print("✓ Semantic NN evaluation complete")
    
except Exception as e:
    print(f"✗ Error evaluating Semantic NN: {e}")
    print(f"  Note: You may need to adjust the feature extraction to match your training process")
    semantic_results = None

# ========== XGBOOST MODEL EVALUATION ==========
print("\n[4/6] Evaluating XGBoost model...")
try:
    xgb_model = xgb.Booster()
    xgb_model.load_model('/home/SexyLadGD/Downloads/NLP models/Xgboost model/xgboost_ai_detector.json')
    
    # Prepare features (using TF-IDF - adjust based on your training)
    vectorizer_xgb = TfidfVectorizer(max_features=5000)
    X_test_xgb = vectorizer_xgb.fit_transform(test_texts)
    dtest = xgb.DMatrix(X_test_xgb)
    
    # Predict
    xgb_proba = xgb_model.predict(dtest)
    xgb_predictions = (xgb_proba > 0.5).astype(int)
    
    xgb_results = evaluate_model_performance(
        test_labels,
        xgb_predictions,
        xgb_proba,
        "XGBoost"
    )
    print_performance_report(xgb_results)
    print("✓ XGBoost evaluation complete")
    
except Exception as e:
    print(f"✗ Error evaluating XGBoost: {e}")
    print(f"  Note: You may need to adjust the feature extraction to match your training process")
    xgb_results = None

# ========== COMBINED COMPARISON ==========
print("\n[5/6] Generating comparison analysis...")
all_results = [r for r in [distilbert_results, semantic_results, xgb_results] if r is not None]

if len(all_results) > 0:
    print(f"\n{'='*90}")
    print(f"{'MODEL PERFORMANCE COMPARISON':^90}")
    print(f"{'='*90}")
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>11} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
    print(f"{'-'*90}")
    
    for result in all_results:
        roc_str = f"{result['roc_auc']:.4f}" if result['roc_auc'] is not None else "N/A"
        print(f"{result['model_name']:<20} {result['accuracy']:>10.4f} {result['precision']:>11.4f} "
              f"{result['recall']:>10.4f} {result['f1_score']:>10.4f} {roc_str:>10}")
    
    print(f"{'='*90}\n")
    
    # Find best model
    best_model = max(all_results, key=lambda x: x['f1_score'])
    print(f"🏆 Best Model: {best_model['model_name']} (F1-Score: {best_model['f1_score']:.4f})")

# ========== VISUALIZATION ==========
print("\n[6/6] Creating visualizations...")
if len(all_results) >= 2:
    # Performance comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    models = [r['model_name'] for r in all_results]
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(models)]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = [r[metric] for r in all_results]
        bars = ax.bar(models, values, color=colors)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/SexyLadGD/Downloads/NLP models/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: model_comparison.png")
    
    # Confusion matrices
    num_models = len(all_results)
    fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5))
    if num_models == 1:
        axes = [axes]
    fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
    
    for idx, result in enumerate(all_results):
        ax = axes[idx]
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'], ax=ax,
                   cbar_kws={'label': 'Count'})
        ax.set_title(f"{result['model_name']}\nAccuracy: {result['accuracy']:.3f}", fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('/home/SexyLadGD/Downloads/NLP models/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: confusion_matrices.png")

print("\n" + "="*80)
print("Evaluation Complete!")
print("="*80)
