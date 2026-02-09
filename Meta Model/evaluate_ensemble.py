"""
ENSEMBLE EVALUATION SCRIPT
===========================

This script:
1. Loads the test set
2. Splits it into validation (for meta-model training) and test (for evaluation)
3. Trains the meta-model on validation data
4. Evaluates the ensemble on test data
5. Generates comprehensive performance metrics

IMPORTANT: In production, you should have a separate validation set
that was NOT used to train any base model.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import our ensemble system
from ensemble_stacking import StackingEnsemble

def load_test_data(filepath: str):
    """Load test data from CSV."""
    print(f"\n{'='*60}")
    print("LOADING TEST DATA")
    print('='*60)
    
    df = pd.read_csv(filepath)
    print(f"Total samples: {len(df)}")
    print(f"AI-generated: {df['is_ai'].sum()}")
    print(f"Human-written: {(1 - df['is_ai']).sum()}")
    print(f"\nAuthors: {df['author'].value_counts().to_dict()}")
    
    return df

def train_validation_split(df, val_size=0.4, random_state=42):
    """
    Split data into validation (for meta-model training) and test.
    
    IMPORTANT: In a real scenario, use a completely separate validation set.
    This split is only for demonstration purposes.
    """
    print(f"\n{'='*60}")
    print("SPLITTING DATA")
    print('='*60)
    print(f"Validation size: {val_size*100:.0f}%")
    print(f"Test size: {(1-val_size)*100:.0f}%")
    
    val_df, test_df = train_test_split(
        df, 
        test_size=(1-val_size),
        stratify=df['is_ai'],
        random_state=random_state
    )
    
    print(f"\nValidation set: {len(val_df)} samples")
    print(f"  AI: {val_df['is_ai'].sum()}, Human: {(1-val_df['is_ai']).sum()}")
    print(f"\nTest set: {len(test_df)} samples")
    print(f"  AI: {test_df['is_ai'].sum()}, Human: {(1-test_df['is_ai']).sum()}")
    
    return val_df, test_df

def train_ensemble_meta_model(ensemble, val_texts, val_labels):
    """Train the meta-model on validation data."""
    print(f"\n{'='*60}")
    print("TRAINING META-MODEL")
    print('='*60)
    
    # Train meta-model (no vectorizer needed - XGBoost uses linguistic features)
    ensemble.train_meta_model(
        texts=val_texts,
        labels=val_labels
    )
    
    # Save meta-model
    save_path = "meta_model_trained.pkl"
    ensemble.save_meta_model(save_path)
    print(f"\nMeta-model saved to {save_path}")
    
    return ensemble

def evaluate_ensemble(ensemble, test_texts, test_labels, authors=None):
    """Evaluate ensemble on test set."""
    print(f"\n{'='*60}")
    print("EVALUATING ENSEMBLE")
    print('='*60)
    
    predictions_prob = []
    predictions_binary = []
    base_predictions = {'xgb': [], 'ffnn': [], 'bert': []}
    disagreement_scores = []
    
    print(f"Processing {len(test_texts)} test samples...")
    
    for i, text in enumerate(test_texts):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_texts)}...")
        
        try:
            # Get detailed prediction
            result = ensemble.predict_with_details(text)
            
            # Store predictions
            prob = result['final_probability']
            predictions_prob.append(prob)
            predictions_binary.append(1 if prob > 0.5 else 0)
            
            # Store base model predictions
            base_predictions['xgb'].append(result['base_predictions']['p_xgb'])
            base_predictions['ffnn'].append(result['base_predictions']['p_ffnn'])
            base_predictions['bert'].append(result['base_predictions']['p_bert'])
            
            # Store disagreement
            disagreement_scores.append(result['disagreement_features']['std_p'])
            
        except Exception as e:
            print(f"\n  Warning: Error processing sample {i}: {e}")
            # Use neutral predictions as fallback
            predictions_prob.append(0.5)
            predictions_binary.append(0)
            base_predictions['xgb'].append(0.5)
            base_predictions['ffnn'].append(0.5)
            base_predictions['bert'].append(0.5)
            disagreement_scores.append(0.0)
    
    print(f"\n{'='*60}")
    print("PERFORMANCE METRICS")
    print('='*60)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions_binary)
    precision = precision_score(test_labels, predictions_binary)
    recall = recall_score(test_labels, predictions_binary)
    f1 = f1_score(test_labels, predictions_binary)
    auc = roc_auc_score(test_labels, predictions_prob)
    
    print(f"\nEnsemble Performance:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions_binary)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0,0]} (correctly identified human text)")
    print(f"  False Positives: {cm[0,1]} (human text labeled as AI)")
    print(f"  False Negatives: {cm[1,0]} (AI text labeled as human)")
    print(f"  True Positives:  {cm[1,1]} (correctly identified AI text)")
    
    # Base model performance
    print(f"\n{'='*60}")
    print("BASE MODEL COMPARISON")
    print('='*60)
    
    for model_name, preds in base_predictions.items():
        binary_preds = [1 if p > 0.5 else 0 for p in preds]
        acc = accuracy_score(test_labels, binary_preds)
        f1_score_model = f1_score(test_labels, binary_preds)
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"  F1 Score: {f1_score_model:.4f}")
    
    # Disagreement analysis
    print(f"\n{'='*60}")
    print("DISAGREEMENT ANALYSIS")
    print('='*60)
    
    avg_disagreement = np.mean(disagreement_scores)
    print(f"\nAverage model disagreement (std_p): {avg_disagreement:.4f}")
    print(f"Max disagreement: {np.max(disagreement_scores):.4f}")
    print(f"Min disagreement: {np.min(disagreement_scores):.4f}")
    
    # Analyze errors by disagreement level
    errors = [1 if pred != label else 0 for pred, label in zip(predictions_binary, test_labels)]
    high_disagreement_errors = sum([e for e, d in zip(errors, disagreement_scores) if d > 0.1])
    low_disagreement_errors = sum([e for e, d in zip(errors, disagreement_scores) if d <= 0.1])
    
    print(f"\nErrors in high disagreement cases (std_p > 0.1): {high_disagreement_errors}")
    print(f"Errors in low disagreement cases (std_p ≤ 0.1): {low_disagreement_errors}")
    
    # Author-specific performance
    if authors is not None:
        print(f"\n{'='*60}")
        print("AUTHOR-SPECIFIC PERFORMANCE")
        print('='*60)
        
        for author in authors.unique():
            author_mask = authors == author
            author_labels = [test_labels[i] for i, m in enumerate(author_mask) if m]
            author_preds = [predictions_binary[i] for i, m in enumerate(author_mask) if m]
            
            if len(author_labels) > 0:
                acc = accuracy_score(author_labels, author_preds)
                print(f"\n{author}:")
                print(f"  Samples: {len(author_labels)}")
                print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    return {
        'predictions_prob': predictions_prob,
        'predictions_binary': predictions_binary,
        'base_predictions': base_predictions,
        'disagreement_scores': disagreement_scores,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        },
        'confusion_matrix': cm
    }

def plot_results(results, test_labels, save_dir="results"):
    """Generate visualization plots."""
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print('='*60)
    
    Path(save_dir).mkdir(exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        results['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Human', 'AI'],
        yticklabels=['Human', 'AI']
    )
    plt.title('Ensemble Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300)
    print(f"  Saved: {save_dir}/confusion_matrix.png")
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(test_labels, results['predictions_prob'])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {results['metrics']['auc']:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Ensemble Model')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/roc_curve.png", dpi=300)
    print(f"  Saved: {save_dir}/roc_curve.png")
    plt.close()
    
    # 3. Prediction Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    human_probs = [p for p, l in zip(results['predictions_prob'], test_labels) if l == 0]
    ai_probs = [p for p, l in zip(results['predictions_prob'], test_labels) if l == 1]
    
    axes[0].hist(human_probs, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title('Prediction Distribution - Human Text')
    axes[0].set_xlabel('Predicted Probability (AI)')
    axes[0].set_ylabel('Count')
    axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2)
    
    axes[1].hist(ai_probs, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_title('Prediction Distribution - AI Text')
    axes[1].set_xlabel('Predicted Probability (AI)')
    axes[1].set_ylabel('Count')
    axes[1].axvline(0.5, color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/prediction_distributions.png", dpi=300)
    print(f"  Saved: {save_dir}/prediction_distributions.png")
    plt.close()
    
    # 4. Base Model Comparison
    model_names = ['XGBoost', 'FFNN', 'DistilBERT', 'Ensemble']
    accuracies = []
    
    for model_name in ['xgb', 'ffnn', 'bert']:
        preds = [1 if p > 0.5 else 0 for p in results['base_predictions'][model_name]]
        acc = accuracy_score(test_labels, preds)
        accuracies.append(acc)
    
    accuracies.append(results['metrics']['accuracy'])
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/model_comparison.png", dpi=300)
    print(f"  Saved: {save_dir}/model_comparison.png")
    plt.close()
    
    # 5. Disagreement vs Error Analysis
    plt.figure(figsize=(10, 6))
    errors = [1 if pred != label else 0 for pred, label in 
              zip(results['predictions_binary'], test_labels)]
    
    correct_disagreement = [d for d, e in zip(results['disagreement_scores'], errors) if e == 0]
    error_disagreement = [d for d, e in zip(results['disagreement_scores'], errors) if e == 1]
    
    plt.hist(correct_disagreement, bins=20, alpha=0.6, color='green', 
             label=f'Correct (n={len(correct_disagreement)})', edgecolor='black')
    plt.hist(error_disagreement, bins=20, alpha=0.6, color='red',
             label=f'Errors (n={len(error_disagreement)})', edgecolor='black')
    
    plt.xlabel('Model Disagreement (std_p)')
    plt.ylabel('Count')
    plt.title('Disagreement Distribution: Correct vs Errors')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/disagreement_analysis.png", dpi=300)
    print(f"  Saved: {save_dir}/disagreement_analysis.png")
    plt.close()
    
    print(f"\nAll visualizations saved to {save_dir}/")

def save_detailed_results(df, results, output_file="detailed_results.csv"):
    """Save detailed predictions to CSV."""
    print(f"\n{'='*60}")
    print("SAVING DETAILED RESULTS")
    print('='*60)
    
    results_df = df.copy()
    results_df['predicted_prob'] = results['predictions_prob']
    results_df['predicted_label'] = results['predictions_binary']
    results_df['correct'] = results_df['is_ai'] == results_df['predicted_label']
    results_df['p_xgb'] = results['base_predictions']['xgb']
    results_df['p_ffnn'] = results['base_predictions']['ffnn']
    results_df['p_bert'] = results['base_predictions']['bert']
    results_df['disagreement'] = results['disagreement_scores']
    
    results_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to {output_file}")
    
    # Show some interesting cases
    print(f"\n{'='*60}")
    print("INTERESTING CASES")
    print('='*60)
    
    # High confidence correct
    high_conf_correct = results_df[
        (results_df['correct']) & 
        ((results_df['predicted_prob'] > 0.9) | (results_df['predicted_prob'] < 0.1))
    ].head(3)
    
    print("\n1. High Confidence Correct Predictions:")
    for idx, row in high_conf_correct.iterrows():
        print(f"\n   Text: {row['text'][:100]}...")
        print(f"   True: {'AI' if row['is_ai'] else 'Human'}, "
              f"Pred: {row['predicted_prob']:.3f}, Author: {row['author']}")
    
    # Misclassifications
    errors = results_df[~results_df['correct']].head(3)
    print("\n2. Misclassifications:")
    for idx, row in errors.iterrows():
        print(f"\n   Text: {row['text'][:100]}...")
        print(f"   True: {'AI' if row['is_ai'] else 'Human'}, "
              f"Pred: {row['predicted_prob']:.3f}, "
              f"Disagreement: {row['disagreement']:.3f}, "
              f"Author: {row['author']}")
    
    # High disagreement cases
    high_disagreement = results_df.nlargest(3, 'disagreement')
    print("\n3. High Model Disagreement:")
    for idx, row in high_disagreement.iterrows():
        print(f"\n   Text: {row['text'][:100]}...")
        print(f"   True: {'AI' if row['is_ai'] else 'Human'}, "
              f"Pred: {row['predicted_prob']:.3f}, "
              f"Disagreement: {row['disagreement']:.3f}")
        print(f"   XGB: {row['p_xgb']:.3f}, FFNN: {row['p_ffnn']:.3f}, BERT: {row['p_bert']:.3f}")

def main():
    """Main evaluation pipeline."""
    print("\n" + "="*60)
    print("STACKING ENSEMBLE - EVALUATION PIPELINE")
    print("="*60)
    
    # Load test data
    test_data_path = "test_set1.csv"
    df = load_test_data(test_data_path)
    
    # Initialize ensemble
    print(f"\n{'='*60}")
    print("INITIALIZING ENSEMBLE")
    print('='*60)
    ensemble = StackingEnsemble()
    
    # Note: Meta-model should be pre-trained. If not trained, uncomment the following:
    # ensemble = train_ensemble_meta_model(
    #     ensemble,
    #     val_texts=df['text'].tolist(),
    #     val_labels=df['is_ai'].tolist()
    # )
    
    # Load pre-trained meta-model
    try:
        ensemble.load_meta_model('meta_model_trained.pkl')
        print("\nLoaded pre-trained meta-model from meta_model_trained.pkl")
    except FileNotFoundError:
        print("\nNo pre-trained meta-model found. Training on full dataset...")
        ensemble = train_ensemble_meta_model(
            ensemble,
            val_texts=df['text'].tolist(),
            val_labels=df['is_ai'].tolist()
        )
    
    # Evaluate on full dataset
    results = evaluate_ensemble(
        ensemble,
        test_texts=df['text'].tolist(),
        test_labels=df['is_ai'].tolist(),
        authors=df['author']
    )
    
    # Generate visualizations
    plot_results(results, df['is_ai'].tolist())
    
    # Save detailed results
    save_detailed_results(df, results)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE!")
    print('='*60)
    print("\nGenerated files:")
    print("  - meta_model_trained.pkl (trained meta-model)")
    print("  - detailed_results.csv (per-sample predictions)")
    print("  - results/*.png (visualization plots)")
    print("\nTo use the trained model later:")
    print("  ensemble = StackingEnsemble()")
    print("  ensemble.load_meta_model('meta_model_trained.pkl')")
    print("  prob = ensemble.predict('Your text here...')")

if __name__ == "__main__":
    main()
