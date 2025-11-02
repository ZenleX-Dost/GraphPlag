"""
Metrics Module

Evaluation metrics for plagiarism detection.
"""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_detection(
    y_true: List[int],
    y_pred: List[int],
    y_scores: List[float]
) -> Dict[str, float]:
    """
    Evaluate plagiarism detection performance.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_scores: Prediction scores [0, 1]
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Add ROC-AUC if we have both classes
    if len(set(y_true)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Specificity
        if (tn + fp) > 0:
            metrics['specificity'] = tn / (tn + fp)
    
    return metrics


def compute_precision_recall_curve(
    y_true: List[int],
    y_scores: List[float],
    thresholds: List[float]
) -> Tuple[List[float], List[float]]:
    """
    Compute precision-recall values for different thresholds.
    
    Args:
        y_true: Ground truth labels
        y_scores: Prediction scores
        thresholds: List of thresholds to evaluate
        
    Returns:
        Tuple of (precision_list, recall_list)
    """
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        precisions.append(precision)
        recalls.append(recall)
    
    return precisions, recalls


def find_optimal_threshold(
    y_true: List[int],
    y_scores: List[float],
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal threshold based on a metric.
    
    Args:
        y_true: Ground truth labels
        y_scores: Prediction scores
        metric: Metric to optimize ('f1', 'precision', 'recall')
        
    Returns:
        Tuple of (optimal_threshold, metric_value)
    """
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return float(best_threshold), float(best_score)


def compute_ranking_metrics(
    similarity_scores: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10
) -> Dict[str, float]:
    """
    Compute ranking metrics for document retrieval.
    
    Args:
        similarity_scores: Similarity scores matrix
        ground_truth: Ground truth plagiarism matrix
        k: Number of top results to consider
        
    Returns:
        Dictionary of ranking metrics
    """
    n = len(similarity_scores)
    
    precisions_at_k = []
    recalls_at_k = []
    
    for i in range(n):
        # Get top-k most similar documents (excluding self)
        scores = similarity_scores[i].copy()
        scores[i] = -1  # Exclude self
        
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        # Count relevant documents in top-k
        relevant = ground_truth[i] == 1
        retrieved_relevant = sum(1 for idx in top_k_indices if relevant[idx])
        total_relevant = sum(relevant) - (1 if relevant[i] else 0)  # Exclude self
        
        # Precision@k
        precision_at_k = retrieved_relevant / k if k > 0 else 0
        precisions_at_k.append(precision_at_k)
        
        # Recall@k
        recall_at_k = retrieved_relevant / total_relevant if total_relevant > 0 else 0
        recalls_at_k.append(recall_at_k)
    
    return {
        f'precision@{k}': np.mean(precisions_at_k),
        f'recall@{k}': np.mean(recalls_at_k),
    }


def print_evaluation_report(metrics: Dict[str, float]):
    """
    Print formatted evaluation report.
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    # Main metrics
    main_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    print("\nMain Metrics:")
    for metric in main_metrics:
        if metric in metrics:
            print(f"  {metric.replace('_', ' ').title():.<30} {metrics[metric]:.4f}")
    
    # Confusion matrix values
    cm_metrics = ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']
    if any(m in metrics for m in cm_metrics):
        print("\nConfusion Matrix:")
        if 'true_positives' in metrics:
            print(f"  True Positives:  {metrics['true_positives']}")
        if 'true_negatives' in metrics:
            print(f"  True Negatives:  {metrics['true_negatives']}")
        if 'false_positives' in metrics:
            print(f"  False Positives: {metrics['false_positives']}")
        if 'false_negatives' in metrics:
            print(f"  False Negatives: {metrics['false_negatives']}")
    
    # Additional metrics
    other_metrics = [k for k in metrics.keys() 
                    if k not in main_metrics + cm_metrics]
    if other_metrics:
        print("\nAdditional Metrics:")
        for metric in other_metrics:
            value = metrics[metric]
            if isinstance(value, (int, float)):
                print(f"  {metric.replace('_', ' ').title():.<30} {value:.4f}")
    
    print("="*60 + "\n")
