"""
Evaluation Metrics and Analysis for LSTM Tire Change Prediction
==============================================================

Sistema completo di valutazione con:
- Metriche comprehensive (F1, Precision, Recall, ROC-AUC, PR-AUC)
- Confusion matrix e classification report
- Threshold analysis e curves
- Feature importance analysis
- Model interpretability tools
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
from pathlib import Path
import json


class ModelEvaluator:
    """
    Evaluator completo per modello LSTM multi-task
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        threshold: float = 0.5
    ):
        self.model = model
        self.device = device
        self.threshold = threshold
        self.model.eval()
        
    def evaluate_dataset(
        self,
        data_loader,
        split_name: str = "Test"
    ) -> Dict:
        """
        Valuta modello su un dataset completo
        
        Args:
            data_loader: DataLoader del dataset
            split_name: Nome del split per logging
            
        Returns:
            Dict con tutte le metriche e predizioni
        """
        print(f"\n📊 Evaluating {split_name} Dataset...")
        
        # Accumulatori
        all_predictions = []
        all_probabilities = []
        all_tire_change_targets = []
        all_tire_type_targets = []
        all_tire_type_predictions = []
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (sequences, tire_change_targets, tire_type_targets) in enumerate(data_loader):
                # Move to device
                sequences = sequences.to(self.device)
                tire_change_targets = tire_change_targets.to(self.device)
                tire_type_targets = tire_type_targets.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences)
                
                # Predizioni cambio gomme
                tire_change_probs = torch.sigmoid(outputs['tire_change_logits']).squeeze()
                tire_change_preds = (tire_change_probs >= self.threshold).long()
                
                # Predizioni tipo mescola (solo per esempi positivi)
                tire_type_probs = torch.softmax(outputs['tire_type_logits'], dim=-1)
                tire_type_preds = torch.argmax(tire_type_probs, dim=-1)
                
                # Accumula risultati
                all_predictions.extend(tire_change_preds.cpu().numpy())
                all_probabilities.extend(tire_change_probs.cpu().numpy())
                all_tire_change_targets.extend(tire_change_targets.cpu().numpy())
                all_tire_type_targets.extend(tire_type_targets.cpu().numpy())
                all_tire_type_predictions.extend(tire_type_preds.cpu().numpy())
                
                num_batches += 1
                
                # Progress ogni 100 batch
                if batch_idx % 100 == 0 and batch_idx > 0:
                    print(f"   Processed {batch_idx}/{len(data_loader)} batches...")
        
        # Converti a numpy arrays
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        tire_change_targets = np.array(all_tire_change_targets)
        tire_type_targets = np.array(all_tire_type_targets)
        tire_type_predictions = np.array(all_tire_type_predictions)
        
        # Calcola metriche principali
        metrics = self._compute_comprehensive_metrics(
            predictions, probabilities, tire_change_targets
        )
        
        # Analisi task secondario (tipo mescola)
        tire_type_metrics = self._evaluate_tire_type_task(
            tire_type_predictions, tire_type_targets, tire_change_targets
        )
        
        print(f"✅ {split_name} Evaluation Completed:")
        print(f"   Samples: {len(predictions)}")
        print(f"   Threshold: {self.threshold:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return {
            'metrics': metrics,
            'tire_type_metrics': tire_type_metrics,
            'predictions': predictions,
            'probabilities': probabilities,
            'targets': tire_change_targets,
            'tire_type_predictions': tire_type_predictions,
            'tire_type_targets': tire_type_targets,
            'split_name': split_name,
            'threshold': self.threshold
        }
    
    def _compute_comprehensive_metrics(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcola tutte le metriche per task principale (cambio gomme)
        """
        
        # Metriche di base
        metrics = {
            'f1_score': f1_score(targets, predictions),
            'precision': precision_score(targets, predictions, zero_division=0),
            'recall': recall_score(targets, predictions, zero_division=0),
            'accuracy': np.mean(targets == predictions)
        }
        
        # ROC e PR AUC (solo se ci sono entrambe le classi)
        if len(np.unique(targets)) > 1:
            metrics['roc_auc'] = roc_auc_score(targets, probabilities)
            metrics['pr_auc'] = average_precision_score(targets, probabilities)
        else:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
            })
            
            # Balanced accuracy
            metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
        
        return metrics
    
    def _evaluate_tire_type_task(
        self,
        tire_type_predictions: np.ndarray,
        tire_type_targets: np.ndarray,
        tire_change_targets: np.ndarray
    ) -> Dict:
        """
        Valuta task secondario (tipo mescola) solo per esempi con cambio gomme
        """
        
        # Filtra solo esempi con cambio gomme
        positive_mask = tire_change_targets == 1
        
        if not np.any(positive_mask):
            return {'error': 'No positive tire change examples found'}
        
        positive_type_predictions = tire_type_predictions[positive_mask]
        positive_type_targets = tire_type_targets[positive_mask]
        
        # Calcola accuracy per task secondario
        tire_type_accuracy = np.mean(positive_type_predictions == positive_type_targets)
        
        # Classification report dettagliato
        try:
            class_report = classification_report(
                positive_type_targets,
                positive_type_predictions,
                output_dict=True,
                zero_division=0
            )
        except:
            class_report = {}
        
        return {
            'accuracy': tire_type_accuracy,
            'num_samples': len(positive_type_predictions),
            'classification_report': class_report,
            'unique_true_classes': len(np.unique(positive_type_targets)),
            'unique_pred_classes': len(np.unique(positive_type_predictions))
        }
    
    def plot_evaluation_curves(
        self,
        evaluation_results: Dict,
        save_dir: Optional[str] = None
    ):
        """
        Crea plots di valutazione (ROC, PR curve, confusion matrix, etc.)
        
        Args:
            evaluation_results: Risultati da evaluate_dataset
            save_dir: Directory per salvare plots (opzionale)
        """
        
        probabilities = evaluation_results['probabilities']
        targets = evaluation_results['targets']
        predictions = evaluation_results['predictions']
        split_name = evaluation_results['split_name']
        
        # Setup figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Evaluation - {split_name} Set', fontsize=16, fontweight='bold')
        
        # 1. ROC Curve
        if len(np.unique(targets)) > 1:
            fpr, tpr, _ = roc_curve(targets, probabilities)
            roc_auc = auc(fpr, tpr)
            
            axes[0, 0].plot(fpr, tpr, 'b-', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            axes[0, 0].plot([0, 1], [0, 1], 'r--', lw=1, alpha=0.8)
            axes[0, 0].set_xlabel('False Positive Rate')
            axes[0, 0].set_ylabel('True Positive Rate')
            axes[0, 0].set_title('ROC Curve')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curve
        if len(np.unique(targets)) > 1:
            precision, recall, _ = precision_recall_curve(targets, probabilities)
            pr_auc = average_precision_score(targets, probabilities)
            
            axes[0, 1].plot(recall, precision, 'g-', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
            axes[0, 1].axhline(y=np.mean(targets), color='r', linestyle='--', alpha=0.8, label='Baseline')
            axes[0, 1].set_xlabel('Recall')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_title('Precision-Recall Curve')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confusion Matrix
        cm = confusion_matrix(targets, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('True')
        axes[1, 0].set_title('Confusion Matrix')
        
        # 4. Prediction Distribution
        axes[1, 1].hist(probabilities[targets == 0], bins=50, alpha=0.7, label='No Change', color='red')
        axes[1, 1].hist(probabilities[targets == 1], bins=50, alpha=0.7, label='Tire Change', color='blue')
        axes[1, 1].axvline(x=self.threshold, color='black', linestyle='--', 
                          label=f'Threshold = {self.threshold:.3f}')
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salva se richiesto
        if save_dir:
            save_path = Path(save_dir) / f'evaluation_curves_{split_name.lower()}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Evaluation curves saved: {save_path}")
        
        plt.show()
    
    def analyze_threshold_sensitivity(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Analizza sensibilità del modello al threshold
        
        Args:
            probabilities: Predicted probabilities
            targets: True targets
            thresholds: Thresholds da testare (opzionale)
            
        Returns:
            Dict con risultati analisi
        """
        
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 81)  # 0.1 to 0.9 step 0.01
        
        results = {
            'thresholds': [],
            'f1_scores': [],
            'precisions': [],
            'recalls': [],
            'accuracies': []
        }
        
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            
            results['thresholds'].append(threshold)
            results['f1_scores'].append(f1_score(targets, predictions, zero_division=0))
            results['precisions'].append(precision_score(targets, predictions, zero_division=0))
            results['recalls'].append(recall_score(targets, predictions, zero_division=0))
            results['accuracies'].append(np.mean(targets == predictions))
        
        return results
    
    def plot_threshold_analysis(
        self,
        threshold_results: Dict,
        target_recall: float = 0.8,
        save_dir: Optional[str] = None
    ):
        """
        Plot analisi threshold sensitivity
        
        Args:
            threshold_results: Risultati da analyze_threshold_sensitivity
            target_recall: Target recall per evidenziare thresholds validi
            save_dir: Directory per salvare plot
        """
        
        thresholds = np.array(threshold_results['thresholds'])
        f1_scores = np.array(threshold_results['f1_scores'])
        precisions = np.array(threshold_results['precisions'])
        recalls = np.array(threshold_results['recalls'])
        
        # Find optimal threshold
        valid_indices = recalls >= target_recall
        if np.any(valid_indices):
            valid_f1 = f1_scores[valid_indices]
            valid_thresholds = thresholds[valid_indices]
            optimal_idx = np.argmax(valid_f1)
            optimal_threshold = valid_thresholds[optimal_idx]
            optimal_f1 = valid_f1[optimal_idx]
        else:
            optimal_threshold = thresholds[np.argmax(recalls)]
            optimal_f1 = f1_scores[np.argmax(recalls)]
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        plt.plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1-Score')
        plt.plot(thresholds, precisions, 'g-', linewidth=2, label='Precision')
        plt.plot(thresholds, recalls, 'r-', linewidth=2, label='Recall')
        
        # Highlight target recall line
        plt.axhline(y=target_recall, color='red', linestyle='--', alpha=0.7, 
                   label=f'Target Recall = {target_recall}')
        
        # Highlight optimal threshold
        plt.axvline(x=optimal_threshold, color='black', linestyle='--', alpha=0.7,
                   label=f'Optimal Threshold = {optimal_threshold:.3f}')
        
        # Highlight optimal point
        plt.scatter([optimal_threshold], [optimal_f1], color='black', s=100, zorder=5)
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold Sensitivity Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(thresholds.min(), thresholds.max())
        plt.ylim(0, 1)
        
        # Add text box with optimal metrics
        textstr = f'Optimal Threshold: {optimal_threshold:.3f}\nF1-Score: {optimal_f1:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        if save_dir:
            save_path = Path(save_dir) / 'threshold_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Threshold analysis saved: {save_path}")
        
        plt.show()
        
        return optimal_threshold, optimal_f1


def create_evaluation_report(
    model: nn.Module,
    test_loader,
    threshold: float,
    device: str = 'cpu',
    save_dir: str = './evaluation_results'
) -> Dict:
    """
    Crea report completo di valutazione
    
    Args:
        model: Modello da valutare
        test_loader: DataLoader per test set
        threshold: Threshold per binary classification
        device: Device di computation
        save_dir: Directory per salvare risultati
        
    Returns:
        Dict con report completo
    """
    
    # Setup directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    # Crea evaluator
    evaluator = ModelEvaluator(model, device, threshold)
    
    # Valuta dataset
    results = evaluator.evaluate_dataset(test_loader, "Test")
    
    # Plot curves
    evaluator.plot_evaluation_curves(results, save_dir=save_path)
    
    # Threshold analysis
    threshold_results = evaluator.analyze_threshold_sensitivity(
        results['probabilities'],
        results['targets']
    )
    
    evaluator.plot_threshold_analysis(
        threshold_results,
        target_recall=0.8,
        save_dir=save_path
    )
    
    # Salva report testuale
    report = {
        'model_summary': {
            'threshold': threshold,
            'device': device,
            'test_samples': len(results['targets'])
        },
        'primary_task_metrics': results['metrics'],
        'secondary_task_metrics': results['tire_type_metrics'],
        'threshold_analysis': threshold_results
    }
    
    report_path = save_path / 'evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📄 Evaluation report saved: {report_path}")
    
    return report


if __name__ == "__main__":
    print("🧪 Testing evaluation utilities...")
    
    # Simula dati per test
    np.random.seed(42)
    n_samples = 1000
    
    # Simula targets sbilanciati
    targets = np.random.choice([0, 1], n_samples, p=[0.97, 0.03])
    
    # Simula predictions con correlazione
    probabilities = np.random.random(n_samples)
    positive_mask = targets == 1
    probabilities[positive_mask] += 0.4  # Migliora predictions per positivi
    probabilities = np.clip(probabilities, 0, 1)
    
    # Test threshold analysis
    dummy_evaluator = ModelEvaluator(None, device='cpu', threshold=0.5)
    threshold_results = dummy_evaluator.analyze_threshold_sensitivity(
        probabilities, targets
    )
    
    print(f"✅ Threshold analysis test:")
    print(f"   Best F1-score: {max(threshold_results['f1_scores']):.4f}")
    print(f"   Best threshold: {threshold_results['thresholds'][np.argmax(threshold_results['f1_scores'])]:.4f}")
    
    print("✅ Evaluation utilities test completed!")
