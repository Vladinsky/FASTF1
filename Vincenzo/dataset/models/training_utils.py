"""
Training Pipeline for LSTM Tire Change Prediction
================================================

Pipeline completo di training con:
- Multi-task training loop con weighted loss
- Early stopping e learning rate scheduling
- Model checkpointing e best model tracking
- TensorBoard logging per monitoring
- Threshold optimization per target recall
- Evaluation metrics comprehensive
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, confusion_matrix, precision_recall_curve
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class EarlyStopping:
    """
    Early stopping per prevenire overfitting
    
    Monitors validation metric e ferma training se non migliora
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'max',  # 'max' per F1-score, 'min' per loss
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.patience_counter = 0
        self.best_weights = None
        self.should_stop = False
        
    def step(self, metric: float, model: nn.Module) -> bool:
        """
        Check if should stop training
        
        Args:
            metric: Current validation metric
            model: Model to save weights from
            
        Returns:
            True if should stop, False otherwise
        """
        
        if self.mode == 'max':
            improved = metric > self.best_metric + self.min_delta
        else:
            improved = metric < self.best_metric - self.min_delta
            
        if improved:
            self.best_metric = metric
            self.patience_counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.patience:
            self.should_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                print(f"🔄 Restored best weights (metric: {self.best_metric:.4f})")
                
        return self.should_stop


class TrainingMetrics:
    """
    Calcola e traccia metriche di training e validation
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.reset()
        
    def reset(self):
        """Reset accumulatori metriche"""
        self.predictions = []
        self.targets = []
        self.losses = []
        
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: float
    ):
        """
        Aggiorna metriche con nuovo batch
        
        Args:
            predictions: Predicted probabilities (0-1)
            targets: True binary targets
            loss: Loss value for this batch
        """
        # Converti a numpy per sklearn
        pred_np = predictions.detach().cpu().numpy().flatten()
        target_np = targets.detach().cpu().numpy().flatten()
        
        self.predictions.extend(pred_np)
        self.targets.extend(target_np)
        self.losses.append(loss)
        
    def compute_metrics(self, threshold: float = 0.5) -> Dict[str, float]:
        """
        Calcola tutte le metriche
        
        Args:
            threshold: Threshold per binary classification
            
        Returns:
            Dict con tutte le metriche
        """
        if len(self.predictions) == 0:
            return {}
            
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Binary predictions
        binary_preds = (preds >= threshold).astype(int)
        
        # Calcola metriche
        metrics = {
            'loss': np.mean(self.losses),
            'f1_score': f1_score(targets, binary_preds, zero_division=0),
            'precision': precision_score(targets, binary_preds, zero_division=0),
            'recall': recall_score(targets, binary_preds, zero_division=0),
            'roc_auc': roc_auc_score(targets, preds) if len(np.unique(targets)) > 1 else 0.0,
            'pr_auc': average_precision_score(targets, preds) if len(np.unique(targets)) > 1 else 0.0
        }
        
        # Confusion matrix
        cm = confusion_matrix(targets, binary_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0
            })
        
        return metrics


def find_optimal_threshold(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_recall: float = 0.8,
    metric: str = 'f1'
) -> Tuple[float, Dict[str, float]]:
    """
    Trova threshold ottimale che massimizza metrica mantenendo recall target
    
    Args:
        predictions: Predicted probabilities
        targets: True binary targets  
        target_recall: Minimum recall required
        metric: Metric to optimize ('f1', 'precision')
        
    Returns:
        Tuple (optimal_threshold, metrics_at_threshold)
    """
    
    # Calcola precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(targets, predictions)
    
    # Trova thresholds che soddisfano recall constraint
    valid_indices = recalls >= target_recall
    
    if not np.any(valid_indices):
        print(f"⚠️  Warning: No threshold achieves recall >= {target_recall}")
        # Usa threshold che massimizza recall
        best_idx = np.argmax(recalls)
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    else:
        # Tra i valid, trova quello che massimizza la metrica
        valid_precisions = precisions[valid_indices]
        valid_recalls = recalls[valid_indices]
        valid_thresholds = thresholds[valid_indices[:-1]]  # thresholds ha 1 elemento in meno
        
        if metric == 'f1':
            # Calcola F1 per ogni threshold valido
            f1_scores = 2 * (valid_precisions[:-1] * valid_recalls[:-1]) / \
                       (valid_precisions[:-1] + valid_recalls[:-1] + 1e-8)
            best_idx = np.argmax(f1_scores)
        elif metric == 'precision':
            best_idx = np.argmax(valid_precisions[:-1])
        else:
            raise ValueError(f"Unsupported metric: {metric}")
            
        optimal_threshold = valid_thresholds[best_idx]
    
    # Calcola metriche al threshold ottimale
    binary_preds = (predictions >= optimal_threshold).astype(int)
    
    metrics = {
        'threshold': optimal_threshold,
        'f1_score': f1_score(targets, binary_preds),
        'precision': precision_score(targets, binary_preds, zero_division=0),
        'recall': recall_score(targets, binary_preds, zero_division=0),
        'roc_auc': roc_auc_score(targets, predictions),
        'pr_auc': average_precision_score(targets, predictions)
    }
    
    return optimal_threshold, metrics


class TireChangeTrainer:
    """
    Trainer principale per il modello LSTM multi-task
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device: str = 'cpu',
        log_dir: str = './logs',
        checkpoint_dir: str = './checkpoints',
        target_recall: float = 0.8
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.target_recall = target_recall
        
        # Setup directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Tracking
        self.current_epoch = 0
        self.best_f1 = 0.0
        self.best_threshold = 0.5
        self.train_history = []
        self.val_history = []
        
        print(f"🚀 Trainer initialized:")
        print(f"   Device: {device}")
        print(f"   Target recall: {target_recall}")
        print(f"   Log dir: {self.log_dir}")
        print(f"   Checkpoint dir: {self.checkpoint_dir}")
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Training per un epoch
        
        Returns:
            Dict con metriche di training
        """
        self.model.train()
        train_metrics = TrainingMetrics(self.device)
        
        total_batches = len(self.train_loader)
        start_time = time.time()
        
        for batch_idx, (sequences, tire_change_targets, tire_type_targets) in enumerate(self.train_loader):
            # Move to device
            sequences = sequences.to(self.device)
            tire_change_targets = tire_change_targets.to(self.device)
            tire_type_targets = tire_type_targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            
            # Calcola loss
            loss_dict = self.loss_fn(
                outputs['tire_change_logits'],
                outputs['tire_type_logits'],
                tire_change_targets,
                tire_type_targets
            )
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            tire_change_probs = torch.sigmoid(outputs['tire_change_logits'])
            train_metrics.update(
                tire_change_probs.squeeze(),
                tire_change_targets.float(),
                loss_dict['total_loss'].item()
            )
            
            # Log progress
            if batch_idx % 50 == 0:
                progress = batch_idx / total_batches * 100
                elapsed = time.time() - start_time
                eta = elapsed / (batch_idx + 1) * (total_batches - batch_idx - 1)
                
                print(f"   Batch {batch_idx:4d}/{total_batches} ({progress:5.1f}%) | "
                      f"Loss: {loss_dict['total_loss'].item():.4f} | "
                      f"ETA: {eta:.1f}s")
                
                # Log to TensorBoard
                global_step = self.current_epoch * total_batches + batch_idx
                self.writer.add_scalar('Training/BatchLoss', loss_dict['total_loss'].item(), global_step)
                self.writer.add_scalar('Training/TireChangeLoss', loss_dict['tire_change_loss'].item(), global_step)
                self.writer.add_scalar('Training/TireTypeLoss', loss_dict['tire_type_loss'].item(), global_step)
        
        # Calcola metriche finali epoch
        epoch_metrics = train_metrics.compute_metrics(threshold=self.best_threshold)
        
        return epoch_metrics
        
    def validate_epoch(self) -> Tuple[Dict[str, float], float]:
        """
        Validation per un epoch con threshold optimization
        
        Returns:
            Tuple (metrics_dict, optimal_threshold)
        """
        self.model.eval()
        val_metrics = TrainingMetrics(self.device)
        
        with torch.no_grad():
            for sequences, tire_change_targets, tire_type_targets in self.val_loader:
                # Move to device
                sequences = sequences.to(self.device)
                tire_change_targets = tire_change_targets.to(self.device)
                tire_type_targets = tire_type_targets.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences)
                
                # Calcola loss
                loss_dict = self.loss_fn(
                    outputs['tire_change_logits'],
                    outputs['tire_type_logits'],
                    tire_change_targets,
                    tire_type_targets
                )
                
                # Update metrics
                tire_change_probs = torch.sigmoid(outputs['tire_change_logits'])
                val_metrics.update(
                    tire_change_probs.squeeze(),
                    tire_change_targets.float(),
                    loss_dict['total_loss'].item()
                )
        
        # Ottimizza threshold
        predictions = np.array(val_metrics.predictions)
        targets = np.array(val_metrics.targets)
        
        optimal_threshold, threshold_metrics = find_optimal_threshold(
            predictions, targets, self.target_recall, metric='f1'
        )
        
        # Add validation loss to threshold metrics
        val_loss = np.mean(val_metrics.losses)
        threshold_metrics['loss'] = val_loss
        
        return threshold_metrics, optimal_threshold
        
    def train(
        self,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        scheduler=None,
        save_frequency: int = 5
    ) -> Dict[str, List]:
        """
        Training loop completo
        
        Args:
            num_epochs: Numero massimo di epochs
            early_stopping_patience: Patience per early stopping
            scheduler: Learning rate scheduler
            save_frequency: Frequenza salvataggio checkpoints
            
        Returns:
            Dict con storia del training
        """
        
        print(f"\n🎯 Starting training for {num_epochs} epochs...")
        print(f"   Early stopping patience: {early_stopping_patience}")
        print(f"   Target recall: {self.target_recall}")
        
        # Setup early stopping
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='max',  # Massimizza F1-score
            restore_best_weights=True
        )
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            print(f"\n📈 Epoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics, optimal_threshold = self.validate_epoch()
            
            # Update best threshold
            if val_metrics['f1_score'] > self.best_f1:
                self.best_f1 = val_metrics['f1_score']
                self.best_threshold = optimal_threshold
            
            # Scheduler step
            if scheduler is not None:
                if hasattr(scheduler, 'step'):
                    if hasattr(scheduler, 'mode'):
                        scheduler.step(val_metrics['f1_score'])  # ReduceLROnPlateau
                    else:
                        scheduler.step()  # Altri scheduler
            
            # Log metriche
            epoch_time = time.time() - epoch_start
            self._log_epoch_metrics(epoch, train_metrics, val_metrics, epoch_time)
            
            # Save checkpoint
            if (epoch + 1) % save_frequency == 0:
                self._save_checkpoint(epoch, val_metrics['f1_score'])
            
            # Early stopping check
            if early_stopping.step(val_metrics['f1_score'], self.model):
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                print(f"   Best F1-score: {early_stopping.best_metric:.4f}")
                break
            
            # Store history
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)
        
        # Save final model
        self._save_final_model()
        
        # Close TensorBoard
        self.writer.close()
        
        print(f"\n✅ Training completed!")
        print(f"   Best F1-score: {self.best_f1:.4f}")
        print(f"   Best threshold: {self.best_threshold:.4f}")
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_f1': self.best_f1,
            'best_threshold': self.best_threshold
        }
    
    def _log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float
    ):
        """Log metriche epoch a console e TensorBoard"""
        
        # Console logging
        print(f"   Time: {epoch_time:.1f}s | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        print(f"   Train - Loss: {train_metrics['loss']:.4f} | "
              f"F1: {train_metrics['f1_score']:.4f} | "
              f"Recall: {train_metrics['recall']:.4f}")
        print(f"   Val   - Loss: {val_metrics['loss']:.4f} | "
              f"F1: {val_metrics['f1_score']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f} | "
              f"Thresh: {val_metrics['threshold']:.4f}")
        
        # TensorBoard logging
        for metric_name, value in train_metrics.items():
            self.writer.add_scalar(f'Training/{metric_name}', value, epoch)
        
        for metric_name, value in val_metrics.items():
            self.writer.add_scalar(f'Validation/{metric_name}', value, epoch)
        
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
    
    def _save_checkpoint(self, epoch: int, f1_score: float):
        """Salva checkpoint del modello"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'f1_score': f1_score,
            'best_threshold': self.best_threshold,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def _save_final_model(self):
        """Salva modello finale con best weights"""
        final_model = {
            'model_state_dict': self.model.state_dict(),
            'best_f1': self.best_f1,
            'best_threshold': self.best_threshold,
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'num_compounds': self.model.num_compounds
            },
            'training_config': {
                'target_recall': self.target_recall,
                'epochs_trained': self.current_epoch + 1
            }
        }
        
        model_path = self.checkpoint_dir / 'best_model.pth'
        torch.save(final_model, model_path)
        
        # Salva anche history come JSON
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'train_history': self.train_history,
                'val_history': self.val_history,
                'best_f1': self.best_f1,
                'best_threshold': self.best_threshold
            }, f, indent=2)
        
        print(f"🏆 Final model saved: {model_path}")
        print(f"📊 Training history saved: {history_path}")


def create_trainer(
    model,
    data_loaders: Dict,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    target_recall: float = 0.8,
    log_dir: str = './logs',
    checkpoint_dir: str = './checkpoints'
) -> TireChangeTrainer:
    """
    Factory function per creare trainer configurato
    
    Args:
        model: Modello LSTM da trainare
        data_loaders: Dict con train/val DataLoader
        learning_rate: Learning rate iniziale
        device: Device di training
        target_recall: Target recall constraint
        log_dir: Directory per TensorBoard logs
        checkpoint_dir: Directory per checkpoints
        
    Returns:
        Trainer configurato e pronto
    """
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Loss function (importa da lstm_architecture.py)
    from .lstm_architecture import CombinedLoss
    loss_fn = CombinedLoss(
        alpha=0.92,  # Peso task primario
        beta=0.08,   # Peso task secondario  
        pos_weight=29.0,  # Class imbalance weight
        device=device
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Massimizza F1-score
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # Crea trainer
    trainer = TireChangeTrainer(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        target_recall=target_recall
    )
    
    return trainer, scheduler


if __name__ == "__main__":
    print("🧪 Testing training utilities...")
    
    # Test threshold optimization
    np.random.seed(42)
    n_samples = 1000
    
    # Simula predictions e targets sbilanciati
    targets = np.random.choice([0, 1], n_samples, p=[0.97, 0.03])  # 3% positive
    predictions = np.random.random(n_samples)
    
    # Migliora predictions per esempi positivi (simula modello che funziona)
    positive_mask = targets == 1
    predictions[positive_mask] += 0.3
    predictions = np.clip(predictions, 0, 1)
    
    # Test threshold optimization
    optimal_threshold, metrics = find_optimal_threshold(
        predictions, targets, target_recall=0.8
    )
    
    print(f"✅ Threshold optimization test:")
    print(f"   Optimal threshold: {optimal_threshold:.4f}")
    print(f"   F1-score: {metrics['f1_score']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    
    print("✅ Training utilities test completed!")
