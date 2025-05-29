"""
Formula 1 Tire Change Prediction - Training da Zero (Colab Pro)
Sistema di training completo ottimizzato per Google Colab Pro
"""

import os
import sys
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import logging
import gc
import psutil

# Aggiungi path per imports locali
sys.path.append('/content')
sys.path.append('/content/colab_models')

from models.lstm_pro_architecture import LSTMProArchitecture, LSTMMultiTaskLoss
from data.data_unifier_complete import CompleteDataUnifier

class ProTrainer:
    """
    Trainer ottimizzato per Google Colab Pro
    
    Features:
    - Mixed precision training
    - Memory management intelligente
    - Monitoring avanzato
    - Checkpoint automatici
    - Resume da interruzioni
    """
    
    def __init__(self, 
                 config_path: str = "configs/model_config_pro.yaml",
                 project_dir: str = "/content/drive/MyDrive/F1_TireChange_Project"):
        
        self.config = self._load_config(config_path)
        self.project_dir = project_dir
        self.device = self._setup_device()
        self.logger = self._setup_logging()
        
        # Directories
        self.model_dir = os.path.join(project_dir, "models", "checkpoints")
        self.results_dir = os.path.join(project_dir, "results", "training_logs") 
        self.data_dir = os.path.join(project_dir, "data", "unified")
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rates': [], 'epochs': []
        }
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        
        # Model and optimizer (will be initialized in setup)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        
        self.logger.info(f"ProTrainer initialized - Device: {self.device}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Carica configurazione"""
        import yaml
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config not found: {config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Configurazione di default"""
        return {
            'training': {
                'batch_size': 512,
                'gradient_accumulation': 4,
                'optimizer': {'type': 'AdamW', 'lr': 1e-3, 'weight_decay': 1e-4},
                'scheduler': {'type': 'ReduceLROnPlateau', 'patience': 10},
                'early_stopping': {'patience': 15, 'min_delta': 1e-4},
                'mixed_precision': {'enabled': True}
            },
            'colab_pro': {
                'memory': {'max_memory_usage': 0.85, 'cleanup_frequency': 100},
                'monitoring': {'log_frequency': 50, 'checkpoint_frequency': 1800}
            }
        }
    
    def _setup_device(self) -> torch.device:
        """Setup device (GPU se disponibile)"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Ottimizzazioni GPU
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            device = torch.device('cpu')
            self.logger.warning("GPU not available, using CPU")
        
        return device
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def check_memory_usage(self) -> Dict[str, float]:
        """Controlla utilizzo memoria sistema e GPU"""
        # Sistema
        system_memory = psutil.virtual_memory()
        system_usage = system_memory.percent / 100.0
        
        memory_info = {
            'system_usage': system_usage,
            'system_available_gb': system_memory.available / 1e9
        }
        
        # GPU se disponibile
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_reserved(0) / torch.cuda.max_memory_reserved(0)
            memory_info['gpu_usage'] = gpu_memory
            memory_info['gpu_allocated_gb'] = torch.cuda.memory_allocated(0) / 1e9
            memory_info['gpu_reserved_gb'] = torch.cuda.memory_reserved(0) / 1e9
        
        # Cleanup se necessario
        max_usage = self.config['colab_pro']['memory']['max_memory_usage']
        if system_usage > max_usage:
            self.logger.warning(f"High memory usage: {system_usage:.1%}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return memory_info
    
    def load_and_prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Carica e prepara dati per training"""
        self.logger.info("Loading and preparing data...")
        
        # Verifica se dataset unificato esiste
        unified_dataset_path = os.path.join(self.data_dir, "f1_complete_dataset.parquet")
        
        if not os.path.exists(unified_dataset_path):
            self.logger.info("Unified dataset not found, creating it...")
            # Unifica dati
            unifier = CompleteDataUnifier()
            dataset = unifier.unify_all_data()
        else:
            self.logger.info("Loading existing unified dataset...")
            dataset = pd.read_parquet(unified_dataset_path)
        
        self.logger.info(f"Dataset loaded: {len(dataset):,} rows, {len(dataset.columns)} columns")
        
        # Preprocessa dati per RNN
        train_loader, val_loader, test_loader = self._preprocess_for_rnn(dataset)
        
        return train_loader, val_loader, test_loader
    
    def _preprocess_for_rnn(self, dataset: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Preprocessa dataset per RNN (conversione da preprocessing Vincenzo)"""
        # Temporary: usa preprocessing già esistente da Vincenzo se disponibile
        preprocessed_dir = "/content/drive/MyDrive/Vincenzo/dataset/preprocessed"
        
        if os.path.exists(preprocessed_dir):
            self.logger.info("Loading preprocessed data from Vincenzo...")
            
            # Carica dati preprocessati
            X_train = np.load(os.path.join(preprocessed_dir, "X_train.npy"))
            X_val = np.load(os.path.join(preprocessed_dir, "X_val.npy"))
            X_test = np.load(os.path.join(preprocessed_dir, "X_test.npy"))
            
            y_change_train = np.load(os.path.join(preprocessed_dir, "y_change_train.npy"))
            y_change_val = np.load(os.path.join(preprocessed_dir, "y_change_val.npy"))
            y_change_test = np.load(os.path.join(preprocessed_dir, "y_change_test.npy"))
            
            y_type_train = np.load(os.path.join(preprocessed_dir, "y_type_train.npy"))
            y_type_val = np.load(os.path.join(preprocessed_dir, "y_type_val.npy"))
            y_type_test = np.load(os.path.join(preprocessed_dir, "y_type_test.npy"))
            
        else:
            self.logger.error("Preprocessed data not found! Need to implement full preprocessing pipeline")
            raise FileNotFoundError("Preprocessed data not available")
        
        # Converti in tensori
        batch_size = self.config['training']['batch_size']
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_change_train),
            torch.LongTensor(y_type_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_change_val),
            torch.LongTensor(y_type_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_change_test),
            torch.LongTensor(y_type_test)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        self.logger.info(f"Data loaders created:")
        self.logger.info(f"  Train: {len(train_loader)} batches")
        self.logger.info(f"  Val: {len(val_loader)} batches")
        self.logger.info(f"  Test: {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader
    
    def setup_model_and_optimizer(self):
        """Setup modello, optimizer e scheduler"""
        self.logger.info("Setting up model and optimizer...")
        
        # Modello
        self.model = LSTMProArchitecture()
        self.model.to(self.device)
        
        # Loss function
        self.loss_fn = LSTMMultiTaskLoss()
        
        # Optimizer
        opt_config = self.config['training']['optimizer']
        if opt_config['type'] == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay'],
                betas=opt_config.get('betas', [0.9, 0.999])
            )
        
        # Scheduler
        sched_config = self.config['training']['scheduler']
        if sched_config['type'] == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 10),
                min_lr=sched_config.get('min_lr', 1e-6),
                verbose=True
            )
        
        # Model info
        model_info = self.model.get_model_info()
        self.logger.info(f"Model setup complete:")
        self.logger.info(f"  Total parameters: {model_info['total_parameters']:,}")
        self.logger.info(f"  Input shape: {model_info['input_shape']}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Esegue un epoch di training"""
        self.model.train()
        
        total_loss = 0.0
        total_change_loss = 0.0
        total_type_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (X_batch, y_change_batch, y_type_batch) in enumerate(pbar):
            # Move to device
            X_batch = X_batch.to(self.device)
            y_change_batch = y_change_batch.to(self.device)
            y_type_batch = y_type_batch.to(self.device)
            
            # Forward pass con mixed precision
            with autocast(enabled=self.scaler is not None):
                outputs = self.model(X_batch)
                
                targets = {
                    'tire_change': y_change_batch,
                    'tire_type': y_type_batch
                }
                
                loss, loss_dict = self.loss_fn(outputs, targets, return_individual=True)
                
                # Gradient accumulation
                grad_accum = self.config['training'].get('gradient_accumulation', 1)
                loss = loss / grad_accum
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % grad_accum == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                
                if (batch_idx + 1) % grad_accum == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Statistiche
            total_loss += loss.item() * grad_accum
            total_change_loss += loss_dict['tire_change_loss'].item()
            total_type_loss += loss_dict['tire_type_loss'].item()
            total_samples += X_batch.size(0)
            
            # Accuracy
            with torch.no_grad():
                predictions = torch.sigmoid(outputs['tire_change_logits']) > 0.5
                correct_predictions += (predictions == y_change_batch).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{correct_predictions/total_samples:.3f}"
            })
            
            # Memory cleanup periodico
            cleanup_freq = self.config['colab_pro']['memory']['cleanup_frequency']
            if batch_idx % cleanup_freq == 0:
                self.check_memory_usage()
        
        # Risultati epoch
        avg_loss = total_loss / len(train_loader)
        avg_change_loss = total_change_loss / len(train_loader)
        avg_type_loss = total_type_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        return {
            'loss': avg_loss,
            'tire_change_loss': avg_change_loss,
            'tire_type_loss': avg_type_loss,
            'accuracy': accuracy
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Esegue validazione"""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_change_batch, y_type_batch in tqdm(val_loader, desc="Validation"):
                X_batch = X_batch.to(self.device)
                y_change_batch = y_change_batch.to(self.device)
                y_type_batch = y_type_batch.to(self.device)
                
                outputs = self.model(X_batch)
                
                targets = {
                    'tire_change': y_change_batch,
                    'tire_type': y_type_batch
                }
                
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                total_samples += X_batch.size(0)
                
                # Collect predictions for metrics
                predictions = torch.sigmoid(outputs['tire_change_logits'])
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(y_change_batch.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        binary_predictions = (all_predictions > 0.5).astype(int)
        accuracy = accuracy_score(all_targets, binary_predictions)
        
        try:
            auc_roc = roc_auc_score(all_targets, all_predictions)
        except ValueError:
            auc_roc = 0.0
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, binary_predictions, average='binary', zero_division=0
        )
        
        avg_loss = total_loss / len(val_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Salva checkpoint del modello"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.model_dir, f"checkpoint_epoch_{self.current_epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.model_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved at epoch {self.current_epoch}")
        
        # Cleanup old checkpoints (keep last 3)
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Rimuove vecchi checkpoint (mantiene ultimi 3)"""
        checkpoint_files = glob.glob(os.path.join(self.model_dir, "checkpoint_epoch_*.pth"))
        if len(checkpoint_files) > 3:
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            for old_checkpoint in checkpoint_files[:-3]:
                os.remove(old_checkpoint)
    
    def train_complete(self, max_epochs: int = 100) -> Dict:
        """Training completo da zero"""
        self.logger.info("🚀 Starting complete training from scratch...")
        
        start_time = time.time()
        
        # Setup
        train_loader, val_loader, test_loader = self.load_and_prepare_data()
        self.setup_model_and_optimizer()
        
        # Early stopping
        early_stopping_config = self.config['training']['early_stopping']
        patience = early_stopping_config['patience']
        min_delta = early_stopping_config['min_delta']
        epochs_without_improvement = 0
        
        self.logger.info(f"Training for max {max_epochs} epochs")
        self.logger.info(f"Early stopping: patience={patience}, min_delta={min_delta}")
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch + 1
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step(val_metrics['loss'])
            
            # Logging
            self.logger.info(f"Epoch {self.current_epoch}/{max_epochs}")
            self.logger.info(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.3f}")
            self.logger.info(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.3f}")
            self.logger.info(f"  Val F1: {val_metrics['f1']:.3f}, Recall: {val_metrics['recall']:.3f}")
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            self.training_history['epochs'].append(self.current_epoch)
            
            # Check for best model
            is_best = val_metrics['loss'] < self.best_val_loss - min_delta
            if is_best:
                self.best_val_loss = val_metrics['loss']
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Save checkpoint
            checkpoint_freq = self.config['colab_pro']['monitoring']['checkpoint_frequency']
            if self.current_epoch % (checkpoint_freq // 60) == 0:  # Convert seconds to epochs
                self.save_checkpoint(is_best)
            
            # Early stopping
            if epochs_without_improvement >= patience:
                self.logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                break
            
            # Memory check
            memory_info = self.check_memory_usage()
            if memory_info['system_usage'] > 0.95:
                self.logger.warning("Memory usage critical, stopping training")
                break
        
        # Final checkpoint
        self.save_checkpoint(is_best=False)
        
        # Training summary
        total_time = time.time() - start_time
        self.logger.info(f"✅ Training completed in {total_time/3600:.1f} hours")
        
        # Final evaluation on test set
        test_metrics = self.validate_epoch(test_loader)
        self.logger.info(f"Final test metrics:")
        self.logger.info(f"  Test Loss: {test_metrics['loss']:.4f}")
        self.logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.3f}")
        self.logger.info(f"  Test F1: {test_metrics['f1']:.3f}")
        self.logger.info(f"  Test Recall: {test_metrics['recall']:.3f}")
        
        return {
            'training_history': self.training_history,
            'final_test_metrics': test_metrics,
            'best_val_loss': self.best_val_loss,
            'total_training_time': total_time
        }


def main():
    """Funzione principale per uso standalone"""
    trainer = ProTrainer()
    results = trainer.train_complete()
    return results

if __name__ == "__main__":
    main()
