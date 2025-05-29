"""
Training Script Principale per LSTM Tire Change Prediction
==========================================================

Script completo per training del modello RNN multi-task con:
- Caricamento dati preprocessati
- Inizializzazione modello ottimizzato
- Training con early stopping e monitoring
- Valutazione e salvataggio best model
"""

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
import time
import json
import numpy as np

# Aggiungi percorso per import moduli
sys.path.append(str(Path(__file__).parent / 'dataset'))
sys.path.append(str(Path(__file__).parent / 'dataset' / 'models'))

# Import moduli custom
from models.lstm_architecture import LSTMTireChangePredictor, CombinedLoss
from models.data_loaders import create_data_loaders, check_data_distribution
from models.training_utils import create_trainer
from models.evaluation import create_evaluation_report

# Configurazione
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = './dataset/preprocessed'
OUTPUT_DIR = './training_output'
LOG_DIR = './training_output/logs'
CHECKPOINT_DIR = './training_output/checkpoints'
EVALUATION_DIR = './training_output/evaluation'

# Parametri training
TRAINING_CONFIG = {
    'num_epochs': 50,
    'batch_size': 32 if DEVICE == 'cpu' else 64,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
    'target_recall': 0.8,
    'save_frequency': 5
}

# Parametri modello
MODEL_CONFIG = {
    'input_size': 51,
    'hidden_size': 32 if DEVICE == 'cpu' else 64,  # CPU-friendly
    'num_layers': 3,
    'num_compounds': 10,  # SOFT, MEDIUM, HARD, INTERMEDIATE, WET, etc.
    'dropout': 0.3
}


def setup_directories():
    """Crea directory necessarie per output"""
    directories = [OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR, EVALUATION_DIR]
    for dir_path in directories:
        Path(dir_path).mkdir(exist_ok=True, parents=True)
    print(f"✅ Setup directories: {directories}")


def check_data_availability():
    """Verifica disponibilità dati preprocessati"""
    data_path = Path(DATA_DIR)
    
    required_files = [
        'X_train.npy', 'X_val.npy', 'X_test.npy',
        'y_change_train.npy', 'y_change_val.npy', 'y_change_test.npy',
        'y_type_train.npy', 'y_type_val.npy', 'y_type_test.npy'
    ]
    
    missing_files = []
    for file_name in required_files:
        if not (data_path / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ Missing preprocessed data files: {missing_files}")
        print(f"   Run data_preprocessing.py first!")
        return False
    
    print(f"✅ All preprocessed data files found in {data_path}")
    return True


def initialize_model():
    """Inizializza modello LSTM con configurazione ottimizzata"""
    print(f"\n🧠 Initializing LSTM Model...")
    print(f"   Device: {DEVICE}")
    print(f"   Config: {MODEL_CONFIG}")
    
    model = LSTMTireChangePredictor(
        input_size=MODEL_CONFIG['input_size'],
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_layers=MODEL_CONFIG['num_layers'],
        num_compounds=MODEL_CONFIG['num_compounds'],
        dropout=MODEL_CONFIG['dropout'],
        device=DEVICE
    ).to(DEVICE)
    
    # Print model summary
    summary = model.get_model_summary()
    print(f"✅ Model initialized:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    return model


def create_data_loaders_optimized():
    """Crea DataLoader ottimizzati per device"""
    print(f"\n📊 Creating DataLoaders...")
    print(f"   Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"   Device: {DEVICE}")
    
    # Configura DataLoader per device
    if DEVICE == 'cpu':
        # Configurazione CPU-optimized
        data_loaders = create_data_loaders(
            data_dir=DATA_DIR,
            batch_size=TRAINING_CONFIG['batch_size'],
            num_workers=0,  # Single-threaded per CPU
            pin_memory=False,
            augment_positive=True,
            use_weighted_sampling=False,
            device=DEVICE
        )
    else:
        # Configurazione GPU-optimized
        data_loaders = create_data_loaders(
            data_dir=DATA_DIR,
            batch_size=TRAINING_CONFIG['batch_size'],
            num_workers=2,
            pin_memory=True,
            augment_positive=True,
            use_weighted_sampling=True,
            device=DEVICE
        )
    
    # Analizza distribuzione
    print(f"\n📈 Analyzing data distribution...")
    check_data_distribution(data_loaders['train'], "Training")
    check_data_distribution(data_loaders['val'], "Validation")
    
    return data_loaders


def run_training(model, data_loaders):
    """Esegue training completo del modello"""
    print(f"\n🚀 Starting Training...")
    print(f"   Target recall: {TRAINING_CONFIG['target_recall']}")
    print(f"   Max epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"   Early stopping patience: {TRAINING_CONFIG['early_stopping_patience']}")
    
    # Crea trainer
    trainer, scheduler = create_trainer(
        model=model,
        data_loaders=data_loaders,
        learning_rate=TRAINING_CONFIG['learning_rate'],
        device=DEVICE,
        target_recall=TRAINING_CONFIG['target_recall'],
        log_dir=LOG_DIR,
        checkpoint_dir=CHECKPOINT_DIR
    )
    
    # Avvia training
    start_time = time.time()
    
    training_history = trainer.train(
        num_epochs=TRAINING_CONFIG['num_epochs'],
        early_stopping_patience=TRAINING_CONFIG['early_stopping_patience'],
        scheduler=scheduler,
        save_frequency=TRAINING_CONFIG['save_frequency']
    )
    
    training_time = time.time() - start_time
    
    print(f"\n⏱️  Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    
    return training_history


def evaluate_final_model(model, data_loaders, best_threshold):
    """Valutazione finale del modello su test set"""
    print(f"\n🎯 Final Model Evaluation...")
    print(f"   Using best threshold: {best_threshold:.4f}")
    
    # Evaluation completa su test set
    evaluation_report = create_evaluation_report(
        model=model,
        test_loader=data_loaders['test'],
        threshold=best_threshold,
        device=DEVICE,
        save_dir=EVALUATION_DIR
    )
    
    # Salva metriche finali
    final_metrics = evaluation_report['primary_task_metrics']
    
    print(f"🏆 Final Test Results:")
    print(f"   F1-Score: {final_metrics['f1_score']:.4f}")
    print(f"   Precision: {final_metrics['precision']:.4f}")
    print(f"   Recall: {final_metrics['recall']:.4f}")
    print(f"   ROC-AUC: {final_metrics['roc_auc']:.4f}")
    print(f"   True Positives: {final_metrics.get('true_positives', 'N/A')}")
    print(f"   False Positives: {final_metrics.get('false_positives', 'N/A')}")
    
    return evaluation_report


def save_training_summary(training_history, evaluation_report, training_time):
    """Salva summary completo del training"""
    
    summary = {
        'training_config': TRAINING_CONFIG,
        'model_config': MODEL_CONFIG,
        'device': DEVICE,
        'training_time_seconds': training_time,
        'best_metrics': {
            'best_f1': training_history['best_f1'],
            'best_threshold': training_history['best_threshold']
        },
        'final_test_metrics': evaluation_report['primary_task_metrics'],
        'tire_type_metrics': evaluation_report['secondary_task_metrics']
    }
    
    summary_path = Path(OUTPUT_DIR) / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"📄 Training summary saved: {summary_path}")
    
    return summary


def main():
    """Funzione principale di training"""
    print("🔥 LSTM Tire Change Prediction - Training Started")
    print("=" * 60)
    
    # Setup
    setup_directories()
    
    # Check data
    if not check_data_availability():
        print("❌ Exiting: Missing preprocessed data")
        return
    
    # Initialize model
    model = initialize_model()
    
    # Create data loaders
    data_loaders = create_data_loaders_optimized()
    
    # Training
    start_time = time.time()
    training_history = run_training(model, data_loaders)
    training_time = time.time() - start_time
    
    # Final evaluation
    evaluation_report = evaluate_final_model(
        model, data_loaders, training_history['best_threshold']
    )
    
    # Save summary
    summary = save_training_summary(training_history, evaluation_report, training_time)
    
    # Final report
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Best F1-Score: {training_history['best_f1']:.4f}")
    print(f"🎯 Best Threshold: {training_history['best_threshold']:.4f}")
    print(f"⏱️  Total Time: {training_time:.1f}s ({training_time/60:.1f}min)")
    print(f"💾 Output Directory: {OUTPUT_DIR}")
    print(f"🏆 Model saved in: {CHECKPOINT_DIR}")
    print(f"📈 Logs saved in: {LOG_DIR}")
    print(f"📊 Evaluation saved in: {EVALUATION_DIR}")
    
    # Check if target recall achieved
    final_recall = evaluation_report['primary_task_metrics']['recall']
    if final_recall >= TRAINING_CONFIG['target_recall']:
        print(f"✅ TARGET RECALL ACHIEVED: {final_recall:.4f} >= {TRAINING_CONFIG['target_recall']}")
    else:
        print(f"⚠️  Target recall not fully achieved: {final_recall:.4f} < {TRAINING_CONFIG['target_recall']}")
    
    return summary


if __name__ == "__main__":
    try:
        summary = main()
        print("\n✅ Training script completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
