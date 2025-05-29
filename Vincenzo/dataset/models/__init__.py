"""
Models Package for LSTM Tire Change Prediction
==============================================

Questo package contiene tutti i componenti per il modello RNN multi-task:
- Architettura LSTM (lstm_architecture.py)
- DataLoader ottimizzati (data_loaders.py) 
- Pipeline di training (training_utils.py)
- Sistema di valutazione (evaluation.py)
"""

from .lstm_architecture import (
    LSTMTireChangePredictor,
    CombinedLoss,
    load_model_from_config,
    create_model_for_cpu_testing
)

from .data_loaders import (
    TireChangeSequenceDataset,
    create_data_loaders,
    create_cpu_optimized_loaders,
    check_data_distribution
)

from .training_utils import (
    TireChangeTrainer,
    EarlyStopping,
    TrainingMetrics,
    create_trainer,
    find_optimal_threshold
)

from .evaluation import (
    ModelEvaluator,
    create_evaluation_report
)

__version__ = "1.0.0"
__author__ = "Vincenzo - RNN Tire Change Prediction"

# Quick access per imports comuni
__all__ = [
    # Architettura
    'LSTMTireChangePredictor',
    'CombinedLoss',
    'load_model_from_config',
    'create_model_for_cpu_testing',
    
    # Data loading
    'TireChangeSequenceDataset', 
    'create_data_loaders',
    'create_cpu_optimized_loaders',
    'check_data_distribution',
    
    # Training
    'TireChangeTrainer',
    'EarlyStopping',
    'TrainingMetrics',
    'create_trainer',
    'find_optimal_threshold',
    
    # Evaluation
    'ModelEvaluator',
    'create_evaluation_report'
]

def get_model_summary():
    """Ritorna summary del package"""
    return {
        'package': 'tire_change_models',
        'version': __version__,
        'components': {
            'lstm_architecture': 'Multi-task LSTM model with combined loss',
            'data_loaders': 'Optimized DataLoaders with augmentation',
            'training_utils': 'Complete training pipeline with early stopping',
            'evaluation': 'Comprehensive evaluation and analysis tools'
        },
        'features': [
            'Multi-task learning (tire change + tire type)',
            'Class imbalance handling (1:29 ratio)',
            'Threshold optimization for target recall',
            'CPU/GPU adaptive configuration', 
            'TensorBoard integration',
            'Comprehensive evaluation metrics'
        ]
    }


# Factory functions per configurazioni comuni
def create_cpu_model(input_size: int = 52) -> LSTMTireChangePredictor:
    """Crea modello ottimizzato per CPU testing"""
    return create_model_for_cpu_testing(input_size)


def create_gpu_model(
    input_size: int = 52,
    hidden_size: int = 128,
    num_layers: int = 3
) -> LSTMTireChangePredictor:
    """Crea modello ottimizzato per GPU training"""
    return LSTMTireChangePredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_compounds=7,
        dropout=0.3,
        device='cuda'
    )


def quick_setup_cpu(data_dir: str):
    """Setup rapido per testing CPU"""
    # Carica data loaders
    data_loaders = create_cpu_optimized_loaders(data_dir)
    
    # Crea modello
    model = create_cpu_model()
    
    # Crea trainer  
    trainer, scheduler = create_trainer(
        model=model,
        data_loaders=data_loaders,
        learning_rate=0.001,
        device='cpu',
        target_recall=0.8
    )
    
    return model, data_loaders, trainer, scheduler


def quick_setup_gpu(data_dir: str, device: str = 'cuda'):
    """Setup rapido per training GPU"""
    # Carica data loaders (ottimizzati GPU)
    data_loaders = create_data_loaders(
        data_dir=data_dir,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        device=device
    )
    
    # Crea modello GPU
    model = create_gpu_model()
    model = model.to(device)
    
    # Crea trainer
    trainer, scheduler = create_trainer(
        model=model,
        data_loaders=data_loaders,
        learning_rate=0.001,
        device=device,
        target_recall=0.8
    )
    
    return model, data_loaders, trainer, scheduler


if __name__ == "__main__":
    print("🔧 Models Package Initialized")
    summary = get_model_summary()
    print(f"   Version: {summary['version']}")
    print(f"   Components: {len(summary['components'])}")
    print(f"   Features: {len(summary['features'])}")
    print("✅ Ready for tire change prediction!")
