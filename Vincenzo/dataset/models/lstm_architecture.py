"""
LSTM Multi-task Architecture for Tire Change Prediction
=======================================================

Implementazione del modello LSTM multi-task per predire:
1. Cambio gomme (task primario) - Binary classification
2. Tipo mescola (task secondario) - Multi-class classification

Architettura:
- Shared LSTM trunk (3 layers, bidirectional=False)
- Task-specific heads con dropout e regularization
- Combined loss function con weighted BCE per class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import yaml


class LSTMTireChangePredictor(nn.Module):
    """
    Multi-task LSTM per predizione cambi gomme Formula 1
    
    Args:
        input_size (int): Numero di features per timestep (52)
        hidden_size (int): Dimensione hidden state LSTM (64 CPU, 128 GPU)
        num_layers (int): Numero layer LSTM (3)
        num_compounds (int): Numero tipi mescola (7: SOFT, MEDIUM, HARD, etc.)
        dropout (float): Dropout rate (0.3)
        device (str): 'cpu' o 'cuda'
    """
    
    def __init__(
        self,
        input_size: int = 52,
        hidden_size: int = 64,  # 64 per CPU, 128 per GPU
        num_layers: int = 3,
        num_compounds: int = 7,
        dropout: float = 0.3,
        device: str = 'cpu'
    ):
        super(LSTMTireChangePredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_compounds = num_compounds
        self.device = device
        
        # Shared LSTM trunk
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False  # Unidirezionale per predizione real-time
        )
        
        # Task 1: Cambio gomme (Binary Classification)
        self.tire_change_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)  # Sigmoid applicato in loss
        )
        
        # Task 2: Tipo mescola (Multi-class Classification)
        # Attivo solo quando c'è cambio gomme
        self.tire_type_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, num_compounds)  # Softmax applicato in loss
        )
        
        # Inizializzazione pesi
        self._init_weights()
        
    def _init_weights(self):
        """Inizializzazione Xavier per stabilità training"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                
    def forward(
        self, 
        x: torch.Tensor,
        return_sequences: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass del modello
        
        Args:
            x: Input tensor (batch_size, seq_length, input_size)
            return_sequences: Se True, ritorna output per ogni timestep
            
        Returns:
            Dict con predizioni per entrambi i task
        """
        batch_size, seq_length, _ = x.shape
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if return_sequences:
            # Use all timesteps (per training con teacher forcing)
            lstm_features = lstm_out  # (batch_size, seq_length, hidden_size)
            lstm_features = lstm_features.reshape(-1, self.hidden_size)  # (batch*seq, hidden)
        else:
            # Use only last timestep (per inference)
            lstm_features = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Task 1: Predizione cambio gomme
        tire_change_logits = self.tire_change_head(lstm_features)  # (batch, 1)
        
        # Task 2: Predizione tipo mescola (conditional)
        tire_type_logits = self.tire_type_head(lstm_features)  # (batch, num_compounds)
        
        if return_sequences:
            # Reshape back to sequence format
            tire_change_logits = tire_change_logits.view(batch_size, seq_length, 1)
            tire_type_logits = tire_type_logits.view(batch_size, seq_length, self.num_compounds)
        
        return {
            'tire_change_logits': tire_change_logits,
            'tire_type_logits': tire_type_logits,
            'lstm_features': lstm_features if not return_sequences else lstm_out
        }
    
    def predict_probabilities(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predizione con probabilità (per inference)
        
        Args:
            x: Input tensor (batch_size, seq_length, input_size)
            
        Returns:
            Dict con probabilità per entrambi i task
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, return_sequences=False)
            
            # Probabilità cambio gomme
            tire_change_probs = torch.sigmoid(outputs['tire_change_logits'])
            
            # Probabilità tipo mescola
            tire_type_probs = F.softmax(outputs['tire_type_logits'], dim=-1)
            
            return {
                'tire_change_probs': tire_change_probs,
                'tire_type_probs': tire_type_probs,
                'tire_change_logits': outputs['tire_change_logits'],
                'tire_type_logits': outputs['tire_type_logits']
            }
    
    def get_model_summary(self) -> Dict:
        """Ritorna summary del modello"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'LSTM Multi-task',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'device': self.device
        }


class CombinedLoss(nn.Module):
    """
    Combined loss function per multi-task learning
    
    Loss = α * tire_change_loss + β * tire_type_loss
    
    Con gestione class imbalance e conditional activation
    """
    
    def __init__(
        self,
        alpha: float = 0.92,  # Peso task primario (cambio gomme)
        beta: float = 0.08,   # Peso task secondario (tipo mescola)
        pos_weight: float = 29.0,  # Peso per class imbalance (1:29 ratio)
        device: str = 'cpu'
    ):
        super(CombinedLoss, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.device = device
        
        # Loss per cambio gomme (weighted BCE per class imbalance)
        self.tire_change_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=device)
        )
        
        # Loss per tipo mescola (standard CrossEntropy)
        self.tire_type_loss = nn.CrossEntropyLoss(ignore_index=-1)  # -1 per esempi non applicabili
        
    def forward(
        self,
        tire_change_logits: torch.Tensor,
        tire_type_logits: torch.Tensor,
        tire_change_targets: torch.Tensor,
        tire_type_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calcola combined loss
        
        Args:
            tire_change_logits: Logits per cambio gomme (batch_size, 1)
            tire_type_logits: Logits per tipo mescola (batch_size, num_compounds)
            tire_change_targets: Target cambio gomme (batch_size,) - binary
            tire_type_targets: Target tipo mescola (batch_size,) - class indices
            
        Returns:
            Dict con loss components e total loss
        """
        
        # Task 1: Loss cambio gomme (sempre attivo)
        change_loss = self.tire_change_loss(
            tire_change_logits.squeeze(-1), 
            tire_change_targets.float()
        )
        
        # Task 2: Loss tipo mescola (solo per cambi gomme positivi)
        # Maschera per esempi con cambio gomme
        positive_mask = tire_change_targets.bool()
        
        if positive_mask.sum() > 0:
            # Calcola loss solo per esempi positivi
            type_loss = self.tire_type_loss(
                tire_type_logits[positive_mask],
                tire_type_targets[positive_mask]
            )
        else:
            # Nessun cambio gomme nel batch
            type_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Combined loss
        total_loss = self.alpha * change_loss + self.beta * type_loss
        
        return {
            'total_loss': total_loss,
            'tire_change_loss': change_loss,
            'tire_type_loss': type_loss,
            'alpha': self.alpha,
            'beta': self.beta
        }


def load_model_from_config(config_path: str, device: str = 'cpu') -> LSTMTireChangePredictor:
    """
    Carica modello da file di configurazione
    
    Args:
        config_path: Path al file YAML di configurazione
        device: Device target ('cpu' o 'cuda')
        
    Returns:
        Modello inizializzato
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']['rnn']
    
    # Adatta hidden_size per device
    if device == 'cpu':
        hidden_size = min(model_config['hidden_size'], 64)  # Limita per CPU
    else:
        hidden_size = model_config['hidden_size']
    
    model = LSTMTireChangePredictor(
        input_size=model_config['input_size'],
        hidden_size=hidden_size,
        num_layers=model_config['num_layers'],
        num_compounds=config['model']['heads']['tire_type']['num_classes'],
        dropout=model_config['dropout'],
        device=device
    )
    
    return model.to(device)


def create_model_for_cpu_testing(input_size: int = 52) -> LSTMTireChangePredictor:
    """
    Crea modello ottimizzato per test rapidi su CPU
    
    Args:
        input_size: Numero features di input
        
    Returns:
        Modello leggero per CPU testing
    """
    return LSTMTireChangePredictor(
        input_size=input_size,
        hidden_size=32,  # Ridotto per CPU
        num_layers=2,    # Ridotto per velocità
        num_compounds=7,
        dropout=0.2,
        device='cpu'
    )


if __name__ == "__main__":
    # Test del modello
    print("Testing LSTM Tire Change Predictor...")
    
    # Crea modello per test
    model = create_model_for_cpu_testing()
    print(f"Model Summary: {model.get_model_summary()}")
    
    # Test forward pass
    batch_size, seq_length, input_size = 4, 10, 52
    x = torch.randn(batch_size, seq_length, input_size)
    
    # Forward pass
    outputs = model(x)
    print(f"Tire change logits shape: {outputs['tire_change_logits'].shape}")
    print(f"Tire type logits shape: {outputs['tire_type_logits'].shape}")
    
    # Test prediction
    predictions = model.predict_probabilities(x)
    print(f"Tire change probabilities: {predictions['tire_change_probs'].squeeze()}")
    
    # Test loss function
    loss_fn = CombinedLoss(device='cpu')
    
    # Dummy targets
    tire_change_targets = torch.randint(0, 2, (batch_size,))
    tire_type_targets = torch.randint(0, 7, (batch_size,))
    
    loss_dict = loss_fn(
        outputs['tire_change_logits'],
        outputs['tire_type_logits'],
        tire_change_targets,
        tire_type_targets
    )
    
    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Change loss: {loss_dict['tire_change_loss'].item():.4f}")
    print(f"Type loss: {loss_dict['tire_type_loss'].item():.4f}")
    
    print("✅ Model test completed successfully!")
