"""
Formula 1 LSTM Architecture - Colab Pro Optimized
Architettura LSTM multi-task ottimizzata per Google Colab Pro
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import yaml
from typing import Dict, Tuple, Optional
import logging

class LSTMProArchitecture(nn.Module):
    """
    Architettura LSTM multi-task ottimizzata per Colab Pro
    
    Features:
    - Mixed precision training support
    - Memory efficient implementation
    - Multi-task learning (tire change + tire type)
    - Configurable architecture via YAML
    """
    
    def __init__(self, config_path: str = "configs/model_config_pro.yaml"):
        super(LSTMProArchitecture, self).__init__()
        
        self.config = self._load_config(config_path)
        self.model_config = self.config['model']
        
        # Architecture dimensions
        self.sequence_length = self.model_config['input']['sequence_length']
        self.feature_dim = self.model_config['input']['feature_dim']
        self.hidden_size = self.model_config['lstm']['hidden_size']
        self.num_layers = self.model_config['lstm']['num_layers']
        
        # Build model layers
        self._build_lstm_layers()
        self._build_shared_trunk()
        self._build_task_heads()
        
        # Initialize weights
        self._initialize_weights()
        
        self.logger = self._setup_logging()
        self.logger.info(f"Model initialized with {self._count_parameters():,} parameters")
    
    def _load_config(self, config_path: str) -> Dict:
        """Carica configurazione da file YAML"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Configurazione di default"""
        return {
            'model': {
                'input': {'sequence_length': 10, 'feature_dim': 52},
                'lstm': {'hidden_size': 256, 'num_layers': 3, 'dropout': 0.3},
                'shared_trunk': {'hidden_sizes': [512, 256, 128], 'dropout': 0.4},
                'heads': {
                    'tire_change': {'hidden_sizes': [64, 32], 'dropout': 0.3},
                    'tire_type': {'num_classes': 9, 'hidden_sizes': [64, 32], 'dropout': 0.3}
                }
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger
    
    def _build_lstm_layers(self):
        """Costruisce layer LSTM"""
        lstm_config = self.model_config['lstm']
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=lstm_config['dropout'] if self.num_layers > 1 else 0,
            bidirectional=lstm_config.get('bidirectional', False),
            batch_first=True
        )
        
        # Dropout post-LSTM
        self.lstm_dropout = nn.Dropout(lstm_config['dropout'])
    
    def _build_shared_trunk(self):
        """Costruisce trunk condiviso post-LSTM"""
        trunk_config = self.model_config['shared_trunk']
        hidden_sizes = trunk_config['hidden_sizes']
        
        # Input size dal LSTM
        input_size = self.hidden_size
        if self.model_config['lstm'].get('bidirectional', False):
            input_size *= 2
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization se abilitata
            if trunk_config.get('batch_norm', True):
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            activation = trunk_config.get('activation', 'relu')
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            
            # Dropout
            layers.append(nn.Dropout(trunk_config['dropout']))
            
            prev_size = hidden_size
        
        self.shared_trunk = nn.Sequential(*layers)
        self.trunk_output_size = prev_size
    
    def _build_task_heads(self):
        """Costruisce head specifici per task"""
        heads_config = self.model_config['heads']
        
        # Head per tire change (binary classification)
        self.tire_change_head = self._build_classification_head(
            heads_config['tire_change'], 
            output_size=1,
            task_name="tire_change"
        )
        
        # Head per tire type (multi-class classification)
        num_classes = heads_config['tire_type'].get('num_classes', 9)
        self.tire_type_head = self._build_classification_head(
            heads_config['tire_type'],
            output_size=num_classes,
            task_name="tire_type"
        )
    
    def _build_classification_head(self, head_config: Dict, output_size: int, task_name: str) -> nn.Module:
        """Costruisce un head di classificazione"""
        hidden_sizes = head_config['hidden_sizes']
        dropout = head_config['dropout']
        
        layers = []
        prev_size = self.trunk_output_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer (senza attivazione, verrà applicata nella loss)
        layers.append(nn.Linear(prev_size, output_size))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Inizializza pesi del modello"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    @autocast()
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass del modello
        
        Args:
            x: Input tensor di shape (batch_size, sequence_length, feature_dim)
            return_attention: Se True, ritorna anche attention weights
            
        Returns:
            Dict con outputs per ogni task
        """
        batch_size = x.size(0)
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Usa ultimo output della sequenza
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Applica dropout
        last_output = self.lstm_dropout(last_output)
        
        # Shared trunk
        shared_features = self.shared_trunk(last_output)
        
        # Task-specific heads
        tire_change_logits = self.tire_change_head(shared_features)
        tire_type_logits = self.tire_type_head(shared_features)
        
        outputs = {
            'tire_change_logits': tire_change_logits.squeeze(-1),  # (batch_size,)
            'tire_type_logits': tire_type_logits,  # (batch_size, num_classes)
            'shared_features': shared_features
        }
        
        # Aggiungi attention se richiesto
        if return_attention:
            # Semplice attention sui timesteps LSTM
            attention_weights = self._compute_attention(lstm_out)
            outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def _compute_attention(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """Calcola attention weights sui timesteps"""
        # lstm_out: (batch_size, sequence_length, hidden_size)
        
        # Semplice attention lineare
        attention_scores = torch.mean(lstm_out, dim=-1)  # (batch_size, sequence_length)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        return attention_weights
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Ottiene embeddings dal shared trunk"""
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs['shared_features']
    
    def _count_parameters(self) -> int:
        """Conta parametri totali del modello"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict:
        """Ritorna informazioni sul modello"""
        total_params = self._count_parameters()
        
        # Calcola parametri per componente
        lstm_params = sum(p.numel() for p in self.lstm.parameters() if p.requires_grad)
        trunk_params = sum(p.numel() for p in self.shared_trunk.parameters() if p.requires_grad)
        tire_change_params = sum(p.numel() for p in self.tire_change_head.parameters() if p.requires_grad)
        tire_type_params = sum(p.numel() for p in self.tire_type_head.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'lstm_parameters': lstm_params,
            'shared_trunk_parameters': trunk_params,
            'tire_change_head_parameters': tire_change_params,
            'tire_type_head_parameters': tire_type_params,
            'input_shape': (self.sequence_length, self.feature_dim),
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        }


class LSTMMultiTaskLoss(nn.Module):
    """
    Loss function multi-task per il modello LSTM
    Gestisce sbilanciamento classi e weighted combination
    """
    
    def __init__(self, config_path: str = "configs/model_config_pro.yaml"):
        super(LSTMMultiTaskLoss, self).__init__()
        
        self.config = self._load_config(config_path)
        loss_config = self.config['loss']
        
        # Task weights
        self.alpha = loss_config['task_weights']['alpha']  # tire_change weight
        self.beta = loss_config['task_weights']['beta']    # tire_type weight
        
        # Tire change loss (binary)
        pos_weight = loss_config['tire_change_loss']['pos_weight']
        self.tire_change_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight) if pos_weight else None
        )
        
        # Tire type loss (multi-class)
        label_smoothing = loss_config['tire_type_loss'].get('label_smoothing', 0.0)
        self.tire_type_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )
        
        self.logger = self._setup_logging()
    
    def _load_config(self, config_path: str) -> Dict:
        """Carica configurazione"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {'loss': {'task_weights': {'alpha': 0.92, 'beta': 0.08}}}
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                return_individual: bool = False) -> torch.Tensor:
        """
        Calcola loss combinata
        
        Args:
            predictions: Dict con 'tire_change_logits' e 'tire_type_logits'
            targets: Dict con 'tire_change' e 'tire_type'
            return_individual: Se True, ritorna anche loss individuali
            
        Returns:
            Combined loss (e opzionalmente dict con loss individuali)
        """
        
        # Primary task: tire change
        tire_change_loss = self.tire_change_loss(
            predictions['tire_change_logits'],
            targets['tire_change'].float()
        )
        
        # Secondary task: tire type (solo quando c'è cambio gomme)
        # Maschera per considerare solo i casi con cambio gomme
        change_mask = targets['tire_change'] == 1
        
        if change_mask.sum() > 0:
            tire_type_loss = self.tire_type_loss(
                predictions['tire_type_logits'][change_mask],
                targets['tire_type'][change_mask]
            )
        else:
            # Se non ci sono cambi gomme nel batch, loss = 0
            tire_type_loss = torch.tensor(0.0, device=predictions['tire_change_logits'].device)
        
        # Combined loss
        combined_loss = self.alpha * tire_change_loss + self.beta * tire_type_loss
        
        if return_individual:
            return combined_loss, {
                'tire_change_loss': tire_change_loss,
                'tire_type_loss': tire_type_loss,
                'combined_loss': combined_loss
            }
        
        return combined_loss


def create_model(config_path: str = "configs/model_config_pro.yaml") -> Tuple[nn.Module, nn.Module]:
    """
    Factory function per creare modello e loss
    
    Returns:
        Tuple (model, loss_function)
    """
    model = LSTMProArchitecture(config_path)
    loss_fn = LSTMMultiTaskLoss(config_path)
    
    return model, loss_fn


def test_model():
    """Test del modello con dati dummy"""
    print("Testing LSTM Pro Architecture...")
    
    # Crea modello
    model, loss_fn = create_model()
    
    # Dati dummy
    batch_size = 32
    seq_length = 10
    feature_dim = 52
    
    x = torch.randn(batch_size, seq_length, feature_dim)
    targets = {
        'tire_change': torch.randint(0, 2, (batch_size,)),
        'tire_type': torch.randint(0, 9, (batch_size,))
    }
    
    # Forward pass
    outputs = model(x)
    print(f"Output shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # Loss calculation
    loss = loss_fn(outputs, targets)
    print(f"Loss: {loss.item():.4f}")
    
    # Model info
    info = model.get_model_info()
    print(f"Model info:")
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_model()
