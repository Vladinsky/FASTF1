# Predizione Cambi Gomme F1 - Rete Neurale RNN

## Panoramica Progetto

Questo progetto implementa una rete neurale ricorrente (RNN) multi-task per predire i cambi gomme in Formula 1, utilizzando dati sequenziali di telemetria e gara.

### Obiettivi Principali

1. **Task Primario**: Predire se un pilota cambierà le gomme nel prossimo giro (target recall: 80%)
2. **Task Secondario**: Predire il tipo di mescola da montare (attivo solo quando probabilità cambio > 70%)

## Architettura Modello

### Multi-Task Learning Architecture

```python
class TireChangePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, n_tire_types, num_layers=2):
        super().__init__()
        # Tronco comune - impara rappresentazioni condivise
        self.shared_rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                                 batch_first=True, dropout=0.2)
        
        # Head primario - probabilità cambio gomme (task principale)
        self.change_head = nn.Linear(hidden_size, 1)
        
        # Head secondario - tipo di mescola (task secondario)
        self.tire_type_head = nn.Linear(hidden_size, n_tire_types)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_features)
        rnn_out, _ = self.shared_rnn(x)  # (batch_size, seq_len, hidden_size)
        
        # Usa solo l'ultimo timestep per predizione
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Output heads
        change_logits = self.change_head(last_output)  # (batch_size, 1)
        tire_logits = self.tire_type_head(last_output)  # (batch_size, n_tire_types)
        
        return change_logits, tire_logits
```

### Loss Function Combinata

```python
def combined_loss(change_logits, tire_logits, change_targets, tire_targets, 
                 alpha=0.9, beta=0.1, threshold=0.7):
    """
    Loss function pesata per multi-task learning
    
    Args:
        alpha: Peso task primario (cambio gomme) - default 0.9
        beta: Peso task secondario (tipo gomma) - default 0.1
        threshold: Soglia probabilità per attivare predizione tipo - default 0.7
    """
    # Loss task primario (weighted per class imbalance)
    pos_weight = torch.tensor(19.0)  # Stimato da dataset sbilanciato
    change_loss = F.binary_cross_entropy_with_logits(
        change_logits.squeeze(), change_targets, pos_weight=pos_weight
    )
    
    # Loss task secondario (attivo solo se prob_cambio > threshold)
    change_probs = torch.sigmoid(change_logits.squeeze())
    tire_mask = (change_probs > threshold).float()
    
    tire_loss = F.cross_entropy(tire_logits, tire_targets, reduction='none')
    tire_loss = (tire_loss * tire_mask).mean()
    
    # Loss combinata
    total_loss = alpha * change_loss + beta * tire_loss
    
    return total_loss, change_loss, tire_loss
```

## Strategia per Class Imbalance

### 1. Weighted Loss
- **Problema**: ~5% cambi gomme vs 95% non-cambi nel dataset
- **Soluzione**: pos_weight = ~19 per bilanciare contributi nella loss
- **Effetto**: Ogni cambio gomme "pesa" come 19 non-cambi

### 2. Focal Loss (Opzionale)
- **Obiettivo**: Focus su esempi difficili da classificare
- **Implementazione**: Riduce peso degli esempi facili, aumenta quello dei difficili
- **Parametri**: α=0.25, γ=2.0 (standard)

### 3. Smart Data Augmentation
- **Metodo**: Perturbazione gaussiana minimale delle sequenze positive
- **Parametri**: μ=0, σ=0.01 (1% dei valori)
- **Effetto**: Da 500 → 2000 esempi positivi senza overfitting

## Feature Engineering

### Sequenze Temporali
- **Lunghezza**: 8-12 giri precedenti + giro corrente
- **Features per timestep**: ~30 variabili
- **Sliding window**: Ogni giro genera una sequenza con storia precedente

### Trasformazioni Avanzate

#### Normalizzazioni Relative
```python
# Progresso gara normalizzato
df['lap_progress'] = df['LapNumber'] / df['TotalLaps']  # 0-1

# Posizione relativa (gestisce ritiri)
df['position_normalized'] = (df['Position'] - 1) / (df['drivers_in_race'] - 1)

# Progresso stint pneumatici
df['stint_progress'] = df['TyreLife'] / df['ExpectedStintLength']
```

#### Trasformazioni Logaritmiche
```python
# Comprime grandi distanze, enfatizza piccole
df['log_delta_ahead'] = np.log1p(np.abs(df['TimeDeltaToDriverAhead']))
df['log_delta_behind'] = np.log1p(np.abs(df['TimeDeltaToDriverBehind']))
```

#### Domain Knowledge Features
```python
# Stint ottimali per compound
typical_stint = {'SOFT': 15, 'MEDIUM': 25, 'HARD': 35}
df['stint_length_ratio'] = df['TyreLife'] / df['Compound'].map(typical_stint)

# Finestre pit-stop tipiche
df['in_pit_window'] = ((df['LapNumber'] % 20).between(15, 20)).astype(int)
```

## Pipeline di Training

### 1. Divisione Dataset
```python
# Split temporale per evitare data leakage
train_years = [2018, 2019, 2020, 2021]  # 80%
val_years = [2023]                       # 10%
test_years = [2024]                      # 10%
```

### 2. Metriche di Valutazione
- **Primarie**: Recall (target: 0.8), Precision, F1-Score
- **Curve**: Precision-Recall, ROC-AUC
- **Threshold tuning**: Ottimizzazione per recall target

### 3. Early Stopping
- **Monitor**: Validation F1-Score
- **Patience**: 10 epoche
- **Restore**: Best model weights

## Gestione Variabili Categoriche

### Robust Encoding
```python
class RobustEncoder:
    """Gestisce nuovi valori categorici in test set"""
    def __init__(self, unknown_strategy='other'):
        self.value_to_idx = {}
        self.unknown_strategy = unknown_strategy
    
    def fit_transform(self, train_data):
        unique_values = list(train_data.unique())
        self.value_to_idx = {val: idx for idx, val in enumerate(unique_values)}
        # Categoria speciale per valori non visti
        self.value_to_idx['__OTHER__'] = len(unique_values)
        return self._encode(train_data)
    
    def transform(self, data):
        return self._encode(data)
    
    def _encode(self, data):
        encoded = [self.value_to_idx.get(val, self.value_to_idx['__OTHER__']) 
                  for val in data]
        return F.one_hot(torch.tensor(encoded), len(self.value_to_idx))
```

## Struttura File Progetto

```
Vincenzo/dataset/
├── README.md                          # Questo file
├── dataset.parquet                    # Dataset consolidato
├── data_consolidation.py             # Fase 1: Unione file parquet
├── data_preprocessing.py             # Feature engineering avanzato
├── model_architecture.py            # Definizione modelli RNN
├── training_pipeline.py             # Training loop e validation
├── evaluation_metrics.py            # Metriche e threshold tuning
├── domain_knowledge_features.py     # Features specifiche F1
└── configs/
    ├── model_config.yaml            # Hyperparameters modello
    ├── training_config.yaml         # Parametri training
    └── preprocessing_config.yaml    # Configurazioni preprocessing
```

## Hyperparameters Consigliati

### Modello
- **hidden_size**: 128-256
- **num_layers**: 2-3
- **dropout**: 0.2-0.3
- **sequence_length**: 10

### Training
- **batch_size**: 64-128
- **learning_rate**: 1e-3 (con scheduler)
- **epochs**: 100 (con early stopping)
- **optimizer**: AdamW

### Loss Weights
- **alpha** (task primario): 0.9
- **beta** (task secondario): 0.1
- **pos_weight**: 19.0 (da aggiustare su dataset reale)

## Procedure di Validation

### 1. Threshold Tuning
```python
def find_optimal_threshold(y_pred_proba, y_true, target_recall=0.8):
    """Trova threshold per recall target mantenendo F1 ottimale"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1, best_threshold = 0, 0.5
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        recall = recall_score(y_true, y_pred)
        
        if recall >= target_recall:
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1, best_threshold = f1, thresh
    
    return best_threshold
```

### 2. Cross-Validation Temporale
- Validate su anni futuri per testare generalizzazione
- Monitoring distribuzione features nel tempo
- Detecting concept drift

## Note Implementative

### Gestione Memory
- **Gradient accumulation** per batch size efficaci grandi
- **Mixed precision** (FP16) per training più veloce
- **DataLoader** con num_workers ottimizzato

### Reproducibilità
- **Random seeds** fissi per numpy, torch, random
- **Deterministic** operations quando possibile
- **Version pinning** delle dipendenze

### Monitoraggio
- **Tensorboard** per loss curves e metriche
- **Weights & Biases** per experiment tracking
- **Model checkpointing** ogni epoch

## Troubleshooting Comune

### Gradient Vanishing/Exploding
- **Gradient clipping**: max_norm=1.0
- **Learning rate scheduling**: ReduceLROnPlateau
- **Layer normalization** in RNN se necessario

### Overfitting
- **Dropout**: 0.2-0.4 nei layer densi
- **Weight decay**: 1e-4 nell'optimizer
- **Early stopping** basato su validation

### Class Imbalance
- **Monitoring precision/recall** oltre accuracy
- **Threshold tuning** post-training
- **Stratified sampling** se necessario

## Roadmap Sviluppo

### Fase 1: Setup e Consolidamento ✅
- [x] Struttura progetto
- [ ] Consolidamento dataset
- [ ] EDA preliminare

### Fase 2: Feature Engineering
- [ ] Trasformazioni avanzate
- [ ] Domain knowledge features
- [ ] Sequence preparation

### Fase 3: Modello Baseline
- [ ] Architettura semplice
- [ ] Training pipeline
- [ ] Validation metrics

### Fase 4: Ottimizzazione
- [ ] Hyperparameter tuning
- [ ] Advanced techniques
- [ ] Performance analysis

### Fase 5: Production Ready
- [ ] Model deployment
- [ ] Inference pipeline
- [ ] Monitoring system
