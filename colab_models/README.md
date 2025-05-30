# Formula 1 Tire Change Prediction - Google Colab Edition 🏎️

Versione completa ottimizzata per Google Colab Pro del sistema di predizione cambi gomme F1 con RNN multi-task e inference real-time.

## 🚀 Caratteristiche Principali

### **Colab Pro Optimized Features**
- 🔗 **Google Drive Integration**: Unione automatica dati distribuiti
- 🧠 **Training da Zero**: Pipeline completa per training from scratch
- ⚡ **GPU Acceleration**: Ottimizzato per T4/A100 con mixed precision
- 🎯 **Real-time Inference**: Sistema completo per predizioni live
- 💾 **Memory Management**: Gestione intelligente 25GB RAM Colab Pro

### **Advanced ML Pipeline**
- 🎯 **Multi-task Learning**: Predizione cambio + tipo mescola
- 📊 **Imbalanced Learning**: Weighted loss per target sbilanciato (3.3%)
- 🔄 **Sequence Modeling**: LSTM per dati sequenziali (10 timesteps)
- 📈 **Performance Tracking**: Monitoring completo training + evaluation
- 🎛️ **Hyperparameter Tuning**: Ottimizzazione automatica parametri

### **Production Ready**
- 🚀 **API Server**: REST API per inference in tempo reale
- 📱 **Web Interface**: Dashboard interattivo per demo
- 📊 **Advanced Visualizations**: Analisi predizioni e performance
- 🔧 **Model Versioning**: Gestione automatica checkpoint e versioni

## 📋 Prerequisiti

1. **Google Colab Pro** (raccomandato per GPU A100 e 25GB RAM)
2. **Google Drive** con dati F1 caricati
3. **Account Google** per mount automatico Drive

## 🛠️ Quick Start Guide

### Step 1: Setup Ambiente

```python
# Su Google Colab, esegui questa cella per setup completo
!git clone https://github.com/your-repo/f1-tire-prediction.git
%cd f1-tire-prediction/colab_models

# Setup automatico ambiente
%run setup_colab_pro.py
```

### Step 2: Unione Dati Distribuiti

```python
# I dati sono attualmente distribuiti in più cartelle su Drive
# Lo script li unirà automaticamente in un dataset unico

from data.data_unifier_complete import CompleteDataUnifier

unifier = CompleteDataUnifier()
dataset = unifier.unify_all_data()
print(f"Dataset finale: {len(dataset)} righe")
```

### Step 3: Training Completo

```python
# Training da zero con parametri ottimizzati per Colab Pro
from training.train_from_scratch_pro import ProTrainer

trainer = ProTrainer()
model = trainer.train_complete()
```

### Step 4: Evaluation & Inference

```python
# Valutazione modello
from models.evaluation_advanced import AdvancedEvaluator
evaluator = AdvancedEvaluator()
results = evaluator.evaluate_model(model)

# Setup inference real-time
from inference.real_time_predictor import RealTimePredictor
predictor = RealTimePredictor(model)
```

## 📁 Struttura Progetto

```
colab_models/
├── 📋 README.md                    # Questa guida
├── 🚀 setup_colab_pro.py          # Setup automatico ambiente
├── 📦 requirements_pro.txt         # Dipendenze ottimizzate
│
├── 📊 configs/                     # Configurazioni
│   ├── model_config_pro.yaml      # Config modello Pro
│   ├── training_config_pro.json   # Parametri training
│   ├── data_config_unified.json   # Config unione dati
│   └── inference_config.json      # Config inference
│
├── 🧠 models/                      # Architettura modelli
│   ├── __init__.py
│   ├── lstm_pro_architecture.py   # LSTM ottimizzato Pro
│   ├── training_utils_pro.py      # Training utilities
│   ├── evaluation_advanced.py     # Evaluation completa
│   ├── data_loaders_pro.py        # Data loading Pro
│   └── inference_engine.py        # Engine inference
│
├── 📊 data/                        # Gestione dati
│   ├── data_unifier_complete.py   # Unione dati distribuiti
│   ├── data_preprocessor_pro.py   # Preprocessing Pro
│   ├── data_validator.py          # Validazione dataset
│   └── data_augmentation.py       # Augmentation avanzato
│
├── 🏋️ training/                    # Pipeline training
│   ├── train_from_scratch_pro.py  # Training principale
│   ├── train_notebook_pro.ipynb   # Notebook interattivo
│   ├── hyperparameter_tuning.py   # Tuning automatico
│   └── distributed_training.py    # Training distribuito
│
├── 🔧 utils/                       # Utilities
│   ├── colab_pro_helpers.py       # Helper Colab Pro
│   ├── memory_management_pro.py   # Gestione 25GB RAM
│   ├── drive_integration.py       # Integrazione Drive
│   ├── monitoring_advanced.py     # Monitoring completo
│   └── notification_system.py     # Notifiche avanzate
│
├── 🚀 inference/                   # Sistema inference
│   ├── real_time_predictor.py     # Predizioni real-time
│   ├── web_interface.py           # Interface web
│   ├── api_server.py              # API REST
│   └── visualization_engine.py    # Visualizzazioni
│
└── 📓 notebooks/                   # Notebook completi
    ├── 01_quick_start_pro.ipynb   # Avvio rapido
    ├── 02_data_exploration.ipynb  # Esplorazione dati
    ├── 03_training_complete.ipynb # Training guidato
    ├── 04_model_evaluation.ipynb  # Valutazione modello
    ├── 05_hyperparameter_opt.ipynb # Ottimizzazione
    └── 06_real_time_demo.ipynb    # Demo inference

## 🎯 Features Avanzate

### **Gestione Dati da Google Drive**
- **Fonte unica**: Tutti i dati provengono dalla cartella `/content/drive/MyDrive/F1_Project/processed_races/`
- **Auto-discovery**: Rileva automaticamente tutti i file Parquet (`*.parquet`)
- **Consolidamento**: Unisce tutti i file in un unico dataset
- **Validazione**: Controlli di qualità e consistenza dei dati

### **Variabili Predittive (52+ Features)**

#### **🎯 Target Variables (2)**
- `tire_change_next_lap`: Variabile binaria (0/1) che indica se nel prossimo giro avverrà un cambio gomme
- `next_tire_type`: Tipo di mescola del prossimo stint (9 categorie: SOFT, MEDIUM, HARD, etc.)

#### **⏱️ Features Temporali (6)**
- `lap_progress`: Progresso gara normalizzato (0-1, dove 0=inizio, 1=fine gara)
- `stint_progress`: Rapporto età pneumatico / durata attesa stint per compound
- `position_inverted`: Posizione invertita (21 - Position) per ranking feature
- `is_top_3`: Flag binario se il pilota è nei primi 3 posti
- `is_points_position`: Flag binario se il pilota è in zona punti (top 10)
- `expected_stint_length`: Durata attesa stint basata su compound utilizzato

#### **🏁 Features Performance (8)**
- `laptime_trend_3`: Trend degradazione tempo giro (slope ultimi 3 giri)
- `delta_ahead_trend`: Trend del gap con il pilota davanti
- `tire_degradation_rate`: Velocità di degrado prestazioni pneumatico
- `compound_age_ratio`: Età relativa pneumatico rispetto ad altri dello stesso compound
- `log_delta_ahead`: Gap logaritmico con pilota davanti (gestisce outlier)
- `log_delta_behind`: Gap logaritmico con pilota dietro
- `LapTime`: Tempo giro attuale normalizzato
- `TimeDeltaToDriverAhead/Behind`: Delta temporali con avversari

#### **🌦️ Features Meteorologiche (7)**
- `AirTemp_stability`: Stabilità temperatura aria (varianza rolling 5 giri)
- `TrackTemp_stability`: Stabilità temperatura pista
- `Humidity_stability`: Stabilità umidità
- `WindSpeed_stability`: Stabilità velocità vento
- `difficult_conditions`: Flag condizioni difficili (pioggia, vento forte, umidità alta)
- `temp_delta`: Differenza temperatura pista - aria
- `Rainfall`: Flag presenza pioggia

#### **🏎️ Domain Knowledge F1 (15+)**
- `stint_length_ratio`: Rapporto durata attuale / durata tipica per compound
- `in_pit_window_early/mid/late`: Flag finestre pit-stop tipiche F1 (giri 10-20, 35-45, 55-65)
- `likely_one_stop/two_stop`: Pattern strategia inferita da comportamento stint
- `expected_stint_length_domain`: Durata attesa basata su knowledge F1 (Soft: 15, Medium: 25, Hard: 35 giri)
- `compound_strategy_freq`: Frequency encoding strategia compound utilizzata
- `TyreLife`: Età pneumatico in giri
- `Position`: Posizione attuale in gara (P1=1, P2=2, etc.)
- `Stint`: Numero stint attuale

#### **🏷️ Features Categoriche Encoded (4)**
- `Compound_encoded`: Tipo mescola codificato (0-8 per 9 compound types)
- `Team_encoded`: Team codificato (0-14 per 15 team)
- `Driver_encoded`: Pilota codificato (0-31 per 32 piloti)
- `Location_encoded`: Circuito codificato (0-26 per 27 location)

#### **📈 Features Tecniche Aggiuntive (10+)**
- Sector times, track status, weather details, etc.

### **Training Ottimizzato Pro**
- **Mixed Precision**: 2x velocità training su GPU moderne
- **Dynamic Batch Sizing**: Adatta batch size alla memoria disponibile
- **Gradient Accumulation**: Simula batch più grandi
- **Smart Checkpointing**: Salvataggio automatico ogni 30min

### **Inference Real-time**
- **Sub-100ms latency**: Ottimizzato per predizioni veloci
- **Streaming pipeline**: Gestione dati live
- **REST API**: Integrazione semplice con applicazioni
- **Interactive dashboard**: Demo immediato

## 📊 Dataset e Performance

### **Flusso di Lavoro su Colab**
1. **Setup**: Esegui `setup_colab_pro.py` per configurare l'ambiente
2. **Unificazione**: Unisci i dati da `/content/drive/MyDrive/F1_Project/processed_races/`
3. **Training**: Avvia il training con `train_from_scratch_pro.py`
4. **Analisi**: Visualizza i risultati con i notebook
5. **Inference**: Testa il modello con dati di esempio

### **Path Importanti**
- **Dati**: `/content/drive/MyDrive/F1_Project/processed_races/`
- **Best Model**: `/content/drive/MyDrive/F1_TireChange_Project/models/checkpoints/best_model.pth`

### **Dataset Finale Atteso**
- **Anni coperti**: 2018-2024 (tutti disponibili)
- **Righe totali**: ~150K+ (unendo tutti i dati)
- **Features**: 52+ features da RNN
- **Target balance**: ~3-5% cambi gomme (gestito con weighted loss)

### **Performance Target**
- **Training time**: 4-6 ore su Colab Pro
- **Model accuracy**: >85% con recall >80%
- **Inference latency**: <100ms
- **Memory usage**: <20GB (margine sicurezza su 25GB)

## 🔧 Configurazioni Chiave

### **Memory Management**
```python
# Configurazione ottimizzata per Colab Pro
MEMORY_CONFIG = {
    "max_batch_size": 512,          # Sfrutta 25GB RAM
    "gradient_accumulation": 4,      # Batch virtuali
    "mixed_precision": True,         # FP16 per velocità
    "dynamic_loss_scaling": True     # Stabilità numerica
}
```

### **Training Parameters**
```python
# Parametri ottimizzati per dataset completo
TRAINING_CONFIG = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "pos_weight": 29.0,              # Gestione sbilanciamento
    "alpha_primary": 0.92,           # Weight task primario
    "beta_secondary": 0.08,          # Weight task secondario
    "early_stopping_patience": 15
}
```

## 🚨 Troubleshooting

### **Problemi Comuni**

**❌ Out of Memory**
```python
# Riduci batch size automaticamente
from utils.memory_management_pro import auto_adjust_batch_size
batch_size = auto_adjust_batch_size()
```

**❌ Dati non trovati**
```python
# Verifica mount Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

**❌ Training lento**
```python
# Verifica GPU assignment
import torch
print(f"GPU disponibile: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name()}")
```

## 📱 Demo e Esempi

### **Quick Demo**
```python
# Demo rapida con modello pre-addestrato
%run notebooks/06_real_time_demo.ipynb
```

### **Training Personalizzato**
```python
# Training con hyperparameters custom
from training.train_from_scratch_pro import ProTrainer

trainer = ProTrainer(
    learning_rate=5e-4,
    batch_size=256,
    epochs=50
)
model = trainer.train_complete()
```

### **API Usage**
```python
# Usa modello via API
import requests

response = requests.post('/predict', json={
    'tire_age': 15,
    'lap': 25,
    'position': 3,
    'weather': 'DRY'
})
prediction = response.json()
```

## 🏆 Risultati Attesi

Con questa configurazione ottimizzata per Colab Pro:

✅ **Training completo da zero** in 4-6 ore  
✅ **Gestione dataset completo** (150K+ righe)  
✅ **Inference real-time** (<100ms)  
✅ **Performance elevate** (85%+ accuracy)  
✅ **Monitoring avanzato** con dashboard  
✅ **Production ready** con API e web interface  

---

## 🎉 Ready to Start!

Segui i notebook nella cartella `notebooks/` per una guida step-by-step completa, o esegui direttamente il setup automatico con:

```python
%run setup_colab_pro.py
```

**Happy Racing! 🏁**
