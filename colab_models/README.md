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

## 📖 Panoramica della Cartella `colab_models`

Questa cartella (`colab_models/`) è progettata per contenere tutto il necessario per eseguire il progetto di predizione dei cambi gomme F1 interamente su Google Colab. L'obiettivo è rendere questa cartella il più possibile autonoma dal resto della struttura del progetto locale, facilitando l'esecuzione e il testing in un ambiente cloud standardizzato.

**Componenti Chiave:**
- **Configurazioni (`configs/`)**: Contiene i file di configurazione specifici per Colab, come `data_config_colab.json` (che definisce i percorsi dei dati su Google Drive) e `model_config_pro.yaml`.
- **Dati (`data/`)**: Script per la gestione e l'unificazione dei dati (es. `data_unifier_complete.py`). I dati grezzi e processati sono attesi su Google Drive.
- **Modelli (`models/`)**: Architetture dei modelli neurali (es. `lstm_pro_architecture.py`) e utility per il training e la valutazione.
- **Notebooks (`notebooks/`)**: Jupyter notebooks per guidare l'utente attraverso i vari step del progetto (es. `01_quick_start_pro.ipynb`).
- **Training (`training/`)**: Script per avviare e gestire il processo di training dei modelli (es. `train_from_scratch_pro.py`).
- **Inference (`inference/`)**: Script e strumenti per effettuare predizioni con i modelli addestrati.
- **Utility (`utils/`)**: Funzioni di supporto e helper specifici per l'ambiente Colab.
- **`setup_colab_pro.py`**: Script per automatizzare il setup dell'ambiente su Colab.
- **`requirements_pro.txt`**: Elenco delle dipendenze Python necessarie.

**Flusso di Lavoro Tipico su Colab:**
1.  Clonare il repository o caricare la cartella `colab_models` su Google Drive.
2.  Aprire il notebook `notebooks/01_quick_start_pro.ipynb` in Google Colab.
3.  Eseguire le celle del notebook per:
    *   Montare Google Drive.
    *   Eseguire lo script `setup_colab_pro.py` per installare dipendenze e configurare l'ambiente.
    *   Utilizzare `data_unifier_complete.py` (configurato tramite `data_config_colab.json`) per caricare e unificare i dati da `/content/drive/MyDrive/F1_Project/processed_races/`.
    *   Avviare il training del modello, che salverà checkpoint e output su `/content/drive/MyDrive/F1_Project/`.
    *   Valutare il modello e testare l'inference.

L'intera pipeline è pensata per leggere input da Google Drive e scrivere output su Google Drive, mantenendo la portabilità e la riproducibilità su Colab.

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
# I dati verranno caricati da Google Drive come specificato in data_config_colab.json
# Il notebook 01_quick_start_pro.ipynb è già configurato per usare data_config_colab.json

from data.data_unifier_complete import CompleteDataUnifier

# Il notebook chiamerà CompleteDataUnifier con il config corretto:
# unifier = CompleteDataUnifier(config_path="colab_models/configs/data_config_colab.json")
# Per esecuzione manuale o test, assicurarsi di passare il config corretto.
# Qui sotto un esempio di come verrebbe chiamato nel notebook:

# unifier = CompleteDataUnifier(config_path="configs/data_config_colab.json") # Assumendo che il CWD sia colab_models
# dataset = unifier.unify_all_data()
# if dataset is not None:
#     print(f"Dataset finale: {len(dataset)} righe")
# else:
#     print("Errore durante l'unificazione del dataset.")

print("Consultare il notebook 01_quick_start_pro.ipynb per l'unificazione dati.")
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
│   ├── data_config_unified.json   # Config unione dati (per uso locale/generale)
│   ├── data_config_colab.json     # Config unione dati specifica per Colab (USA QUESTA SU COLAB)
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

### **Gestione Dati da Google Drive (Configurazione Colab)**
- **Fonte Dati Primaria**: Tutti i dati di input devono trovarsi su Google Drive nella cartella `/content/drive/MyDrive/F1_Project/processed_races/`.
- **Configurazione**: Il file `colab_models/configs/data_config_colab.json` definisce questa sorgente dati (sotto la chiave `"drive_processed"`) e specifica che i file sono in formato `*.parquet`.
- **Script di Unificazione**: `data_unifier_complete.py` (chiamato dal notebook `01_quick_start_pro.ipynb`) utilizza questa configurazione per:
    - Rilevare automaticamente tutti i file Parquet nella cartella specificata.
    - Consolidare i file in un unico dataset pandas.
    - Eseguire validazioni e preprocessing.
- **Output su Drive**: Il dataset unificato, i log, i checkpoint del modello e i risultati finali vengono salvati in sottocartelle di `/content/drive/MyDrive/F1_Project/` (es. `unified_data/`, `logs/`, `checkpoints/`, `results/`), come definito in `data_config_colab.json`.

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

### **Flusso di Lavoro su Colab (dettagliato)**
1. **Preparazione Google Drive**:
   - Assicurati che i tuoi dati F1 processati (file `.parquet`) siano presenti in `/content/drive/MyDrive/F1_Project/processed_races/`.
   - Crea le cartelle `/content/drive/MyDrive/F1_Project/unified_data/`, `/content/drive/MyDrive/F1_Project/logs/`, `/content/drive/MyDrive/F1_Project/checkpoints/`, `/content/drive/MyDrive/F1_Project/backups/` e `/content/drive/MyDrive/F1_Project/results/` sul tuo Google Drive se non esistono già (anche se gli script dovrebbero crearle se mancano).
2. **Apri Notebook in Colab**: Carica e apri `colab_models/notebooks/01_quick_start_pro.ipynb` in Google Colab.
3. **Esegui Celle Iniziali**:
   - Monta il tuo Google Drive.
   - Clona il repository (se non l'hai già fatto e caricato `colab_models` su Drive) e naviga in `FASTF1/colab_models`.
   - Esegui lo script `setup_colab_pro.py` tramite `%run setup_colab_pro.py`.
4. **Unificazione Dati**:
   - La cella relativa all'unificazione dati nel notebook istanzierà `CompleteDataUnifier` usando `config_path="colab_models/configs/data_config_colab.json"`.
   - Questo assicura che i dati vengano letti da `/content/drive/MyDrive/F1_Project/processed_races/` e che l'output venga salvato in `/content/drive/MyDrive/F1_Project/unified_data/`.
5. **Training**:
   - Le celle di training useranno i dati unificati e salveranno i modelli e i checkpoint in `/content/drive/MyDrive/F1_Project/models/checkpoints/` (o percorso simile definito in `model_config_pro.yaml` e gestito da `ProTrainer`).
6. **Analisi e Inference**: Segui le celle del notebook per analizzare i risultati e testare l'inference.

### **Path Chiave (configurati in `data_config_colab.json`)**
- **Sorgente Dati Parquet**: `/content/drive/MyDrive/F1_Project/processed_races/`
- **Dataset Unificato Output**: `/content/drive/MyDrive/F1_Project/unified_data/f1_complete_dataset_colab.parquet`
- **Log Unificazione**: `/content/drive/MyDrive/F1_Project/logs/data_unification_colab.log`
- **Checkpoint Unificazione**: `/content/drive/MyDrive/F1_Project/checkpoints/unification_checkpoint_colab.pkl`
- **Modelli Addestrati (esempio, il path esatto dipende da `model_config_pro.yaml`)**: `/content/drive/MyDrive/F1_Project/models/checkpoints/best_model.pth`

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
