# Formula 1 Data Extraction - Google Colab Edition 🏎️

Versione ottimizzata per Google Colab del sistema di estrazione dati Formula 1 con robustezza avanzata, gestione degli errori granulare e capacità di resume automatico.

## 🚀 Caratteristiche Principali

### **Colab-Optimized Features**
- 🔗 **Google Drive Integration**: Montaggio automatico e salvataggio persistente
- 🔄 **Runtime Disconnection Resilience**: Resume automatico dopo disconnessioni
- 💾 **Memory Management**: Monitoraggio e cleanup automatico della memoria
- 📊 **Progress Widgets**: Interface interattive con progress bar e statistiche
- ⚡ **Error Recovery**: Retry automatico con exponential backoff

### **Error Handling & Tracking**
- 🎯 **Granular Error Logging**: Tracking separato per gare, piloti e giri
- 📝 **Detailed Error Reports**: JSON con traceback completi e timestamp
- 🔍 **Error Classification**: Categorizzazione automatica degli errori
- 📈 **Progress Persistence**: Stato salvato su Drive per resume perfetto

### **Robustezza & Performance**
- 🛡️ **Retry Mechanisms**: Exponential backoff per errori temporanei
- 🧹 **Memory Cleanup**: Garbage collection automatico dopo ogni gara
- 📊 **Real-time Monitoring**: Alerts per uso memoria e performance
- 💿 **Immediate Save**: Ogni gara salvata immediatamente su Drive

## 📋 Prerequisiti

1. **Google Account** con accesso a Google Drive
2. **Google Colab** (gratuito o Pro)
3. **FastF1 Library** (installata automaticamente)

## 🛠️ Setup Rapido

### Step 1: Upload Files su Colab

```python
# Upload dei file necessari su Colab
from google.colab import files
import os

# Crea directory di lavoro
os.makedirs('Vincenzo', exist_ok=True)

# Upload dei file (esegui questa cella e seleziona i file)
uploaded = files.upload()

# Sposta i file nella directory corretta
for filename in uploaded.keys():
    if filename.endswith('.py'):
        !mv {filename} Vincenzo/
    elif filename.endswith('.json'):
        !mv {filename} Vincenzo/
```

### Step 2: Installazione Dipendenze

```python
# Installa le librerie necessarie
!pip install fastf1 pandas numpy pyarrow tqdm psutil

# Verifica installazione
import fastf1
print(f"FastF1 version: {fastf1.__version__}")
```

### Step 3: Configurazione

```python
# Importa e configura l'ambiente
import sys
sys.path.append('/content')

from Vincenzo.data_extraction_colab import main, ColabEnvironment

# Setup ambiente (monterà automaticamente Google Drive)
colab_env = ColabEnvironment()
colab_env.setup_environment()
```

## 🏁 Esecuzione

### **Modalità Automatica (Consigliata)**

```python
# Esecuzione completa con configurazione di default
from Vincenzo.data_extraction_colab import main
main()
```

### **Modalità Personalizzata**

```python
# Con configurazione custom
import sys
sys.argv = ['data_extraction_colab.py', '--config', 'Vincenzo/config_colab.json']

from Vincenzo.data_extraction_colab import main
main()
```

### **Modalità Force Restart**

```python
# Restart completo ignorando progress esistente
import sys
sys.argv = ['data_extraction_colab.py', '--force-restart']

from Vincenzo.data_extraction_colab import main
main()
```

## ⚙️ Configurazione

Il file `config_colab.json` contiene tutte le impostazioni:

```json
{
  "years_to_process": [2023, 2024],
  "races_per_year_test_limit": 3,
  "cache_directory": "ff1_cache",
  "drive_output_dir": "/content/drive/MyDrive/F1_Project/processed_races",
  "log_level": "INFO",
  "data_to_load": {
    "laps": true,
    "weather": true,
    "telemetry": false,
    "messages": false
  }
}
```

### **Parametri Chiave**

- `years_to_process`: Lista anni da processare
- `races_per_year_test_limit`: Limite gare per testing (null = tutte)
- `drive_output_dir`: Directory output su Google Drive
- `log_level`: Livello di logging (DEBUG, INFO, WARNING, ERROR)

## 📁 Struttura Output su Google Drive

```
/content/drive/MyDrive/F1_Project/
├── processed_races/           # File parquet delle gare
│   ├── 2023_01_Bahrain_Grand_Prix.parquet
│   ├── 2023_02_Saudi_Arabian_Grand_Prix.parquet
│   └── ...
├── ff1_cache/                # Cache FastF1 persistente
├── extraction_log.txt        # Log completo dell'estrazione
├── error_log.json           # Log errori strutturato
└── progress_state.json      # Stato per resume automatico
```

## 🔧 Monitoraggio e Debug

### **Widget di Progresso**

Il sistema mostra automaticamente un widget interattivo con:
- Percentuale di completamento
- Numero gare completate/totali
- Conteggio errori per tipo
- Ultimo aggiornamento

### **Log Real-time**

```python
# Visualizza log in tempo reale
!tail -f /content/drive/MyDrive/F1_Project/extraction_log.txt
```

### **Analisi Errori**

```python
# Carica e analizza errori
import json
with open('/content/drive/MyDrive/F1_Project/error_log.json', 'r') as f:
    errors = json.load(f)

# Statistiche errori
print(f"Totale errori: {errors['summary']['total_errors']}")
print(f"Gare fallite: {len(errors['failed_races'])}")
print(f"Piloti con errori: {len(errors['failed_drivers'])}")
```

## 🔄 Resume e Recovery

### **Resume Automatico**

Il sistema riprende automaticamente dall'ultima gara completata:

```python
# Il sistema rileva automaticamente il progresso esistente
# e salta le gare già processate
main()  # Riprende da dove aveva interrotto
```

### **Recovery da Errori**

```python
# Per processare solo le gare fallite
import json
with open('/content/drive/MyDrive/F1_Project/error_log.json', 'r') as f:
    errors = json.load(f)

failed_races = errors['failed_races']
print(f"Gare da riprocessare: {len(failed_races)}")
```

## 🎯 Best Practices

### **Memory Management**
- Il sistema monitora automaticamente la memoria
- Cleanup forzato quando necessario
- Alerts a 85% e 90% di utilizzo

### **Error Handling**
- Retry automatico per errori temporanei
- Skip automatico per errori permanenti
- Log dettagliato per debug

### **Performance**
- Una gara alla volta per efficienza memoria
- Salvataggio immediato su Drive
- Cache persistente per velocità

## 🚨 Troubleshooting

### **Problemi Comuni**

**❌ Drive non montato**
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

**❌ Out of Memory**
```python
# Il sistema gestisce automaticamente, ma puoi forzare cleanup:
import gc
gc.collect()
```

**❌ Sessione disconnessa**
```python
# Semplicemente riavvia main(), il resume è automatico
main()
```

**❌ Errori FastF1**
```python
# Controlla il log errori per dettagli specifici
!cat /content/drive/MyDrive/F1_Project/error_log.json | jq '.failed_races[-5:]'
```

## 📊 Monitoraggio Performance

### **Statistiche Real-time**

```python
# Durante l'esecuzione, il sistema mostra:
# - Memoria utilizzata (GB e %)
# - Velocità di processing (gare/ora)
# - ETA stimato per completamento
# - Errori per categoria
```

### **Post-Processing Analysis**

```python
# Analisi finale dei dati estratti
import pandas as pd
import glob

# Carica tutti i file parquet
files = glob.glob('/content/drive/MyDrive/F1_Project/processed_races/*.parquet')
print(f"File estratti: {len(files)}")

# Esempio caricamento
df = pd.read_parquet(files[0])
print(f"Colonne: {list(df.columns)}")
print(f"Righe: {len(df)}")
```

## 🏆 Features Avanzate

### **Custom Error Handlers**

```python
# Puoi estendere la gestione errori
from Vincenzo.data_extraction_colab import ErrorTracker

error_tracker = ErrorTracker('/path/to/custom/errors.json')
# Aggiungi custom error handlers
```

### **Progress Callbacks**

```python
# Hook personalizzati per progress updates
def custom_progress_callback(completed, total, current_race):
    print(f"Custom: {completed}/{total} - Processing {current_race}")

# Integra nel main loop
```

### **Memory Optimization**

```python
# Configurazione avanzata memoria
colab_env = ColabEnvironment()
colab_env.memory_warning_threshold = 75.0  # Alert più aggressivo
```

---

## 🎉 Ready to Extract!

Con questa configurazione robusta, puoi estrarre grandi quantità di dati F1 su Colab con fiducia, sapendo che il sistema gestirà automaticamente errori, disconnessioni e problemi di memoria.

**Happy Racing! 🏁**
