# Progetto RNN per Predizione Cambio Gomme F1 - Tracciamento Steps

## Obiettivo
Creare una rete neurale RNN consistente e utile per prevedere il momento di cambio gomme in Formula 1.

## Steps Completati

### 1. Setup Iniziale del Progetto
- ✅ Creata struttura directory per Google Colab Pro
- ✅ Preparati file di configurazione ottimizzati
- ✅ Creato script di setup automatico per Colab
- ✅ Implementata architettura LSTM multi-task

### 2. Preparazione Dati
- ✅ Creato unificatore dati completo per consolidare dataset F1
- ✅ Gestione di multiple sorgenti dati (anni 2019, 2023, 2024)
- ✅ Implementata gestione feature con normalizzazione
- ✅ Corretti path nel file di configurazione `data_config_unified.json` per l'esecuzione locale:
  - Modificati i percorsi delle `source_directories` (`domenicoDL`, `Vincenzo/processed_races`) per puntare a directory locali relative (modifica poi annullata per focus su Colab).
  - Aggiornati i percorsi di `target_directory` (output e backup) a sottocartelle locali in `colab_models/data/` (modifica poi annullata per focus su Colab).
  - Aggiornato `log_file` a `colab_models/results/` (modifica poi annullata per focus su Colab).
  - Modificati i percorsi in `colab_specific` (`drive_mount_path`, `temp_directory`, `checkpoint_file`) per l'ambiente locale (modifica poi annullata per focus su Colab).
- ✅ Configurazione specifica per Google Colab:
  - Creato `colab_models/configs/data_config_colab.json` con percorsi Google Drive:
    - Unica sorgente dati `"drive_processed"`: `/content/drive/MyDrive/F1_Project/processed_races/*.parquet`.
    - Output, log e checkpoint salvati in sottocartelle di `/content/drive/MyDrive/F1_Project/`.
  - Modificato `colab_models/notebooks/01_quick_start_pro.ipynb` per usare `data_config_colab.json`.
  - Modificato `colab_models/data/data_unifier_complete.py`:
    - Rinominata `load_vincenzo_data` in `load_data_from_drive`.
    - Aggiornata logica per usare la chiave `"drive_processed"` e la funzione `load_data_from_drive`.

### 3. Architettura Modello
- ✅ LSTM multi-task con shared trunk
- ✅ Heads specifiche per: classificazione pit stop, regressione lap rimanenti, classificazione strategia
- ✅ Implementato dropout e batch normalization

### 4. Training Pipeline
- ✅ Creato trainer ottimizzato per Colab Pro
- ✅ Implementato checkpoint automatico ogni 30 minuti
- ✅ Gestione resume training dopo interruzioni
- ✅ Early stopping e learning rate scheduling

### 5. Inference e Testing
- ✅ Creato script per test su nuovi dati
- ✅ Supporto per predizioni real-time

### 6. Fix Formattazione Notebook Colab (Tentativo #2) ✅

#### Problema Iniziale
Il file `colab_models/notebooks/01_quick_start_pro.ipynb` non veniva letto correttamente da Google Colab perché:
- Non era in formato JSON valido per Jupyter Notebook.
- Conteneva marcatori di testo (`# %% [markdown]`, `Text cell <undefined>`, ecc.) invece della struttura JSON richiesta.
- Mancava una cella iniziale per il montaggio di Google Drive.

#### Soluzione Implementata (Tentativo #2)
✅ Ricostruito il file `01_quick_start_pro.ipynb` come JSON valido:
1. **Struttura JSON Corretta**: Utilizzata la struttura standard `.ipynb` con `nbformat`, `metadata` e un array di `cells`.
2. **Montaggio Google Drive**: Inserita una cella di codice all'inizio del notebook per montare Google Drive (`from google.colab import drive; drive.mount('/content/drive')`).
3. **Conversione Celle**: Ogni blocco di testo e codice dal formato testuale fornito è stato convertito in una cella JSON appropriata (`markdown` o `code`).
4. **Rimozione Marcatori**: Eliminati tutti i marcatori testuali non standard (es. `Text cell <title>`, `# %% [code]`) dal contenuto sorgente delle celle.
5. **Metadata Standard**: Inclusi metadata di base per `kernelspec` e `language_info`, oltre ai metadati specifici di Colab come `accelerator: GPU`.
6. **Robustezza Aggiunta**: Inseriti controlli `if 'variabile' in locals()` e `os.path.exists()` per migliorare la robustezza delle celle di codice durante l'esecuzione sequenziale su Colab.
7. **Output Dettagliati**: Migliorati i messaggi di print per fornire un feedback più chiaro durante l'esecuzione delle celle.

## Prossimi Steps

1. **Testing**: Verificare che il notebook funzioni correttamente su Colab
2. **Ottimizzazione**: Assicurarsi che tutti i path e le dipendenze siano corretti
3. **Documentazione**: Aggiornare README con istruzioni per l'uso
4. **Deploy**: Caricare su Google Drive e testare l'intero workflow

## Note Tecniche

### Struttura JSON Notebook Corretta
```json
{
  "cells": [
    {
      "cell_type": "markdown|code",
      "metadata": {},
      "source": ["contenuto"],
      "outputs": []  // solo per code cells
    }
  ],
  "metadata": {
    "kernelspec": {...},
    "language_info": {...},
    "colab": {"name": "...", "provenance": []},
    "accelerator": "GPU"
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

### Formati Celle
- **Markdown**: Per testo, titoli, spiegazioni
- **Code**: Per codice Python eseguibile

## Metriche Target
- Accuracy classificazione pit stop: > 85%
- MAE predizione lap rimanenti: < 3 lap
- Training time su Colab Pro: 4-6 ore

## Changelog
- 30/05/2025: Risolto problema formattazione notebook Colab (Tentativo #1)
- 30/05/2025: Secondo tentativo di fix formattazione notebook Colab, aggiunta cella mount Drive e miglioramenti robustezza (Tentativo #2)
- 30/05/2025: Aggiornato `colab_models/configs/data_config_unified.json` per utilizzare percorsi locali (poi annullato).
- 30/05/2025: Creato `data_config_colab.json` e aggiornati script e notebook per l'esecuzione su Google Colab con dati da Drive.
