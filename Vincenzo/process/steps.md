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

### 6. Fix Formattazione Notebook Colab ✅

#### Problema Risolto
Il file `colab_models/notebooks/01_quick_start_pro.ipynb` non veniva letto correttamente da Google Colab perché:
- Non era in formato JSON valido per Jupyter Notebook
- Conteneva marcatori di testo invece della struttura JSON richiesta
- Le celle non erano formattate secondo lo standard .ipynb

#### Soluzione Implementata
✅ Convertito il file in formato JSON valido con:
1. Struttura base del notebook (cells, metadata, nbformat)
2. Conversione di ogni sezione in cella appropriata (markdown o code)
3. Mantenimento del contenuto e della sequenza logica
4. Aggiunta metadata per Google Colab (acceleratore GPU)
5. Rimozione di tutti i marcatori di testo non standard

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
- 30/05/2025: Risolto problema formattazione notebook Colab
