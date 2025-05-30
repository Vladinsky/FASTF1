import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Aggiungi path per imports locali
sys.path.append('/content')
sys.path.append('/content/colab_models')

from models.lstm_pro_architecture import LSTMProArchitecture, LSTMMultiTaskLoss
from data.data_unifier_complete import CompleteDataUnifier
from data.data_preprocessing import handle_categorical_variables, normalize_features

def test_model_new_data(driver: str, year: int, circuit: str, model_path: str = "/content/drive/MyDrive/F1_TireChange_Project/models/checkpoints/best_model.pth"):
    """
    Testa il modello con nuovi dati e genera grafici.
    """

    # 1. Carica il modello
    model = LSTMProArchitecture()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Carica i dati
    data_dir = "/content/drive/MyDrive/F1_Project/processed_races"
    data_unifier = CompleteDataUnifier()
    all_files = data_unifier.discover_data_files()
    
    # Trova il file corrispondente
    target_file = None
    for source, files in all_files.items():
        for file in files:
            if str(year) in file and circuit in file:
                target_file = file
                break
    
    if target_file is None:
        print(f"File non trovato per {driver}, {year}, {circuit}")
        return

    df = pd.read_parquet(target_file)
    df = df[df['Driver'] == driver]

    # 3. Preprocessa i dati
    # Carica encoders e scaler
    preprocessed_dir = "/content/drive/MyDrive/Vincenzo/dataset/preprocessed"
    encoders = torch.load(os.path.join(preprocessed_dir, "encoders.pkl"))
    scaler = torch.load(os.path.join(preprocessed_dir, "scaler.pkl"))

    # Applica preprocessing
    df, _ = handle_categorical_variables(df, encoders)
    df, _ = normalize_features(df, is_training=False, scaler=scaler)

    # Crea sequenze RNN
    X, _, _, feature_cols = create_sequences_for_rnn(df, sequence_length=10, encoders=encoders)
    X = torch.tensor(X).float()

    # 4. Esegui l'inference
    tire_change_probs = []
    tire_type_probs = []

    with torch.no_grad():
        for i in range(len(X)):
            output = model(X[i].unsqueeze(0))
            tire_change_prob = torch.sigmoid(output['tire_change_logits']).item()
            tire_type_prob = torch.softmax(output['tire_type_logits'], dim=1).cpu().numpy()[0]

            tire_change_probs.append(tire_change_prob)
            tire_type_probs.append(tire_type_prob)

    # 5. Genera i grafici
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Grafico 1: Probabilità cambio gomme
    ax1.plot(tire_change_probs)
    ax1.set_xlabel("Lap Number")
    ax1.set_ylabel("Probabilità Cambio Gomme")
    ax1.set_title(f"Probabilità Cambio Gomme - {driver} - {year} - {circuit}")
    ax1.set_ylim(0, 1)

    # Grafico 2: Probabilità tipi gomme
    ax2.plot(tire_type_probs)
    ax2.set_xlabel("Lap Number")
    ax2.set_ylabel("Probabilità Tipo Gomma")
    ax2.set_title("Probabilità Tipo Gomma per Giro")
    ax2.legend(encoders['Compound'].classes_)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Esempio di utilizzo
    test_model_new_data(driver="VER", year=2024, circuit="Bahrain")
