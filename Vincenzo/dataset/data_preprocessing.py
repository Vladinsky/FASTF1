"""
Data Preprocessing Script - Fase 2
==================================

Feature engineering avanzato per RNN tire change prediction.
Corregge la logica di normalizzazione posizione e aggiunge features derivate.

Autore: Data Science Team  
Data: 2025-05-29
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_dataset(dataset_path: str = "dataset.parquet") -> pd.DataFrame:
    """Carica il dataset consolidato."""
    path = Path(__file__).parent / dataset_path
    if not path.exists():
        raise FileNotFoundError(f"Dataset non trovato: {path}")
    
    df = pd.read_parquet(path)
    logger.info(f"Dataset caricato: {len(df):,} righe, {len(df.columns)} colonne")
    return df

def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea le variabili target per multi-task learning.
    
    Target primario: cambio_gomme_prossimo_giro (binary)
    Target secondario: tipo_mescola_prossima (categorical)
    """
    logger.info("Creazione target variables...")
    
    df_target = df.copy()
    
    # Ordina per garantire sequenza temporale corretta
    df_target = df_target.sort_values(['DriverRaceID', 'LapNumber'])
    
    # Target primario: cambio gomme nel prossimo giro
    # Logica: cambio di stint nel giro successivo
    df_target['NextStint'] = df_target.groupby('DriverRaceID')['Stint'].shift(-1)
    df_target['tire_change_next_lap'] = (
        (df_target['Stint'] != df_target['NextStint']) & 
        df_target['NextStint'].notna()
    ).astype(int)
    
    # Target secondario: tipo mescola nel prossimo stint
    df_target['NextCompound'] = df_target.groupby('DriverRaceID')['Compound'].shift(-1)
    
    # Il target secondario è valido solo quando c'è cambio gomme
    df_target['next_tire_type'] = df_target['NextCompound'].where(
        df_target['tire_change_next_lap'] == 1, 
        'NO_CHANGE'
    )
    
    # Cleanup colonne temporanee
    df_target = df_target.drop(['NextStint', 'NextCompound'], axis=1)
    
    # Statistiche target
    n_changes = df_target['tire_change_next_lap'].sum()
    total = len(df_target)
    logger.info(f"Target creato: {n_changes:,} cambi gomme su {total:,} giri ({n_changes/total*100:.2f}%)")
    
    return df_target

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features temporali e di progresso."""
    logger.info("Creazione features temporali...")
    
    df_temp = df.copy()
    
    # 1. Progresso gara normalizzato (0-1)
    df_temp['lap_progress'] = df_temp.groupby('RaceID')['LapNumber'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    
    # 2. Progresso stint pneumatico
    # Calcola stint length tipico per compound
    compound_stint_avg = df_temp.groupby('Compound')['TyreLife'].quantile(0.75)
    df_temp['expected_stint_length'] = df_temp['Compound'].map(compound_stint_avg)
    df_temp['stint_progress'] = df_temp['TyreLife'] / df_temp['expected_stint_length']
    df_temp['stint_progress'] = df_temp['stint_progress'].clip(0, 2)  # Cap a 200%
    
    # 3. Features posizione CORRETTE (non normalizzate erroneamente)
    # Posizione è già meaningful (P1=1, P2=2, etc.)
    # Aggiungiamo solo features derivate utili
    df_temp['position_inverted'] = 21 - df_temp['Position']  # Per ranking feature
    df_temp['is_top_3'] = (df_temp['Position'] <= 3).astype(int)
    df_temp['is_points_position'] = (df_temp['Position'] <= 10).astype(int)
    
    logger.info("Features temporali create con successo")
    return df_temp

def create_performance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features di performance e trend."""
    logger.info("Creazione features di performance...")
    
    df_perf = df.copy()
    
    # 1. Trend degradazione laptime (rolling window)
    df_perf['laptime_trend_3'] = df_perf.groupby('DriverRaceID')['LapTime'].rolling(
        window=3, min_periods=2
    ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0).reset_index(0, drop=True)
    
    # 2. Trend gap con avversari
    df_perf['delta_ahead_trend'] = df_perf.groupby('DriverRaceID')['TimeDeltaToDriverAhead'].rolling(
        window=3, min_periods=2
    ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0).reset_index(0, drop=True)
    
    # 3. Velocità degradazione pneumatico
    df_perf['tire_degradation_rate'] = df_perf.groupby(['DriverRaceID', 'Stint'])['LapTime'].pct_change()
    df_perf['tire_degradation_rate'] = df_perf['tire_degradation_rate'].fillna(0)
    
    # 4. Età relativa pneumatico per compound
    df_perf['compound_age_ratio'] = df_perf.groupby(['RaceID', 'Compound'])['TyreLife'].transform(
        lambda x: x / x.quantile(0.9) if x.quantile(0.9) > 0 else x
    )
    
    # 5. Trasformazioni logaritmiche per gap (come pianificato)
    df_perf['log_delta_ahead'] = np.log1p(np.abs(df_perf['TimeDeltaToDriverAhead']))
    df_perf['log_delta_behind'] = np.log1p(np.abs(df_perf['TimeDeltaToDriverBehind']))
    
    logger.info("Features di performance create con successo")
    return df_perf

def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features meteorologiche derivate."""
    logger.info("Creazione features meteo...")
    
    df_weather = df.copy()
    
    # 1. Stabilità condizioni meteo (varianza rolling)
    weather_cols = ['AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed']
    
    for col in weather_cols:
        if col in df_weather.columns:
            # Varianza rolling 5 giri per stabilità
            df_weather[f'{col}_stability'] = df_weather.groupby('RaceID')[col].rolling(
                window=5, min_periods=3
            ).std().reset_index(0, drop=True).fillna(0)
    
    # 2. Indice condizioni difficili
    df_weather['difficult_conditions'] = (
        (df_weather['Rainfall'] == True) |
        (df_weather['Humidity'] > 80) |
        (df_weather['WindSpeed'] > 15)
    ).astype(int)
    
    # 3. Delta temperature (differenza air-track)
    df_weather['temp_delta'] = df_weather['TrackTemp'] - df_weather['AirTemp']
    
    logger.info("Features meteo create con successo")
    return df_weather

def create_domain_knowledge_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features basate su domain knowledge F1."""
    logger.info("Creazione domain knowledge features...")
    
    df_domain = df.copy()
    
    # 1. Stint length tipici per compound (basati su dati reali)
    typical_stint_lengths = {
        'SOFT': 15,
        'MEDIUM': 25, 
        'HARD': 35,
        'SUPERSOFT': 12,
        'ULTRASOFT': 10,
        'INTERMEDIATE': 20,
        'WET': 15
    }
    
    df_domain['expected_stint_length_domain'] = df_domain['Compound'].map(
        typical_stint_lengths
    ).fillna(20)  # Default 20 giri
    
    df_domain['stint_length_ratio'] = df_domain['TyreLife'] / df_domain['expected_stint_length_domain']
    
    # 2. Finestre pit-stop tipiche F1
    # Primi stint: giri 10-20, secondi stint: 35-45, etc.
    df_domain['in_pit_window_early'] = (df_domain['LapNumber'].between(10, 20)).astype(int)
    df_domain['in_pit_window_mid'] = (df_domain['LapNumber'].between(35, 45)).astype(int)
    df_domain['in_pit_window_late'] = (df_domain['LapNumber'].between(55, 65)).astype(int)
    
    # 3. Strategia stint (inferita da pattern)
    df_domain['likely_one_stop'] = (df_domain['Stint'] == 1) & (df_domain['stint_length_ratio'] > 1.5)
    df_domain['likely_two_stop'] = (df_domain['Stint'] >= 2) & (df_domain['stint_length_ratio'] < 1.2)
    
    # 4. Compound strategy pattern
    df_domain['compound_strategy'] = df_domain.groupby('DriverRaceID')['Compound'].transform(
        lambda x: '_'.join(x.astype(str).unique())
    )
    
    logger.info("Domain knowledge features create con successo")
    return df_domain

def handle_categorical_variables(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Gestisce variabili categoriche con robust encoding.
    Ritorna DataFrame processato e mappings per inference.
    """
    logger.info("Gestione variabili categoriche...")
    
    df_cat = df.copy()
    encoders = {}
    
    # Variabili categoriche da encodare
    categorical_columns = ['Compound', 'Team', 'Location', 'Driver']
    
    for col in categorical_columns:
        if col in df_cat.columns:
            # Crea label encoder
            le = LabelEncoder()
            
            # Fit su tutti i valori unici, inclusi NaN and 'NO_CHANGE'
            unique_values = df_cat[col].astype(str).unique()
            if 'NO_CHANGE' not in unique_values:
                unique_values = np.append(unique_values, 'NO_CHANGE')
            le.fit(unique_values)

            # Transform
            df_cat[f'{col}_encoded'] = le.transform(df_cat[col].astype(str))

            # Salva encoder per future predictions
            encoders[col] = le

            logger.info(f"Encoded {col}: {len(unique_values)} categorie uniche")

    # Gestione compound strategy (troppo variegata per label encoding)
    # Usa frequency encoding
    if 'compound_strategy' in df_cat.columns:
        strategy_freq = df_cat['compound_strategy'].value_counts()
        df_cat['compound_strategy_freq'] = df_cat['compound_strategy'].map(strategy_freq)
        df_cat['compound_strategy_freq'] = df_cat['compound_strategy_freq'].fillna(1)
    
    logger.info("Variabili categoriche processate con successo")
    return df_cat, encoders

def normalize_features(df: pd.DataFrame, is_training: bool = True, scaler = None):
    """
    Normalizza features numeriche con RobustScaler.
    """
    logger.info("Normalizzazione features...")
    
    df_norm = df.copy()
    
    # Features da normalizzare (escludiamo target e categoriche)
    exclude_cols = [
        'tire_change_next_lap', 'next_tire_type',  # Target
        'Year', 'DriverID', 'LapNumber', 'Position', 'Stint',  # Identifiers/discrete
        'RaceID', 'DriverRaceID', 'GlobalLapID',  # IDs
        'GranPrix', 'Location', 'Driver', 'Team', 'Compound',  # Categorical raw
        'PitInTime', 'PitOutTime', 'IsFreshTire', 'Rainfall'  # Boolean
    ]
    
    # Colonne categoriche encoded
    encoded_cols = [col for col in df_norm.columns if col.endswith('_encoded')]
    exclude_cols.extend(encoded_cols)
    
    # Features numeriche da normalizzare
    numeric_cols = [col for col in df_norm.columns 
                   if col not in exclude_cols and df_norm[col].dtype in ['float64', 'int64']]
    
    if is_training:
        # Fit scaler su training data
        scaler = RobustScaler()
        df_norm[numeric_cols] = scaler.fit_transform(df_norm[numeric_cols])
        logger.info(f"Scaler fitted su {len(numeric_cols)} features numeriche")
    else:
        # Transform con scaler esistente
        if scaler is None:
            raise ValueError("Scaler necessario per validation/test data")
        df_norm[numeric_cols] = scaler.transform(df_norm[numeric_cols])
        logger.info(f"Features normalizzate con scaler esistente")
    
    return df_norm, scaler

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Gestisce valori mancanti con strategie appropriate."""
    logger.info("Gestione valori mancanti...")
    
    df_clean = df.copy()
    
    # Strategia per tipo di variabile
    strategies = {
        'forward_fill': ['TyreLife', 'Stint', 'Compound'],  # Mantieni ultimo valore valido
        'interpolate': ['AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed'],  # Interpola meteo
        'median': ['LapTime', 'TimeDeltaToDriverAhead', 'TimeDeltaToDriverBehind'],  # Mediana per performance
        'zero': ['tire_degradation_rate', 'laptime_trend_3', 'delta_ahead_trend']  # Zero per rates/trends
    }
    
    for strategy, columns in strategies.items():
        for col in columns:
            if col in df_clean.columns:
                if strategy == 'forward_fill':
                    df_clean[col] = df_clean.groupby('DriverRaceID')[col].ffill()
                elif strategy == 'interpolate':
                    df_clean[col] = df_clean.groupby('RaceID')[col].transform(lambda x: x.interpolate(method='linear'))
                elif strategy == 'median':
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                elif strategy == 'zero':
                    df_clean[col] = df_clean[col].fillna(0)
    
    # Report final missing values
    remaining_missing = df_clean.isnull().sum().sum()
    logger.info(f"Valori mancanti residui: {remaining_missing}")
    
    return df_clean

def temporal_train_val_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split temporale del dataset per evitare data leakage.
    Train: 2018-2021, Validation: 2023, Test: 2024
    """
    logger.info("Split temporale dataset...")
    
    train_years = [2018, 2019, 2020, 2021]
    val_years = [2023]
    test_years = [2024]
    
    train_df = df[df['Year'].isin(train_years)].copy()
    val_df = df[df['Year'].isin(val_years)].copy()
    test_df = df[df['Year'].isin(test_years)].copy()
    
    # Statistiche split
    logger.info("Split completato:")
    logger.info(f"  Train: {len(train_df):,} righe ({train_df['tire_change_next_lap'].mean()*100:.2f}% positive)")
    logger.info(f"  Val:   {len(val_df):,} righe ({val_df['tire_change_next_lap'].mean()*100:.2f}% positive)")
    logger.info(f"  Test:  {len(test_df):,} righe ({test_df['tire_change_next_lap'].mean()*100:.2f}% positive)")
    
    return train_df, val_df, test_df

def create_sequences_for_rnn(df: pd.DataFrame, sequence_length: int = 10, encoders: Dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crea sequenze temporali per RNN.

    Returns:
        X: Array di sequenze (n_sequences, sequence_length, n_features)
        y_change: Target primario cambio gomme (n_sequences,)
        y_type: Target secondario tipo mescola (n_sequences,)
    """
    logger.info(f"Creazione sequenze RNN con lunghezza {sequence_length}...")

    sequences = []
    targets_change = []
    targets_type = []

    # Features numeriche per RNN (escludiamo IDs e target)
    feature_cols = [col for col in df.columns
                   if col not in ['RaceID', 'DriverRaceID', 'GlobalLapID', 'GranPrix',
                                 'Location', 'Driver', 'Team', 'Compound',
                                 'tire_change_next_lap', 'next_tire_type',
                                 'Year', 'DriverID', 'LapNumber', 'compound_strategy']]

    # Raggruppa per driver-race per mantenere continuità temporale
    for driver_race_id, group in df.groupby('DriverRaceID'):
        group_sorted = group.sort_values('LapNumber')

        # Estrai features e targets
        features = group_sorted[feature_cols].values.astype(np.float32)
        change_targets = group_sorted['tire_change_next_lap'].values

        # Encode tire type using the Compound encoder
        compound_encoder = encoders['Compound']
        type_targets = compound_encoder.transform(group_sorted['next_tire_type'].values.astype(str))

        # Crea sequenze sliding window
        for i in range(len(features) - sequence_length + 1):
            seq_features = features[i:i + sequence_length]
            seq_change_target = change_targets[i + sequence_length - 1]  # Target dell'ultimo giro
            seq_type_target = type_targets[i + sequence_length - 1]

            sequences.append(seq_features)
            targets_change.append(seq_change_target)
            targets_type.append(seq_type_target)

    X = np.array(sequences)
    y_change = np.array(targets_change)
    y_type = np.array(targets_type)

    logger.info(f"Sequenze create: {X.shape[0]} sequenze, {X.shape[1]} timesteps, {X.shape[2]} features")
    logger.info(f"Target distribution: {y_change.mean()*100:.2f}% positive changes")

    return X, y_change, y_type, feature_cols

def save_preprocessed_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                          encoders: Dict, scaler, feature_cols: List[str]):
    """Salva dati preprocessati e artifacts."""
    output_dir = Path(__file__).parent / "preprocessed"
    output_dir.mkdir(exist_ok=True)
    
    # Salva datasets
    train_df.to_parquet(output_dir / "train_processed.parquet", index=False)
    val_df.to_parquet(output_dir / "val_processed.parquet", index=False)
    test_df.to_parquet(output_dir / "test_processed.parquet", index=False)
    
    # Salva artifacts per inference
    import joblib
    joblib.dump(encoders, output_dir / "encoders.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")
    joblib.dump(feature_cols, output_dir / "feature_columns.pkl")
    
    logger.info(f"Dati preprocessati salvati in: {output_dir}")

def main():
    """Pipeline completa di preprocessing."""
    logger.info("=== INIZIO PREPROCESSING AVANZATO ===")
    
    # Carica dataset
    df = load_dataset()
    
    # Feature engineering pipeline
    logger.info("Applicazione feature engineering pipeline...")
    
    # 1. Target variables
    df = create_target_variables(df)
    
    # 2. Temporal features
    df = create_temporal_features(df)
    
    # 3. Performance features  
    df = create_performance_features(df)
    
    # 4. Weather features
    df = create_weather_features(df)
    
    # 5. Domain knowledge features
    df = create_domain_knowledge_features(df)
    
    # 6. Handle missing values
    df = handle_missing_values(df)
    
    # 7. Split temporale
    train_df, val_df, test_df = temporal_train_val_test_split(df)
    
    # 8. Categorical encoding
    train_df, encoders = handle_categorical_variables(train_df)
    val_df, _ = handle_categorical_variables(val_df)  # Usa encoders del training
    test_df, _ = handle_categorical_variables(test_df)
    
    # 9. Normalizzazione
    train_df, scaler = normalize_features(train_df, is_training=True)
    val_df, _ = normalize_features(val_df, is_training=False, scaler=scaler)
    test_df, _ = normalize_features(test_df, is_training=False, scaler=scaler)
    
    # 10. Crea sequenze RNN per train
    X_train, y_change_train, y_type_train, feature_cols = create_sequences_for_rnn(train_df, encoders=encoders)
    X_val, y_change_val, y_type_val, _ = create_sequences_for_rnn(val_df, encoders=encoders)
    X_test, y_change_test, y_type_test, _ = create_sequences_for_rnn(test_df, encoders=encoders)

    # Salva tutto
    save_preprocessed_data(train_df, val_df, test_df, encoders, scaler, feature_cols)
    
    # Salva sequenze RNN
    output_dir = Path(__file__).parent / "preprocessed"
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_change_train.npy", y_change_train)
    np.save(output_dir / "y_type_train.npy", y_type_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "y_change_val.npy", y_change_val)
    np.save(output_dir / "y_type_val.npy", y_type_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_change_test.npy", y_change_test)
    np.save(output_dir / "y_type_test.npy", y_type_test)
    
    logger.info("=== PREPROCESSING COMPLETATO ===")
    logger.info(f"Output salvato in: {output_dir}")
    logger.info(f"Sequenze training: {X_train.shape}")
    logger.info(f"Features per timestep: {X_train.shape[2]}")
    logger.info(f"Target ratio: {y_change_train.mean()*100:.2f}%")

if __name__ == "__main__":
    main()
