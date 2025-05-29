"""
Dataset Explorer - Analisi Rapida del Dataset Consolidato
=========================================================

Script per analisi esplorativa del dataset F1 consolidato
per validare la qualità dei dati e preparare il feature engineering.

Autore: Data Science Team
Data: 2025-05-29
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def load_dataset():
    """Carica il dataset consolidato."""
    dataset_path = Path(__file__).parent / "dataset.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset non trovato: {dataset_path}")
    
    df = pd.read_parquet(dataset_path)
    print(f"✅ Dataset caricato: {len(df):,} righe, {len(df.columns)} colonne")
    return df

def basic_info(df):
    """Analisi di base del dataset."""
    print("\n" + "="*60)
    print("INFORMAZIONI GENERALI")
    print("="*60)
    
    print(f"📊 Forma dataset: {df.shape}")
    print(f"📅 Anni disponibili: {sorted(df['Year'].unique())}")
    print(f"🏁 Gare totali: {df['RaceID'].nunique()}")
    print(f"👨‍💼 Piloti unici: {df['DriverID'].nunique()}")
    print(f"⏱️ Giri totali: {len(df)}")
    
    # Statistiche per anno
    print("\n📈 DISTRIBUZIONE PER ANNO:")
    year_stats = df.groupby('Year').agg({
        'RaceID': 'nunique',
        'DriverID': 'nunique', 
        'LapNumber': 'count'
    }).rename(columns={
        'RaceID': 'Gare',
        'DriverID': 'Piloti',
        'LapNumber': 'Giri'
    })
    print(year_stats)

def data_quality_check(df):
    """Controllo qualità dei dati."""
    print("\n" + "="*60)
    print("CONTROLLO QUALITÀ DATI")
    print("="*60)
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Colonna': missing.index,
        'Missing': missing.values,
        'Percentuale': missing_pct.values
    }).query('Missing > 0').sort_values('Missing', ascending=False)
    
    if not missing_df.empty:
        print("🔍 VALORI MANCANTI:")
        for _, row in missing_df.head(10).iterrows():
            print(f"  {row['Colonna']}: {row['Missing']:,} ({row['Percentuale']:.1f}%)")
    else:
        print("✅ Nessun valore mancante trovato!")
    
    # Duplicati
    duplicates = df.duplicated().sum()
    print(f"\n🔄 Righe duplicate: {duplicates:,} ({duplicates/len(df)*100:.1f}%)")
    
    # Range valori chiave
    print("\n📊 RANGE VALORI CHIAVE:")
    numeric_cols = ['LapNumber', 'Position', 'LapTime', 'TyreLife']
    for col in numeric_cols:
        if col in df.columns:
            print(f"  {col}: {df[col].min():.2f} - {df[col].max():.2f}")

def tire_analysis(df):
    """Analisi pneumatici per target RNN."""
    print("\n" + "="*60)
    print("ANALISI PNEUMATICI (TARGET RNN)")
    print("="*60)
    
    # Distribuzione compound
    compound_dist = df['Compound'].value_counts()
    print("🏎️ DISTRIBUZIONE MESCOLE:")
    for compound, count in compound_dist.items():
        pct = count/len(df)*100
        print(f"  {compound}: {count:,} ({pct:.1f}%)")
    
    # Analisi stint
    stint_stats = df.groupby('Stint')['TyreLife'].agg(['count', 'mean', 'max'])
    print(f"\n🔧 STATISTICHE STINT:")
    print(f"  Stint massimo: {df['Stint'].max()}")
    print(f"  Durata media stint: {df['TyreLife'].mean():.1f} giri")
    print(f"  Stint più lungo: {df['TyreLife'].max()} giri")
    
    # Pit stops (cambio stint)
    pit_stops = df[df['PitInTime'] == True].shape[0]
    print(f"  Pit stop registrati: {pit_stops:,}")

def target_variable_analysis(df):
    """Analisi per creare variabile target cambio gomme."""
    print("\n" + "="*60)
    print("ANALISI TARGET VARIABLE")
    print("="*60)
    
    # Creiamo una versione semplificata del target
    # Cambio gomme = cambio di stint nel giro successivo
    df_sorted = df.sort_values(['DriverRaceID', 'LapNumber'])
    df_sorted['NextStint'] = df_sorted.groupby('DriverRaceID')['Stint'].shift(-1)
    tire_changes = (df_sorted['Stint'] != df_sorted['NextStint']) & df_sorted['NextStint'].notna()
    
    n_changes = tire_changes.sum()
    total_laps = len(df_sorted)
    
    print(f"🎯 CAMBIO GOMME (TARGET):")
    print(f"  Cambi gomme totali: {n_changes:,}")
    print(f"  Giri senza cambio: {total_laps - n_changes:,}")
    print(f"  Percentuale cambio: {n_changes/total_laps*100:.2f}%")
    print(f"  Rapporto sbilanciamento: 1:{(total_laps-n_changes)/n_changes:.0f}")
    
    return tire_changes

def weather_and_performance_analysis(df):
    """Analisi variabili meteo e performance."""
    print("\n" + "="*60)
    print("ANALISI METEO E PERFORMANCE")
    print("="*60)
    
    # Variabili meteo
    weather_cols = ['AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed', 'Rainfall']
    available_weather = [col for col in weather_cols if col in df.columns]
    
    print("🌤️ DATI METEO DISPONIBILI:")
    for col in available_weather:
        non_null = df[col].count()
        print(f"  {col}: {non_null:,} valori ({non_null/len(df)*100:.1f}%)")
    
    # Performance variables
    perf_cols = ['LapTime', 'Position', 'TimeDeltaToDriverAhead']
    print("\n⚡ VARIABILI PERFORMANCE:")
    for col in perf_cols:
        if col in df.columns:
            non_null = df[col].count()
            mean_val = df[col].mean()
            print(f"  {col}: {non_null:,} valori, media: {mean_val:.2f}")

def create_summary_visualizations(df):
    """Crea visualizzazioni di riepilogo."""
    print("\n" + "="*60)
    print("CREAZIONE VISUALIZZAZIONI")
    print("="*60)
    
    # Setup plot style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Dataset F1 - Analisi Esplorativa', fontsize=16, fontweight='bold')
    
    # 1. Distribuzione gare per anno
    year_counts = df['Year'].value_counts().sort_index()
    axes[0, 0].bar(year_counts.index, year_counts.values, color='steelblue', alpha=0.7)
    axes[0, 0].set_title('Distribuzione Gare per Anno')
    axes[0, 0].set_xlabel('Anno')
    axes[0, 0].set_ylabel('Numero Giri')
    
    # 2. Distribuzione mescole
    compound_counts = df['Compound'].value_counts()
    axes[0, 1].pie(compound_counts.values, labels=compound_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Distribuzione Mescole')
    
    # 3. Distribuzione TyreLife
    axes[1, 0].hist(df['TyreLife'].dropna(), bins=50, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Distribuzione Durata Pneumatici')
    axes[1, 0].set_xlabel('TyreLife (giri)')
    axes[1, 0].set_ylabel('Frequenza')
    
    # 4. LapTime distribution (sample per performance)
    lap_times = df['LapTime'].dropna()
    if len(lap_times) > 0:
        # Filtra outliers estremi per visualizzazione
        q1, q99 = lap_times.quantile([0.01, 0.99])
        filtered_times = lap_times[(lap_times >= q1) & (lap_times <= q99)]
        axes[1, 1].hist(filtered_times, bins=50, color='green', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Distribuzione Tempi sul Giro')
        axes[1, 1].set_xlabel('LapTime (secondi)')
        axes[1, 1].set_ylabel('Frequenza')
    
    plt.tight_layout()
    
    # Salva plot
    plot_path = Path(__file__).parent / "dataset_exploration.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualizzazioni salvate: {plot_path}")
    plt.show()

def main():
    """Funzione principale."""
    print("🔍 ANALISI ESPLORATIVA DATASET F1")
    print("=" * 60)
    
    # Carica dataset
    df = load_dataset()
    
    # Analisi di base
    basic_info(df)
    
    # Controllo qualità
    data_quality_check(df)
    
    # Analisi pneumatici
    tire_analysis(df)
    
    # Analisi target variable
    tire_changes = target_variable_analysis(df)
    
    # Analisi meteo e performance
    weather_and_performance_analysis(df)
    
    # Visualizzazioni
    create_summary_visualizations(df)
    
    print("\n" + "="*60)
    print("✅ ANALISI COMPLETATA!")
    print("📋 Prossimi passi:")
    print("   1. Feature engineering avanzato")
    print("   2. Creazione target variables")
    print("   3. Preparazione sequenze per RNN")
    print("   4. Implementazione modello")
    print("="*60)

if __name__ == "__main__":
    main()
