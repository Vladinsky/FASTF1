"""
Data Consolidation Script - Fase 1
==================================

Questo script consolida tutti i file parquet individuali delle gare F1 
in un unico dataset per il training della rete neurale RNN.

Funzionalità:
- Lettura di tutti i file .parquet da processed_races/
- Validazione e pulizia dati
- Aggiunta metadati di consolidamento
- Salvataggio dataset unificato

Autore: Data Science Team
Data: 2025-05-29
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import warnings

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_consolidation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_paths() -> Tuple[Path, Path]:
    """
    Configura i percorsi per input e output dei dati.
    
    Returns:
        Tuple[Path, Path]: (input_dir, output_dir)
    """
    base_dir = Path(__file__).parent.parent  # Vincenzo/
    input_dir = base_dir / "processed_races"
    output_dir = Path(__file__).parent  # Vincenzo/dataset/
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory input non trovata: {input_dir}")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creata directory output: {output_dir}")
    
    return input_dir, output_dir

def get_parquet_files(input_dir: Path) -> List[Path]:
    """
    Trova tutti i file .parquet nella directory di input.
    
    Args:
        input_dir (Path): Directory contenente i file parquet
        
    Returns:
        List[Path]: Lista dei file parquet ordinati per nome
    """
    parquet_files = list(input_dir.glob("*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"Nessun file .parquet trovato in {input_dir}")
    
    # Ordina per nome file (cronologico)
    parquet_files.sort()
    
    logger.info(f"Trovati {len(parquet_files)} file parquet:")
    for file in parquet_files[:5]:  # Mostra primi 5
        logger.info(f"  - {file.name}")
    if len(parquet_files) > 5:
        logger.info(f"  ... e altri {len(parquet_files) - 5} file")
    
    return parquet_files

def validate_dataframe(df: pd.DataFrame, filename: str) -> Dict[str, any]:
    """
    Valida un DataFrame e raccoglie statistiche.
    
    Args:
        df (pd.DataFrame): DataFrame da validare
        filename (str): Nome del file per logging
        
    Returns:
        Dict[str, any]: Statistiche di validazione
    """
    stats = {
        'filename': filename,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'n_drivers': df['DriverID'].nunique() if 'DriverID' in df.columns else 0,
        'n_laps': df['LapNumber'].nunique() if 'LapNumber' in df.columns else 0,
        'missing_values': df.isnull().sum().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'years': df['Year'].unique().tolist() if 'Year' in df.columns else [],
        'gran_prix': df['GranPrix'].unique().tolist() if 'GranPrix' in df.columns else []
    }
    
    # Validazioni specifiche
    issues = []
    
    # Check colonne essenziali
    essential_columns = ['Year', 'DriverID', 'LapNumber', 'Position', 'LapTime']
    missing_cols = [col for col in essential_columns if col not in df.columns]
    if missing_cols:
        issues.append(f"Colonne mancanti: {missing_cols}")
    
    # Check valori negativi anomali
    if 'LapNumber' in df.columns and (df['LapNumber'] <= 0).any():
        issues.append("Valori LapNumber <= 0 trovati")
    
    if 'Position' in df.columns and (df['Position'] <= 0).any():
        issues.append("Valori Position <= 0 trovati")
    
    # Check duplicate entries
    if 'Year' in df.columns and 'DriverID' in df.columns and 'LapNumber' in df.columns:
        duplicates = df.groupby(['Year', 'DriverID', 'LapNumber']).size()
        if (duplicates > 1).any():
            n_duplicates = (duplicates > 1).sum()
            issues.append(f"{n_duplicates} gruppi di righe duplicate trovati")
    
    stats['issues'] = issues
    
    if issues:
        logger.warning(f"Issues in {filename}: {'; '.join(issues)}")
    
    return stats

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pulisce un DataFrame applicando correzioni standard.
    
    Args:
        df (pd.DataFrame): DataFrame da pulire
        
    Returns:
        pd.DataFrame: DataFrame pulito
    """
    df_clean = df.copy()
    original_rows = len(df_clean)
    
    # Rimuovi righe con LapNumber o DriverID nulli (essenziali)
    essential_cols = ['LapNumber', 'DriverID']
    existing_essential = [col for col in essential_cols if col in df_clean.columns]
    df_clean = df_clean.dropna(subset=existing_essential)
    
    # Converti tipi appropriati
    if 'LapNumber' in df_clean.columns:
        df_clean['LapNumber'] = pd.to_numeric(df_clean['LapNumber'], errors='coerce')
        df_clean = df_clean.dropna(subset=['LapNumber'])
        df_clean['LapNumber'] = df_clean['LapNumber'].astype(int)
    
    if 'DriverID' in df_clean.columns:
        df_clean['DriverID'] = pd.to_numeric(df_clean['DriverID'], errors='coerce')
        df_clean = df_clean.dropna(subset=['DriverID'])
        df_clean['DriverID'] = df_clean['DriverID'].astype(int)
    
    if 'Position' in df_clean.columns:
        df_clean['Position'] = pd.to_numeric(df_clean['Position'], errors='coerce')
    
    if 'Year' in df_clean.columns:
        df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
        df_clean = df_clean.dropna(subset=['Year'])
        df_clean['Year'] = df_clean['Year'].astype(int)
    
    # Rimuovi valori anomali evidenti
    if 'LapNumber' in df_clean.columns:
        df_clean = df_clean[df_clean['LapNumber'] > 0]
        df_clean = df_clean[df_clean['LapNumber'] <= 100]  # Max ragionevole per F1
    
    if 'Position' in df_clean.columns:
        df_clean = df_clean[df_clean['Position'] > 0]
        df_clean = df_clean[df_clean['Position'] <= 25]  # Max ragionevole per F1
    
    # Rimuovi duplicati esatti
    df_clean = df_clean.drop_duplicates()
    
    rows_removed = original_rows - len(df_clean)
    if rows_removed > 0:
        logger.info(f"Rimosse {rows_removed} righe durante pulizia ({rows_removed/original_rows*100:.1f}%)")
    
    return df_clean

def add_consolidation_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge metadati utili per il consolidamento.
    
    Args:
        df (pd.DataFrame): DataFrame originale
        
    Returns:
        pd.DataFrame: DataFrame con metadati aggiuntivi
    """
    df_meta = df.copy()
    
    # Race identifier unico
    if all(col in df_meta.columns for col in ['Year', 'GranPrix']):
        df_meta['RaceID'] = df_meta['Year'].astype(str) + "_" + \
                           df_meta['GranPrix'].str.replace(' ', '_').str.upper()
    
    # Driver-Race identifier unico
    if all(col in df_meta.columns for col in ['Year', 'DriverID', 'GranPrix']):
        df_meta['DriverRaceID'] = df_meta['Year'].astype(str) + "_" + \
                                 df_meta['DriverID'].astype(str) + "_" + \
                                 df_meta['GranPrix'].str.replace(' ', '_').str.upper()
    
    # Lap identifier unico globale
    if all(col in df_meta.columns for col in ['Year', 'DriverID', 'GranPrix', 'LapNumber']):
        df_meta['GlobalLapID'] = df_meta['Year'].astype(str) + "_" + \
                                df_meta['DriverID'].astype(str) + "_" + \
                                df_meta['GranPrix'].str.replace(' ', '_').str.upper() + "_" + \
                                df_meta['LapNumber'].astype(str)
    
    logger.info(f"Aggiunti metadati di consolidamento: {df_meta.shape[1] - df.shape[1]} nuove colonne")
    
    return df_meta

def consolidate_data(input_dir: Path, output_dir: Path) -> Dict[str, any]:
    """
    Funzione principale per consolidare tutti i file parquet.
    
    Args:
        input_dir (Path): Directory con file parquet di input
        output_dir (Path): Directory per output consolidato
        
    Returns:
        Dict[str, any]: Statistiche del consolidamento
    """
    logger.info("=== Inizio Consolidamento Dataset F1 ===")
    
    # Trova file parquet
    parquet_files = get_parquet_files(input_dir)
    
    # Strutture per raccogliere dati e statistiche
    dataframes = []
    file_stats = []
    errors = []
    
    # Processa ogni file
    for i, file_path in enumerate(parquet_files):
        logger.info(f"Processando {i+1}/{len(parquet_files)}: {file_path.name}")
        
        try:
            # Leggi file
            df = pd.read_parquet(file_path)
            
            # Valida
            stats = validate_dataframe(df, file_path.name)
            file_stats.append(stats)
            
            # Pulisci
            df_clean = clean_dataframe(df)
            
            # Aggiungi metadati
            df_final = add_consolidation_metadata(df_clean)
            
            dataframes.append(df_final)
            
            logger.info(f"  ✓ Successo: {len(df_final)} righe, {len(df_final.columns)} colonne")
            
        except Exception as e:
            error_msg = f"Errore processando {file_path.name}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            continue
    
    if not dataframes:
        raise RuntimeError("Nessun file processato con successo!")
    
    # Consolida tutti i DataFrame
    logger.info("Consolidando tutti i DataFrame...")
    
    try:
        consolidated_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Dataset consolidato: {len(consolidated_df)} righe, {len(consolidated_df.columns)} colonne")
    except Exception as e:
        logger.error(f"Errore durante concatenazione: {e}")
        raise
    
    # Validazione finale
    logger.info("Validazione finale del dataset consolidato...")
    final_stats = validate_dataframe(consolidated_df, "consolidated_dataset")
    
    # Salva dataset consolidato
    output_file = output_dir / "dataset.parquet"
    logger.info(f"Salvando dataset consolidato in: {output_file}")
    
    try:
        consolidated_df.to_parquet(output_file, index=False, compression='snappy')
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        logger.info(f"  ✓ Dataset salvato: {file_size_mb:.1f} MB")
    except Exception as e:
        logger.error(f"Errore durante salvataggio: {e}")
        raise
    
    # Prepara statistiche di consolidamento
    consolidation_stats = {
        'input_files_processed': len(dataframes),
        'input_files_errors': len(errors),
        'total_rows': len(consolidated_df),
        'total_columns': len(consolidated_df.columns),
        'unique_years': sorted(consolidated_df['Year'].unique().tolist()) if 'Year' in consolidated_df.columns else [],
        'unique_drivers': consolidated_df['DriverID'].nunique() if 'DriverID' in consolidated_df.columns else 0,
        'unique_races': consolidated_df['RaceID'].nunique() if 'RaceID' in consolidated_df.columns else 0,
        'total_laps': len(consolidated_df),
        'memory_usage_mb': final_stats['memory_usage_mb'],
        'output_file_size_mb': file_size_mb,
        'errors': errors,
        'file_statistics': file_stats
    }
    
    return consolidation_stats

def generate_summary_report(stats: Dict[str, any], output_dir: Path) -> None:
    """
    Genera un report di riepilogo del consolidamento.
    
    Args:
        stats (Dict[str, any]): Statistiche del consolidamento
        output_dir (Path): Directory per salvare il report
    """
    report_file = output_dir / "consolidation_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("REPORT CONSOLIDAMENTO DATASET F1\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("RIEPILOGO GENERALE\n")
        f.write("-" * 20 + "\n")
        f.write(f"File processati con successo: {stats['input_files_processed']}\n")
        f.write(f"File con errori: {stats['input_files_errors']}\n")
        f.write(f"Righe totali nel dataset: {stats['total_rows']:,}\n")
        f.write(f"Colonne totali: {stats['total_columns']}\n")
        f.write(f"Anni coperti: {', '.join(map(str, stats['unique_years']))}\n")
        f.write(f"Piloti unici: {stats['unique_drivers']}\n")
        f.write(f"Gare uniche: {stats['unique_races']}\n")
        f.write(f"Utilizzo memoria: {stats['memory_usage_mb']:.1f} MB\n")
        f.write(f"Dimensione file output: {stats['output_file_size_mb']:.1f} MB\n\n")
        
        if stats['errors']:
            f.write("ERRORI RISCONTRATI\n")
            f.write("-" * 20 + "\n")
            for error in stats['errors']:
                f.write(f"- {error}\n")
            f.write("\n")
        
        f.write("STATISTICHE PER FILE\n")
        f.write("-" * 20 + "\n")
        for file_stat in stats['file_statistics']:
            f.write(f"\nFile: {file_stat['filename']}\n")
            f.write(f"  Righe: {file_stat['n_rows']:,}\n")
            f.write(f"  Piloti: {file_stat['n_drivers']}\n")
            f.write(f"  Giri: {file_stat['n_laps']}\n")
            f.write(f"  Valori mancanti: {file_stat['missing_values']:,}\n")
            f.write(f"  Anni: {', '.join(map(str, file_stat['years']))}\n")
            if file_stat['issues']:
                f.write(f"  Issues: {'; '.join(file_stat['issues'])}\n")
    
    logger.info(f"Report di consolidamento salvato in: {report_file}")

def main():
    """Funzione principale dello script."""
    try:
        # Setup
        input_dir, output_dir = setup_paths()
        
        # Consolida dati
        stats = consolidate_data(input_dir, output_dir)
        
        # Genera report
        generate_summary_report(stats, output_dir)
        
        # Log finale
        logger.info("=" * 60)
        logger.info("CONSOLIDAMENTO COMPLETATO CON SUCCESSO!")
        logger.info(f"Dataset finale: {stats['total_rows']:,} righe")
        logger.info(f"File output: {output_dir / 'dataset.parquet'}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"ERRORE CRITICO: {e}")
        raise

if __name__ == "__main__":
    # Sopprimi warning pandas per performance
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    
    main()
