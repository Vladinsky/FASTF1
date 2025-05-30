"""
Formula 1 Complete Data Unifier
Unisce tutti i dati F1 distribuiti in cartelle separate in un dataset unico
Ottimizzato per Google Colab Pro
"""

import os
import pandas as pd
import numpy as np
import glob
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm.auto import tqdm
import gc
import psutil

class CompleteDataUnifier:
    """Unifica dati F1 da sorgenti multiple in un dataset completo"""
    
    def __init__(self, config_path: str = "configs/data_config_unified.json"):
        """
        Inizializza il data unifier
        
        Args:
            config_path: Path al file di configurazione
        """
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        
        # Paths
        self.drive_path = self.config["colab_specific"]["drive_mount_path"]
        self.temp_dir = self.config["colab_specific"]["temp_directory"]
        self.checkpoint_file = self.config["colab_specific"]["checkpoint_recovery"]["checkpoint_file"]
        self.problematic_files_log_path = self.config.get("logging", {}).get("problematic_files_log", "problematic_parquet_files.txt") # Get path, with a default
        
        # Data tracking
        self.processed_files = set()
        self.unified_data = []
        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "total_rows": 0,
            "duplicates_removed": 0,
            "missing_values_handled": 0
        }
        
        # Ensure directories exist
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict:
        """Carica configurazione"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Configurazione di default"""
        return {
            "data_unification": {
                "source_directories": {
                    "domenico_data": {
                        "path": "/content/drive/MyDrive/domenicoDL",
                        "pattern": "*.csv"
                    },
                    "vincenzo_processed": {
                        "path": "/content/drive/MyDrive/Vincenzo/processed_races", 
                        "pattern": "*.parquet"
                    }
                },
                "target_directory": {
                    "path": "/content/drive/MyDrive/F1_TireChange_Project/data/unified",
                    "filename": "f1_complete_dataset.parquet"
                }
            },
            "colab_specific": {
                "drive_mount_path": "/content/drive",
                "temp_directory": "/content/temp_data",
                "max_memory_usage": 0.8
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def check_memory_usage(self) -> float:
        """Controlla utilizzo memoria"""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent / 100.0
        
        if usage_percent > 0.9:
            self.logger.warning(f"High memory usage: {usage_percent:.1%}")
            gc.collect()
        
        return usage_percent
    
    def discover_data_files(self) -> Dict[str, List[str]]:
        """Scopre tutti i file dati disponibili"""
        discovered_files = {}
        
        for source_name, source_config in self.config["data_unification"]["source_directories"].items():
            source_path = source_config["path"]
            pattern = source_config["pattern"]
            
            if not os.path.exists(source_path):
                self.logger.warning(f"Source directory not found: {source_path}")
                discovered_files[source_name] = []
                continue
            
            # Cerca file con pattern
            file_pattern = os.path.join(source_path, pattern)
            files = glob.glob(file_pattern)
            
            discovered_files[source_name] = sorted(files)
            self.logger.info(f"Found {len(files)} files in {source_name}")
        
        return discovered_files
    
    def load_domenico_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Carica dati CSV da cartella domenicoDL"""
        try:
            # Determina tipo file dal nome
            filename = os.path.basename(file_path)
            
            # Carica CSV
            df = pd.read_csv(file_path)
            
            # Parse informazioni dal filename
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 5:
                year = parts[0]
                gp_name = '_'.join(parts[1:-2])  # Nome GP può avere underscore
                location = parts[-2]
                data_type = parts[-1]
                
                df['Year'] = int(year) if year.isdigit() else 2023
                df['EventName'] = gp_name.replace('_', ' ')
                df['Location'] = location
                df['DataType'] = data_type
            
            # Standardizza nomi colonne
            df = self._standardize_column_names(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def load_data_from_drive(self, file_path: str) -> Optional[pd.DataFrame]:
        """Carica dati Parquet da Google Drive (specificamente da F1_Project/processed_races)"""
        try:
            # Ensure the directory for problematic files log exists
            os.makedirs(os.path.dirname(self.problematic_files_log_path), exist_ok=True)

            df = pd.read_parquet(file_path)

            if df.empty:
                self.logger.warning(f"File is empty: {file_path}")
                with open(self.problematic_files_log_path, 'a') as f_log:
                    f_log.write(f"EMPTY_FILE: {file_path}\n")
                return None
            
            # Parse anno dal filename se non presente
            if 'Year' not in df.columns:
                filename = os.path.basename(file_path)
                year_match = filename[:4]
                if year_match.isdigit():
                    df['Year'] = int(year_match)
            
            # Standardizza nomi colonne
            df = self._standardize_column_names(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            # Ensure the directory for problematic files log exists before writing
            os.makedirs(os.path.dirname(self.problematic_files_log_path), exist_ok=True)
            with open(self.problematic_files_log_path, 'a') as f_log:
                f_log.write(f"LOAD_ERROR: {file_path} - Error: {str(e)}\n")
            return None
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizza nomi colonne tra sorgenti diverse"""
        # Mapping colonne comuni
        column_mapping = {
            'lap_number': 'LapNumber',
            'lapnumber': 'LapNumber', 
            'driver': 'Driver',
            'driver_name': 'Driver',
            'compound': 'Compound',
            'tyre_life': 'TyreLife',
            'tyrelife': 'TyreLife',
            'position': 'Position',
            'lap_time': 'LapTime',
            'laptime': 'LapTime',
            'sector_1_time': 'Sector1Time',
            'sector_2_time': 'Sector2Time', 
            'sector_3_time': 'Sector3Time',
            'track_status': 'TrackStatus',
            'air_temp': 'AirTemp',
            'humidity': 'Humidity',
            'pressure': 'Pressure',
            'rainfall': 'Rainfall',
            'wind_direction': 'WindDirection',
            'wind_speed': 'WindSpeed'
        }
        
        # Applica mapping
        df = df.rename(columns=column_mapping)
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """Valida qualità dati"""
        if df is None or df.empty:
            return False
        
        # Check colonne richieste
        required_cols = self.config["data_unification"]["data_validation"]["required_columns"]
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            self.logger.warning(f"Missing required columns: {missing_cols}")
            # Non bloccare per colonne mancanti, ma avvisare
        
        # Check qualità dati
        quality_checks = self.config["data_unification"]["data_validation"]["data_quality_checks"]
        
        # Check missing values
        missing_pct = (df.isnull().sum().sum() / df.size) * 100
        if missing_pct > quality_checks["max_missing_percentage"]:
            self.logger.warning(f"High missing values: {missing_pct:.1f}%")
        
        # Check numero righe minimo
        if len(df) < quality_checks["min_rows_per_race"]:
            self.logger.warning(f"Too few rows: {len(df)}")
            return False
        
        return True
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocessa DataFrame"""
        original_rows = len(df)
        
        # Remove duplicates
        if self.config["data_unification"]["preprocessing_steps"]["duplicate_removal"]["enabled"]:
            subset_cols = self.config["data_unification"]["preprocessing_steps"]["duplicate_removal"]["subset_columns"]
            available_subset = [col for col in subset_cols if col in df.columns]
            
            if available_subset:
                df = df.drop_duplicates(subset=available_subset, keep='first')
                duplicates_removed = original_rows - len(df)
                self.stats["duplicates_removed"] += duplicates_removed
        
        # Handle missing values
        if self.config["data_unification"]["preprocessing_steps"]["missing_value_handling"]["strategy"] == "contextual":
            # Interpolate numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].interpolate(method='linear', limit=3)
            
            # Fill categorical with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_value)
        
        # Data type optimization
        if self.config["data_unification"]["preprocessing_steps"]["data_type_optimization"]["enabled"]:
            df = self._optimize_data_types(df)
        
        return df
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ottimizza tipi di dati per ridurre memoria"""
        # Downcast integers
        int_cols = df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Downcast floats
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert suitable string columns to category
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            if df[col].nunique() < len(df) * 0.5:  # Se ha meno del 50% valori unici
                df[col] = df[col].astype('category')
        
        return df
    
    def save_checkpoint(self):
        """Salva checkpoint progresso"""
        checkpoint_data = {
            'processed_files': list(self.processed_files),
            'stats': self.stats
        }
        
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        self.logger.info(f"Checkpoint saved: {len(self.processed_files)} files processed")
    
    def load_checkpoint(self) -> bool:
        """Carica checkpoint se esiste"""
        if not os.path.exists(self.checkpoint_file):
            return False
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.processed_files = set(checkpoint_data['processed_files'])
            self.stats = checkpoint_data['stats']
            
            self.logger.info(f"Checkpoint loaded: {len(self.processed_files)} files already processed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return False
    
    def unify_all_data(self) -> pd.DataFrame:
        """Funzione principale per unire tutti i dati"""
        self.logger.info("🚀 Starting complete data unification...")
        
        # Carica checkpoint se esiste
        self.load_checkpoint()
        
        # Scopri tutti i file
        discovered_files = self.discover_data_files()
        
        all_files = []
        for source_name, files in discovered_files.items():
            for file_path in files:
                all_files.append((source_name, file_path))
        
        self.stats["total_files"] = len(all_files)
        self.logger.info(f"Found {len(all_files)} total files to process")
        
        # Processa file in batch per gestire memoria
        batch_size = 50
        unified_chunks = []
        
        for i in tqdm(range(0, len(all_files), batch_size), desc="Processing batches"):
            batch_files = all_files[i:i + batch_size]
            batch_data = []
            
            for source_name, file_path in batch_files:
                # Skip se già processato
                if file_path in self.processed_files:
                    continue
                
                # Carica dati in base alla sorgente
                # La sorgente "domenico_data" non è più prevista nella configurazione per Colab.
                # Se presente per errore, verrà loggato come "Unknown source".
                if source_name == "drive_processed": # Chiave aggiornata per Colab
                    df = self.load_data_from_drive(file_path) # Funzione rinominata
                else:
                    self.logger.warning(f"Unknown source: {source_name}")
                    continue
                
                if df is not None and self.validate_data_quality(df):
                    # Preprocessa
                    df = self.preprocess_dataframe(df)
                    
                    # Aggiungi metadati
                    df['source'] = source_name
                    df['source_file'] = os.path.basename(file_path)
                    
                    batch_data.append(df)
                    self.processed_files.add(file_path)
                    self.stats["processed_files"] += 1
                    self.stats["total_rows"] += len(df)
                else:
                    self.stats["failed_files"] += 1
                
                # Check memoria
                self.check_memory_usage()
            
            # Unisci batch
            if batch_data:
                batch_df = pd.concat(batch_data, ignore_index=True)
                unified_chunks.append(batch_df)
                self.logger.info(f"Batch {i//batch_size + 1} processed: {len(batch_df)} rows")
            
            # Salva checkpoint periodicamente
            if i % (batch_size * 5) == 0:
                self.save_checkpoint()
            
            # Cleanup memoria
            del batch_data
            gc.collect()
        
        # Unisci tutti i chunk
        self.logger.info("Combining all batches...")
        final_df = pd.concat(unified_chunks, ignore_index=True)
        
        # Post-processing finale
        final_df = self._final_postprocessing(final_df)
        
        # Salva risultato finale
        self._save_unified_dataset(final_df)
        
        # Salva checkpoint finale
        self.save_checkpoint()
        
        self.logger.info("✅ Data unification completed!")
        self._print_final_stats(final_df)
        
        return final_df
    
    def _final_postprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-processing finale del dataset unificato"""
        self.logger.info("Applying final post-processing...")
        
        # Sort cronologico
        if 'Year' in df.columns and 'LapNumber' in df.columns:
            df = df.sort_values(['Year', 'EventName', 'Driver', 'LapNumber'])
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Final optimization
        df = self._optimize_data_types(df)
        
        return df
    
    def _save_unified_dataset(self, df: pd.DataFrame):
        """Salva dataset unificato"""
        target_config = self.config["data_unification"]["target_directory"]
        target_path = target_config["path"]
        filename = target_config["filename"]
        
        os.makedirs(target_path, exist_ok=True)
        full_path = os.path.join(target_path, filename)
        
        # Salva in formato Parquet per efficienza
        df.to_parquet(full_path, compression='snappy', index=False)
        
        self.logger.info(f"Dataset saved to: {full_path}")
        self.logger.info(f"File size: {os.path.getsize(full_path) / 1024 / 1024:.1f} MB")
    
    def _print_final_stats(self, df: pd.DataFrame):
        """Stampa statistiche finali"""
        stats_msg = f"""
        📊 UNIFICATION COMPLETED
        ========================
        Total files found: {self.stats['total_files']}
        Files processed: {self.stats['processed_files']}
        Files failed: {self.stats['failed_files']}
        
        Final dataset:
        - Rows: {len(df):,}
        - Columns: {len(df.columns)}
        - Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB
        
        Years covered: {sorted(df['Year'].unique()) if 'Year' in df.columns else 'Unknown'}
        Drivers: {df['Driver'].nunique() if 'Driver' in df.columns else 'Unknown'}
        Events: {df['EventName'].nunique() if 'EventName' in df.columns else 'Unknown'}
        """
        
        self.logger.info(stats_msg)

def main():
    """Funzione principale per uso standalone"""
    unifier = CompleteDataUnifier()
    dataset = unifier.unify_all_data()
    return dataset

if __name__ == "__main__":
    main()
