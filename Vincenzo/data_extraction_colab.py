"""
Formula 1 Data Extraction Script - Google Colab Optimized
===========================================================

This script is specifically designed for Google Colab environments with:
- Google Drive integration for persistent storage
- Runtime disconnection resilience
- Memory-conscious processing
- Granular error tracking and recovery
- Auto-resume capabilities

Author: AI Assistant for F1 Data Science Project
"""

import argparse
import json
import logging
import os
import pandas as pd
import numpy as np
import fastf1 as ff1
import gc
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import traceback

# Colab-specific imports
try:
    from google.colab import drive, widgets
    from IPython.display import display, clear_output, HTML
    import ipywidgets as widgets
    COLAB_ENV = True
except ImportError:
    COLAB_ENV = False
    print("Not running in Colab environment")

# Progress bar
from tqdm.auto import tqdm

# Retry decorator
import functools
import random

# FastF1 error handling
try:
    from fastf1.core import SessionNotAvailableError, DataNotLoadedError
    from fastf1.ergast import ErgastError
    FastF1Error = Exception  # Fallback
except ImportError:
    # Define dummy exceptions if not available
    class SessionNotAvailableError(Exception): pass
    class DataNotLoadedError(Exception): pass
    class ErgastError(Exception): pass
    FastF1Error = Exception

# =============================================================================
# COLAB-SPECIFIC UTILITIES
# =============================================================================

class ColabEnvironment:
    """Manages Colab-specific environment setup and utilities."""
    
    def __init__(self, drive_path: str = "/content/drive"):
        self.drive_path = drive_path
        self.is_drive_mounted = False
        
    def setup_environment(self) -> bool:
        """Sets up the Colab environment including Drive mounting."""
        if not COLAB_ENV:
            print("⚠️  Not running in Colab. Drive mounting skipped.")
            return True
            
        try:
            # Mount Google Drive
            if not os.path.exists(self.drive_path):
                print("🔗 Mounting Google Drive...")
                drive.mount(self.drive_path, force_remount=True)
            
            self.is_drive_mounted = os.path.exists(f"{self.drive_path}/MyDrive")
            
            if self.is_drive_mounted:
                print("✅ Google Drive mounted successfully")
                return True
            else:
                print("❌ Google Drive mounting failed")
                return False
                
        except Exception as e:
            print(f"❌ Error setting up Colab environment: {e}")
            return False
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Returns current memory usage statistics."""
        memory_info = psutil.virtual_memory()
        return {
            'total_gb': memory_info.total / (1024**3),
            'available_gb': memory_info.available / (1024**3),
            'used_gb': memory_info.used / (1024**3),
            'percent': memory_info.percent
        }
    
    def check_memory_warning(self, threshold_percent: float = 85.0) -> bool:
        """Checks if memory usage is above threshold and issues warning."""
        memory = self.get_memory_usage()
        if memory['percent'] > threshold_percent:
            print(f"⚠️  Memory usage high: {memory['percent']:.1f}% ({memory['used_gb']:.2f}GB/{memory['total_gb']:.2f}GB)")
            return True
        return False
    
    def force_cleanup(self):
        """Forces garbage collection and memory cleanup."""
        gc.collect()
        if COLAB_ENV:
            # Additional Colab-specific cleanup if needed
            pass

# =============================================================================
# RETRY MECHANISMS
# =============================================================================

def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 2.0, 
                      max_delay: float = 60.0, exceptions: tuple = (Exception,)):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for delay between retries
        max_delay: Maximum delay between retries in seconds
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                        
                    delay = min(backoff_factor ** attempt + random.uniform(0, 1), max_delay)
                    logging.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    
            raise last_exception
        return wrapper
    return decorator

# =============================================================================
# ERROR TRACKING SYSTEM
# =============================================================================

class ErrorTracker:
    """Tracks and manages errors during data extraction process."""
    
    def __init__(self, error_log_path: str):
        self.error_log_path = error_log_path
        self.errors = {
            'failed_races': [],
            'failed_drivers': [],
            'failed_laps': [],
            'summary': {
                'total_errors': 0,
                'races_affected': 0,
                'drivers_affected': 0,
                'last_updated': None
            }
        }
        self.load_existing_errors()
    
    def load_existing_errors(self):
        """Loads existing error log if it exists."""
        if os.path.exists(self.error_log_path):
            try:
                with open(self.error_log_path, 'r') as f:
                    self.errors = json.load(f)
                logging.info(f"Loaded existing error log with {self.errors['summary']['total_errors']} errors")
            except Exception as e:
                logging.warning(f"Could not load existing error log: {e}")
    
    def log_race_error(self, year: int, round_num: int, race_name: str, 
                      error_type: str, error_msg: str, traceback_str: str = None):
        """Logs a race-level error."""
        error_entry = {
            'year': year,
            'round': round_num,
            'race_name': race_name,
            'error_type': error_type,
            'error_message': str(error_msg),
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback_str
        }
        
        self.errors['failed_races'].append(error_entry)
        self.errors['summary']['total_errors'] += 1
        self.errors['summary']['races_affected'] += 1
        self._save_errors()
        
        logging.error(f"Race error logged: {year} {race_name} - {error_type}: {error_msg}")
    
    def log_driver_error(self, year: int, race_name: str, driver_id: str, 
                        error_type: str, error_msg: str):
        """Logs a driver-level error."""
        error_entry = {
            'year': year,
            'race_name': race_name,
            'driver_id': driver_id,
            'error_type': error_type,
            'error_message': str(error_msg),
            'timestamp': datetime.now().isoformat()
        }
        
        self.errors['failed_drivers'].append(error_entry)
        self.errors['summary']['total_errors'] += 1
        self.errors['summary']['drivers_affected'] += 1
        self._save_errors()
        
        logging.warning(f"Driver error logged: {year} {race_name} Driver {driver_id} - {error_type}: {error_msg}")
    
    def log_lap_error(self, year: int, race_name: str, driver_id: str, 
                     lap_number: int, error_type: str, error_msg: str):
        """Logs a lap-level error."""
        error_entry = {
            'year': year,
            'race_name': race_name,
            'driver_id': driver_id,
            'lap_number': lap_number,
            'error_type': error_type,
            'error_message': str(error_msg),
            'timestamp': datetime.now().isoformat()
        }
        
        self.errors['failed_laps'].append(error_entry)
        self.errors['summary']['total_errors'] += 1
        self._save_errors()
        
        logging.debug(f"Lap error logged: {year} {race_name} Driver {driver_id} Lap {lap_number} - {error_type}: {error_msg}")
    
    def _save_errors(self):
        """Saves error log to file."""
        self.errors['summary']['last_updated'] = datetime.now().isoformat()
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.error_log_path), exist_ok=True)
            
            with open(self.error_log_path, 'w') as f:
                json.dump(self.errors, f, indent=2)
        except Exception as e:
            logging.error(f"Could not save error log: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Returns error summary statistics."""
        return {
            'total_errors': self.errors['summary']['total_errors'],
            'failed_races': len(self.errors['failed_races']),
            'failed_drivers': len(self.errors['failed_drivers']),
            'failed_laps': len(self.errors['failed_laps']),
            'races_affected': len(set(f"{e['year']}_{e['race_name']}" for e in self.errors['failed_races'])),
            'last_updated': self.errors['summary']['last_updated']
        }

# =============================================================================
# PROGRESS TRACKING SYSTEM
# =============================================================================

class ProgressTracker:
    """Tracks processing progress for resume capability."""
    
    def __init__(self, progress_file_path: str):
        self.progress_file_path = progress_file_path
        self.progress = {
            'completed_races': [],
            'current_year': None,
            'current_round': None,
            'last_updated': None,
            'total_races_planned': 0,
            'total_races_completed': 0
        }
        self.load_existing_progress()
    
    def load_existing_progress(self):
        """Loads existing progress if available."""
        if os.path.exists(self.progress_file_path):
            try:
                with open(self.progress_file_path, 'r') as f:
                    self.progress = json.load(f)
                logging.info(f"Loaded existing progress: {self.progress['total_races_completed']} races completed")
            except Exception as e:
                logging.warning(f"Could not load existing progress: {e}")
    
    def mark_race_completed(self, year: int, round_num: int, race_name: str):
        """Marks a race as completed."""
        race_id = f"{year}_{round_num:02d}_{race_name}"
        if race_id not in self.progress['completed_races']:
            self.progress['completed_races'].append(race_id)
            self.progress['total_races_completed'] += 1
        
        self.progress['current_year'] = year
        self.progress['current_round'] = round_num
        self.progress['last_updated'] = datetime.now().isoformat()
        self._save_progress()
    
    def is_race_completed(self, year: int, round_num: int, race_name: str) -> bool:
        """Checks if a race has already been completed."""
        race_id = f"{year}_{round_num:02d}_{race_name}"
        return race_id in self.progress['completed_races']
    
    def _save_progress(self):
        """Saves progress to file."""
        try:
            os.makedirs(os.path.dirname(self.progress_file_path), exist_ok=True)
            with open(self.progress_file_path, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            logging.error(f"Could not save progress: {e}")
    
    def get_completion_percentage(self) -> float:
        """Returns completion percentage."""
        if self.progress['total_races_planned'] == 0:
            return 0.0
        return (self.progress['total_races_completed'] / self.progress['total_races_planned']) * 100

# =============================================================================
# CONFIGURATION AND LOGGING
# =============================================================================

def parse_arguments():
    """Parses command-line arguments for Colab environment."""
    parser = argparse.ArgumentParser(description="Extract Formula 1 data - Colab Optimized")
    parser.add_argument(
        "--config",
        type=str,
        default="Vincenzo/config_colab.json",
        help="Path to the JSON configuration file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (default behavior in Colab)"
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Force restart ignoring previous progress"
    )
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

def setup_colab_logging(log_file: str, log_level: str = "INFO"):
    """Sets up logging optimized for Colab environment."""
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # File handler (persistent on Drive)
    file_handler = logging.FileHandler(log_file, mode='a')  # Append mode for resume
    file_handler.setFormatter(formatter)
    
    # Console handler for Colab output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=[file_handler, console_handler],
        force=True
    )
    
    logging.info(f"Colab logging setup complete. Log file: {log_file}")

# =============================================================================
# FASTF1 SESSION MANAGEMENT
# =============================================================================

@retry_with_backoff(max_retries=3, backoff_factor=2.0, 
                   exceptions=(SessionNotAvailableError, ErgastError, 
                              ConnectionError, TimeoutError))
def load_session_with_retry(session, **load_params):
    """Loads FastF1 session with retry mechanism."""
    logging.info(f"Loading session data with params: {load_params}")
    session.load(**load_params)
    logging.info("Session data loaded successfully")
    return session

def sanitize_filename(name: str) -> str:
    """Sanitizes string for use in filenames."""
    # Remove or replace problematic characters
    sanitized = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
    sanitized = sanitized.replace(' ', '_')
    return sanitized if sanitized else "unknown"

# =============================================================================
# DATA EXTRACTION FUNCTIONS
# =============================================================================

def get_circuit_info(circuits_file_path: str) -> Dict[str, float]:
    """Loads circuit length information."""
    try:
        with open(circuits_file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading circuit info: {e}")
        return {}

def extract_lap_data_with_error_handling(session, year: int, race_name: str, 
                                       location_name: str, circuits_info: Dict[str, float],
                                       error_tracker: ErrorTracker) -> pd.DataFrame:
    """
    Extracts lap data with comprehensive error handling at driver and lap level.
    """
    all_laps_data = []
    
    # Get circuit length
    circuit_length = circuits_info.get(location_name)
    if circuit_length is None:
        # Try fuzzy matching
        for circuit_key, length in circuits_info.items():
            if location_name.lower() in circuit_key.lower() or circuit_key.lower() in location_name.lower():
                circuit_length = length
                logging.warning(f"Using fuzzy match for circuit '{location_name}' -> '{circuit_key}': {length}m")
                break
        
        if circuit_length is None:
            logging.warning(f"Circuit length for '{location_name}' not found. Using default.")
            circuit_length = 5000  # Default fallback
    
    # Check if session has lap data
    if session.laps is None or session.laps.empty:
        error_tracker.log_race_error(year, 0, race_name, "NoLapData", "Session has no lap data")
        return pd.DataFrame()
    
    # Get weather data
    weather_data = session.weather_data
    if weather_data is None or weather_data.empty:
        logging.warning(f"No weather data for {race_name} ({year})")
    
    # Process each driver
    unique_drivers = session.laps['DriverNumber'].unique()
    logging.info(f"Processing {len(unique_drivers)} drivers for {race_name} ({year})")
    
    for driver_number in unique_drivers:
        try:
            driver_laps = session.laps.pick_drivers(driver_number)
            
            if driver_laps is None or driver_laps.empty:
                error_tracker.log_driver_error(year, race_name, str(driver_number), 
                                             "NoLapData", "No laps found for driver")
                continue
            
            # Get driver info
            driver_name = driver_laps.iloc[0].get('Driver', f"Driver_{driver_number}") if not driver_laps.empty else f"Driver_{driver_number}"
            
            # Process each lap for this driver
            for lap_index, lap in driver_laps.iterrows():
                try:
                    lap_data = extract_single_lap_data(
                        lap, year, race_name, location_name, driver_number, 
                        driver_name, circuit_length, weather_data
                    )
                    all_laps_data.append(lap_data)
                    
                except Exception as e:
                    lap_number = lap.get('LapNumber', 'Unknown')
                    error_tracker.log_lap_error(year, race_name, str(driver_number), 
                                               lap_number, type(e).__name__, str(e))
                    continue
                    
        except Exception as e:
            error_tracker.log_driver_error(year, race_name, str(driver_number), 
                                         type(e).__name__, str(e))
            continue
    
    if not all_laps_data:
        error_tracker.log_race_error(year, 0, race_name, "NoValidLapData", 
                                   "No valid lap data extracted from any driver")
        return pd.DataFrame()
    
    return pd.DataFrame(all_laps_data)

def extract_single_lap_data(lap, year: int, race_name: str, location_name: str,
                           driver_number: int, driver_name: str, circuit_length: float,
                           weather_data) -> Dict[str, Any]:
    """Extracts data for a single lap with error handling."""
    
    # Basic lap information
    lap_info = {
        'Year': year,
        'GranPrix': race_name,
        'Location': location_name,
        'DriverID': driver_number,
        'Driver': driver_name,
        'Team': lap.get('Team', 'Unknown'),
        'LapNumber': int(lap.get('LapNumber', 0)) if pd.notna(lap.get('LapNumber')) else 0,
        'Position': int(lap.get('Position', 0)) if pd.notna(lap.get('Position')) else np.nan,
        'LapTime': lap.get('LapTime').total_seconds() if pd.notna(lap.get('LapTime')) else np.nan,
        'Time': lap.get('Time').total_seconds() if pd.notna(lap.get('Time')) else np.nan,
        'Stint': int(lap.get('Stint', 0)) if pd.notna(lap.get('Stint')) else np.nan,
        'Compound': lap.get('Compound', 'UNKNOWN'),
        'TyreLife': int(lap.get('TyreLife', 0)) if pd.notna(lap.get('TyreLife')) else 0,
        'IsFreshTire': bool(lap.get('FreshTyre', False)),
        'PitInTime': bool(pd.notna(lap.get('PitInTime'))),
        'PitOutTime': bool(pd.notna(lap.get('PitOutTime'))),
    }
    
    # Calculate tire and driver meters
    if pd.notna(circuit_length) and pd.notna(lap_info['TyreLife']):
        lap_info['TireMeters'] = lap_info['TyreLife'] * circuit_length
    else:
        lap_info['TireMeters'] = np.nan
    
    if pd.notna(circuit_length) and pd.notna(lap_info['LapNumber']):
        lap_info['DriverMeters'] = lap_info['LapNumber'] * circuit_length
    else:
        lap_info['DriverMeters'] = np.nan
    
    # Weather data processing
    weather_features = extract_weather_data_for_lap(lap, weather_data)
    lap_info.update(weather_features)
    
    return lap_info

def extract_weather_data_for_lap(lap, weather_data) -> Dict[str, Any]:
    """Extracts weather data for a specific lap."""
    weather_features = {
        'AirTemp': np.nan,
        'Humidity': np.nan,
        'TrackTemp': np.nan,
        'Pressure': np.nan,
        'WindSpeed': np.nan,
        'WindDirection': np.nan,
        'Rainfall': False
    }
    
    if weather_data is None or weather_data.empty:
        return weather_features
    
    lap_start_time = lap.get('LapStartTime')
    lap_end_time = lap.get('Time')
    
    if pd.notna(lap_start_time) and pd.notna(lap_end_time):
        # Get weather data during lap
        lap_weather = weather_data[
            (weather_data['Time'] >= lap_start_time) & 
            (weather_data['Time'] <= lap_end_time)
        ]
        
        if not lap_weather.empty:
            weather_features.update({
                'AirTemp': lap_weather['AirTemp'].mean() if 'AirTemp' in lap_weather else np.nan,
                'Humidity': lap_weather['Humidity'].mean() if 'Humidity' in lap_weather else np.nan,
                'TrackTemp': lap_weather['TrackTemp'].mean() if 'TrackTemp' in lap_weather else np.nan,
                'Pressure': lap_weather['Pressure'].mean() if 'Pressure' in lap_weather else np.nan,
                'WindSpeed': lap_weather['WindSpeed'].mean() if 'WindSpeed' in lap_weather else np.nan,
                'WindDirection': lap_weather['WindDirection'].mean() if 'WindDirection' in lap_weather else np.nan,
                'Rainfall': bool(lap_weather['Rainfall'].any()) if 'Rainfall' in lap_weather else False
            })
        elif pd.notna(lap_start_time):
            # Use nearest weather point
            time_diff = (weather_data['Time'] - lap_start_time).abs()
            closest_idx = time_diff.idxmin()
            closest_weather = weather_data.loc[closest_idx]
            
            weather_features.update({
                'AirTemp': closest_weather.get('AirTemp', np.nan),
                'Humidity': closest_weather.get('Humidity', np.nan),
                'TrackTemp': closest_weather.get('TrackTemp', np.nan),
                'Pressure': closest_weather.get('Pressure', np.nan),
                'WindSpeed': closest_weather.get('WindSpeed', np.nan),
                'WindDirection': closest_weather.get('WindDirection', np.nan),
                'Rainfall': bool(closest_weather.get('Rainfall', False))
            })
    
    return weather_features

def calculate_time_deltas_vectorized(race_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized calculation of time deltas between drivers.
    More memory efficient than nested loops.
    """
    if race_df.empty or not all(col in race_df.columns for col in ['LapNumber', 'Position', 'Time']):
        # Add empty columns if missing
        for col in ['TimeDeltaToDriverAhead', 'TimeDeltaToDriverTwoAhead', 
                   'TimeDeltaToDriverBehind', 'TimeDeltaToDriverTwoBehind']:
            race_df[col] = np.nan
        return race_df
    
    # Initialize delta columns
    race_df['TimeDeltaToDriverAhead'] = np.nan
    race_df['TimeDeltaToDriverTwoAhead'] = np.nan
    race_df['TimeDeltaToDriverBehind'] = np.nan
    race_df['TimeDeltaToDriverTwoBehind'] = np.nan
    
    # Process lap by lap
    for lap_num in race_df['LapNumber'].unique():
        lap_mask = race_df['LapNumber'] == lap_num
        lap_data = race_df[lap_mask].copy()
        
        # Remove NaN positions and sort
        lap_data = lap_data.dropna(subset=['Position', 'Time']).sort_values('Position')
        
        if len(lap_data) < 2:
            continue
        
        # Calculate deltas using vectorized operations
        positions = lap_data['Position'].values
        times = lap_data['Time'].values
        indices = lap_data.index.values
        
        for i, idx in enumerate(indices):
            current_time = times[i]
            current_pos = positions[i]
            
            # Driver ahead (position - 1)
            if i > 0:
                race_df.loc[idx, 'TimeDeltaToDriverAhead'] = current_time - times[i-1]
                
                # Two ahead (position - 2)
                if i > 1:
                    race_df.loc[idx, 'TimeDeltaToDriverTwoAhead'] = current_time - times[i-2]
                else:
                    race_df.loc[idx, 'TimeDeltaToDriverTwoAhead'] = 0.0
            else:
                race_df.loc[idx, 'TimeDeltaToDriverAhead'] = 0.0
                race_df.loc[idx, 'TimeDeltaToDriverTwoAhead'] = 0.0
            
            # Driver behind (position + 1)
            if i < len(indices) - 1:
                race_df.loc[idx, 'TimeDeltaToDriverBehind'] = times[i+1] - current_time
                
                # Two behind (position + 2)
                if i < len(indices) - 2:
                    race_df.loc[idx, 'TimeDeltaToDriverTwoBehind'] = times[i+2] - current_time
                else:
                    race_df.loc[idx, 'TimeDeltaToDriverTwoBehind'] = 0.0
            else:
                race_df.loc[idx, 'TimeDeltaToDriverBehind'] = 0.0
                race_df.loc[idx, 'TimeDeltaToDriverTwoBehind'] = 0.0
    
    return race_df

# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_single_race(year: int, round_num: int, event, config: Dict[str, Any],
                       circuits_info: Dict[str, float], error_tracker: ErrorTracker,
                       progress_tracker: ProgressTracker, colab_env: ColabEnvironment) -> bool:
    """
    Processes a single race with comprehensive error handling.
    Returns True if successful, False otherwise.
    """
    race_name = event['EventName']
    official_name = event['OfficialEventName']
    location = event['Location']
    
    # Check if already processed
    safe_name = sanitize_filename(official_name)
    if progress_tracker.is_race_completed(year, round_num, safe_name):
        logging.info(f"Skipping already completed race: {race_name} ({year})")
        return True
    
    logging.info(f"🏁 Processing: {race_name} ({year}, Round {round_num}) at {location}")
    
    try:
        # Memory check before processing
        colab_env.check_memory_warning(threshold_percent=80.0)
        
        # Get session with retry
        session = ff1.get_session(year, round_num, 'R')
        
        # Load session data with retry mechanism
        data_to_load = config.get('data_to_load', {'laps': True, 'weather': True})
        valid_params = {k: v for k, v in data_to_load.items() if k in ['laps', 'weather', 'telemetry', 'messages']}
        
        session = load_session_with_retry(session, **valid_params)
        
        # Extract lap data
        race_df = extract_lap_data_with_error_handling(
            session, year, race_name, location, circuits_info, error_tracker
        )
        
        if race_df.empty:
            logging.warning(f"No data extracted for {race_name} ({year})")
            return False
        
        # Calculate time deltas
        logging.info(f"Calculating time deltas for {race_name} ({year})...")
        race_df = calculate_time_deltas_vectorized(race_df)
        
        # Save to Google Drive immediately
        output_dir = config.get('drive_output_dir', '/content/drive/MyDrive/F1_Project/processed_races')
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = f"{year}_{round_num:02d}_{safe_name}.parquet"
        output_path = os.path.join(output_dir, output_filename)
        
        race_df.to_parquet(output_path, index=False)
        logging.info(f"✅ Saved race data to: {output_path} ({len(race_df)} rows)")
        
        # Mark as completed
        progress_tracker.mark_race_completed(year, round_num, safe_name)
        
        # Force cleanup
        del race_df, session
        colab_env.force_cleanup()
        
        return True
        
    except SessionNotAvailableError as e:
        error_tracker.log_race_error(year, round_num, race_name, "SessionNotAvailable", str(e))
        logging.error(f"Session not available for {race_name} ({year}): {e}")
        return False
        
    except DataNotLoadedError as e:
        error_tracker.log_race_error(year, round_num, race_name, "DataNotLoaded", str(e))
        logging.error(f"Data could not be loaded for {race_name} ({year}): {e}")
        return False
        
    except MemoryError as e:
        error_tracker.log_race_error(year, round_num, race_name, "MemoryError", str(e))
        logging.error(f"Memory error processing {race_name} ({year}): {e}")
        colab_env.force_cleanup()
        return False
        
    except Exception as e:
        tb_str = traceback.format_exc()
        error_tracker.log_race_error(year, round_num, race_name, type(e).__name__, str(e), tb_str)
        logging.error(f"Unexpected error processing {race_name} ({year}): {e}")
        return False

# =============================================================================
# COLAB USER INTERFACE
# =============================================================================

def display_progress_widget(progress_tracker: ProgressTracker, error_tracker: ErrorTracker):
    """Displays interactive progress widget for Colab."""
    if not COLAB_ENV:
        return
    
    try:
        completion_pct = progress_tracker.get_completion_percentage()
        error_summary = error_tracker.get_summary()
        
        progress_html = f"""
        <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px;">
            <h3>🏎️ F1 Data Extraction Progress</h3>
            <div style="background-color: #f0f0f0; border-radius: 10px; padding: 3px;">
                <div style="background-color: #4CAF50; width: {completion_pct:.1f}%; height: 20px; border-radius: 7px; text-align: center; line-height: 20px; color: white; font-weight: bold;">
                    {completion_pct:.1f}%
                </div>
            </div>
            <p><strong>Completed:</strong> {progress_tracker.progress['total_races_completed']} / {progress_tracker.progress['total_races_planned']} races</p>
            <p><strong>Errors:</strong> {error_summary['total_errors']} ({error_summary['failed_races']} race failures)</p>
            <p><strong>Last Updated:</strong> {progress_tracker.progress.get('last_updated', 'Never')}</p>
        </div>
        """
        
        display(HTML(progress_html))
        
    except Exception as e:
        print(f"Could not display progress widget: {e}")

def estimate_completion_time(progress_tracker: ProgressTracker, start_time: datetime) -> str:
    """Estimates completion time based on current progress."""
    try:
        completed = progress_tracker.progress['total_races_completed']
        total = progress_tracker.progress['total_races_planned']
        
        if completed == 0:
            return "Calculating..."
        
        elapsed = datetime.now() - start_time
        avg_time_per_race = elapsed / completed
        remaining_races = total - completed
        eta = avg_time_per_race * remaining_races
        
        return f"ETA: {eta}"
        
    except Exception:
        return "Unknown"

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function for Colab-optimized F1 data extraction."""
    start_time = datetime.now()
    
    # Print banner
    print("="*60)
    print("🏎️  Formula 1 Data Extraction - Google Colab Edition")
    print("="*60)
    
    # Setup Colab environment
    colab_env = ColabEnvironment()
    if not colab_env.setup_environment():
        print("❌ Failed to setup Colab environment. Exiting.")
        return
    
    # Parse arguments (for notebook compatibility)
    try:
        args = parse_arguments()
    except SystemExit:
        # Handle argument parsing in notebook environment
        class DefaultArgs:
            config = "Vincenzo/config_colab.json"
            resume = True
            force_restart = False
        args = DefaultArgs()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return
    
    # Setup file paths for Google Drive
    drive_base = "/content/drive/MyDrive/F1_Project"
    log_file = os.path.join(drive_base, config.get('log_file', 'extraction_log.txt'))
    error_log_file = os.path.join(drive_base, config.get('error_log_file', 'error_log.json'))
    progress_file = os.path.join(drive_base, config.get('progress_file', 'progress_state.json'))
    
    # Setup logging
    setup_colab_logging(log_file, config.get('log_level', 'INFO'))
    
    logging.info("🚀 Starting F1 data extraction process (Colab Edition)")
    logging.info(f"Configuration: {args.config}")
    logging.info(f"Resume mode: {args.resume}")
    logging.info(f"Force restart: {args.force_restart}")
    
    # Setup error tracking and progress
    error_tracker = ErrorTracker(error_log_file)
    progress_tracker = ProgressTracker(progress_file)
    
    if args.force_restart:
        logging.info("🔄 Force restart requested - resetting progress")
        progress_tracker.progress = {
            'completed_races': [],
            'current_year': None,
            'current_round': None,
            'last_updated': None,
            'total_races_planned': 0,
            'total_races_completed': 0
        }
        progress_tracker._save_progress()
    
    # Setup FastF1 cache on Google Drive
    cache_path = os.path.join(drive_base, config.get('cache_directory', 'ff1_cache'))
    os.makedirs(cache_path, exist_ok=True)
    ff1.Cache.enable_cache(cache_path)
    logging.info(f"✅ FastF1 cache enabled at: {cache_path}")
    
    # Load circuit information
    circuits_info = get_circuit_info(config.get('circuits_info_file', 'Vincenzo/circuits_length.json'))
    logging.info(f"📍 Loaded {len(circuits_info)} circuit configurations")
    
    # Calculate total races to process
    years_to_process = config.get('years_to_process', [])
    total_races = 0
    
    for year in years_to_process:
        try:
            schedule = ff1.get_event_schedule(year, include_testing=False)
            conventional_races = schedule[schedule['EventFormat'] == 'conventional']
            races_per_year_limit = config.get('races_per_year_test_limit')
            
            if races_per_year_limit:
                total_races += min(len(conventional_races), races_per_year_limit)
            else:
                total_races += len(conventional_races)
                
        except Exception as e:
            logging.error(f"Error getting schedule for {year}: {e}")
    
    progress_tracker.progress['total_races_planned'] = total_races
    progress_tracker._save_progress()
    
    logging.info(f"📊 Total races to process: {total_races}")
    print(f"📊 Total races to process: {total_races}")
    
    # Initialize progress bar
    with tqdm(total=total_races, desc="Processing F1 Races", unit="race") as pbar:
        pbar.update(progress_tracker.progress['total_races_completed'])
        
        # Process each year
        for year in years_to_process:
            logging.info(f"🗓️ Processing year: {year}")
            print(f"\n🗓️ Processing year: {year}")
            
            try:
                # Get event schedule
                event_schedule = ff1.get_event_schedule(year, include_testing=False)
                conventional_races = event_schedule[event_schedule['EventFormat'] == 'conventional']
                
                races_per_year_limit = config.get('races_per_year_test_limit')
                if races_per_year_limit:
                    conventional_races = conventional_races.head(races_per_year_limit)
                    logging.info(f"Limited to {races_per_year_limit} races for testing")
                
                races_processed_this_year = 0
                
                # Process each race
                for index, event in conventional_races.iterrows():
                    race_name = event['EventName']
                    round_num = event['RoundNumber']
                    
                    # Update progress bar description
                    pbar.set_description(f"Processing {year} {race_name}")
                    
                    # Process the race
                    success = process_single_race(
                        year, round_num, event, config, circuits_info,
                        error_tracker, progress_tracker, colab_env
                    )
                    
                    if success:
                        races_processed_this_year += 1
                        pbar.update(1)
                        
                        # Display progress widget every few races
                        if races_processed_this_year % 5 == 0:
                            clear_output(wait=True)
                            display_progress_widget(progress_tracker, error_tracker)
                            print(f"\n🗓️ Processing year: {year}")
                    
                    # Memory check
                    if colab_env.check_memory_warning(threshold_percent=90.0):
                        logging.warning("⚠️ High memory usage detected. Forcing cleanup.")
                        colab_env.force_cleanup()
                
                logging.info(f"✅ Completed year {year}: {races_processed_this_year} races processed")
                
            except Exception as e:
                logging.error(f"Error processing year {year}: {e}")
                continue
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    error_summary = error_tracker.get_summary()
    completion_pct = progress_tracker.get_completion_percentage()
    
    print("\n" + "="*60)
    print("🏁 EXTRACTION COMPLETED!")
    print("="*60)
    print(f"⏱️  Total time: {duration}")
    print(f"📊 Completion: {completion_pct:.1f}% ({progress_tracker.progress['total_races_completed']}/{progress_tracker.progress['total_races_planned']} races)")
    print(f"❌ Total errors: {error_summary['total_errors']}")
    print(f"🏎️ Failed races: {error_summary['failed_races']}")
    print(f"👤 Failed drivers: {error_summary['failed_drivers']}")
    print(f"🔄 Failed laps: {error_summary['failed_laps']}")
    print(f"📁 Data saved to: {config.get('drive_output_dir', '/content/drive/MyDrive/F1_Project/processed_races')}")
    print(f"📝 Error log: {error_log_file}")
    print(f"📈 Progress log: {progress_file}")
    print("="*60)
    
    logging.info(f"🏁 Extraction completed. Duration: {duration}")
    logging.info(f"📊 Final statistics: {error_summary}")
    
    # Display final progress widget
    if COLAB_ENV:
        display_progress_widget(progress_tracker, error_tracker)

if __name__ == "__main__":
    main()
