import argparse
import json
import logging
import os
import pandas as pd
import numpy as np
import fastf1 as ff1
from datetime import timedelta # Aggiunto per gestire i timedelta se necessario
# Attempting to import a base FastF1Error and specific errors if possible
try:
    from fastf1.error import SessionNotAvailableError, DataNotLoadedError # Common location
    FastF1Error = ff1.FastF1Error # General FastF1 related error base class
except ImportError:
    try:
        from fastf1.core import SessionNotAvailableError, DataNotLoadedError # Alternative location
        FastF1Error = ff1.FastF1Error
    except ImportError:
        logging.warning(
            "Could not import specific FastF1 exceptions (SessionNotAvailableError, DataNotLoadedError) "
            "from fastf1.error or fastf1.core. Will use a generic FastF1Error if available, or base Exception."
        )
        # Define dummy exceptions if specific ones are not found
        class SessionNotAvailableError(Exception): pass
        class DataNotLoadedError(Exception): pass
        # Try to get a base FastF1Error if it exists, otherwise default to base Exception for broader catches
        try:
            FastF1Error = ff1.FastF1Error
        except AttributeError:
            logging.warning("ff1.FastF1Error not found, will use base Exception for some FastF1 errors.")
            FastF1Error = Exception # Fallback to base Exception

# --- CONFIGURATION AND LOGGING SETUP ---

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract and process Formula 1 race data.")
    parser.add_argument(
        "--config",
        type=str,
        default="Vincenzo/config.json",  # Default to the full processing config
        help="Path to the JSON configuration file (default: Vincenzo/config.json)"
    )
    return parser.parse_args()

def load_config(config_path):
    """Loads configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from configuration file: {config_path}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading configuration: {e}")
        raise

def setup_logging(log_file, config_log_level_str):
    """Sets up logging configuration."""
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        logging.info(f"Created log directory: {log_dir}")

    numeric_log_level = getattr(logging, config_log_level_str.upper(), None)
    if not isinstance(numeric_log_level, int):
        logging.warning(f"Invalid log level: {config_log_level_str}. Defaulting to INFO.")
        numeric_log_level = logging.INFO

    logging.basicConfig(
        level=numeric_log_level,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'), # Overwrite log file each run
            logging.StreamHandler() # Also log to console
        ]
    )
    logging.info(f"Logging setup complete. Log level: {config_log_level_str.upper()}. Log file: {log_file}")

# --- MAIN DATA EXTRACTION LOGIC ---

def get_circuit_info(circuits_file_path):
    """Loads circuit information (e.g., length) from a JSON file."""
    try:
        with open(circuits_file_path, 'r') as f:
            circuits_info = json.load(f)
        logging.info(f"Circuit information loaded from {circuits_file_path}")
        return circuits_info
    except FileNotFoundError:
        logging.error(f"Circuits information file not found: {circuits_file_path}")
        return {} # Return empty dict to handle missing file gracefully in calling function
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from circuits file: {circuits_file_path}")
        return {}
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading circuit information: {e}")
        return {}

# --- LAP DATA EXTRACTION ---

def extract_lap_data_from_session(session, year, race_name, location_name, circuits_info):
    """
    Extracts lap-by-lap data for all drivers from a loaded FastF1 session object.
    Includes basic weather data averaged over the lap.
    """
    laps_data = []
    
    # Get circuit length for calculating meters
    circuit_length = circuits_info.get(location_name)
    if circuit_length is None:
        # Fallback for similar names like 'Yas Marina' vs 'Yas Island'
        for k, v in circuits_info.items():
            if location_name in k or k in location_name:
                circuit_length = v
                logging.warning(f"Exact match for location '{location_name}' not found in circuits_info. Using '{k}': {v}m.")
                break
        if circuit_length is None:
            logging.error(f"Circuit length for '{location_name}' not found. TireMeters/DriverMeters will be NaN.")
            circuit_length = np.nan # Ensure calculations result in NaN if length is unknown

    if session.laps is None or session.laps.empty:
        logging.warning(f"No lap data available in the session for {race_name} ({year}).")
        return pd.DataFrame(laps_data) # Return empty DataFrame

    # Ensure weather data is available
    weather_data = session.weather_data
    if weather_data is None or weather_data.empty:
        logging.warning(f"No weather data available for {race_name} ({year}). Weather features will be NaN.")

    # Iterate through each driver's laps
    for driver_number in session.laps['DriverNumber'].unique():
        # Use pick_drivers instead of the deprecated pick_driver
        # pick_drivers can accept a single driver identifier (like number or abbreviation)
        driver_laps = session.laps.pick_drivers(driver_number) 
        if driver_laps is None or driver_laps.empty:
            logging.debug(f"No laps found for driver number {driver_number} using pick_drivers in {race_name} ({year}).")
            continue
        
        # Get the driver abbreviation from the first lap of this driver
        # Ensure there's at least one lap and the 'Driver' column exists
        if not driver_laps.empty and 'Driver' in driver_laps.columns:
            driver_name_abbreviation = driver_laps.iloc[0].get('Driver', f"Driver_{driver_number}")
        else:
            # Fallback if no laps or 'Driver' column is missing for some reason
            driver_name_abbreviation = f"Driver_{driver_number}"
            logging.warning(f"Could not determine driver abbreviation for {driver_number} in {race_name} ({year}). Using default.")


        for lap_index, lap in driver_laps.iterrows():
            lap_info = {
                'Year': year,
                'GranPrix': race_name,
                'Location': location_name,
                'DriverID': driver_number, # Using DriverNumber as DriverID
                'Driver': driver_name_abbreviation,
                'Team': lap.get('Team', np.nan),
                'LapNumber': int(lap.get('LapNumber', np.nan)),
                'Position': int(lap.get('Position', np.nan)) if pd.notna(lap.get('Position')) else np.nan,
                'LapTime': lap.get('LapTime').total_seconds() if pd.notna(lap.get('LapTime')) else np.nan,
                'Time': lap.get('Time').total_seconds() if pd.notna(lap.get('Time')) else np.nan, # Cumulative race time
                'Stint': int(lap.get('Stint', np.nan)) if pd.notna(lap.get('Stint')) else np.nan,
                'Compound': lap.get('Compound', 'UNKNOWN'),
                'TyreLife': int(lap.get('TyreLife', 0)) if pd.notna(lap.get('TyreLife')) else 0, # Default to 0 if NaN
                'IsFreshTire': bool(lap.get('FreshTyre', False)), # Default to False
                'PitInTime': bool(pd.notna(lap.get('PitInTime'))),
                'PitOutTime': bool(pd.notna(lap.get('PitOutTime'))),
            }

            # Calculate TireMeters and DriverMeters
            if pd.notna(circuit_length) and pd.notna(lap_info['TyreLife']):
                lap_info['TireMeters'] = lap_info['TyreLife'] * circuit_length
            else:
                lap_info['TireMeters'] = np.nan
            
            if pd.notna(circuit_length) and pd.notna(lap_info['LapNumber']):
                lap_info['DriverMeters'] = lap_info['LapNumber'] * circuit_length
            else:
                lap_info['DriverMeters'] = np.nan

            # Weather data averaging
            lap_start_time = lap.get('LapStartTime')
            lap_end_time = lap.get('Time') # This is effectively lap end time in session time

            if pd.notna(lap_start_time) and pd.notna(lap_end_time) and weather_data is not None and not weather_data.empty:
                lap_weather = weather_data[
                    (weather_data['Time'] >= lap_start_time) & 
                    (weather_data['Time'] <= lap_end_time)
                ]
                if not lap_weather.empty:
                    lap_info['AirTemp'] = lap_weather['AirTemp'].mean()
                    lap_info['Humidity'] = lap_weather['Humidity'].mean()
                    lap_info['TrackTemp'] = lap_weather['TrackTemp'].mean()
                    lap_info['Pressure'] = lap_weather['Pressure'].mean()
                    lap_info['WindSpeed'] = lap_weather['WindSpeed'].mean()
                    lap_info['WindDirection'] = lap_weather['WindDirection'].mean()
                    lap_info['Rainfall'] = lap_weather['Rainfall'].any() # True if any rain during the lap
                elif weather_data is not None and not weather_data.empty and pd.notna(lap_start_time) and 'Time' in weather_data.columns:
                    # Try to get the closest weather point if no data during lap interval
                    # Calculate absolute time difference to find the nearest point
                    time_diff = (weather_data['Time'] - lap_start_time).abs()
                    closest_idx = time_diff.idxmin() # Get index of the row with minimum difference
                    closest_weather = weather_data.loc[closest_idx]
                    
                    lap_info['AirTemp'] = closest_weather.get('AirTemp', np.nan)
                    lap_info['Humidity'] = closest_weather.get('Humidity', np.nan)
                    lap_info['TrackTemp'] = closest_weather.get('TrackTemp', np.nan)
                    lap_info['Pressure'] = closest_weather.get('Pressure', np.nan)
                    lap_info['WindSpeed'] = closest_weather.get('WindSpeed', np.nan)
                    lap_info['WindDirection'] = closest_weather.get('WindDirection', np.nan)
                    lap_info['Rainfall'] = bool(closest_weather.get('Rainfall', False))
                    logging.debug(f"No weather data during lap {lap_info['LapNumber']} for {driver_name_abbreviation}. Used nearest point.")

            else: # Fill with NaN if times or weather_data are missing
                lap_info['AirTemp'] = np.nan
                lap_info['Humidity'] = np.nan
                lap_info['TrackTemp'] = np.nan
                lap_info['Pressure'] = np.nan
                lap_info['WindSpeed'] = np.nan
                lap_info['WindDirection'] = np.nan
                lap_info['Rainfall'] = np.nan # Or False, depending on desired default
                if weather_data is None or weather_data.empty:
                     logging.debug(f"Weather data missing for session. Weather features NaN for lap {lap_info['LapNumber']} of {driver_name_abbreviation}.")
                else:
                    logging.debug(f"Lap start/end time missing for lap {lap_info['LapNumber']} of {driver_name_abbreviation}. Weather features NaN.")
            
            laps_data.append(lap_info)

    if not laps_data:
        logging.info(f"No lap data could be compiled for {race_name} ({year}).")
        return pd.DataFrame() # Return empty DataFrame

    return pd.DataFrame(laps_data)

# --- FEATURE ENGINEERING: TIME DELTAS ---

def calculate_time_deltas(race_df):
    """
    Calculates time deltas to drivers ahead and behind for each lap.
    Assumes race_df contains data for a single race and is sorted by LapNumber, then Position.
    The 'Time' column is expected to be cumulative race time in seconds.
    """
    if 'LapNumber' not in race_df.columns or 'DriverID' not in race_df.columns or \
       'Position' not in race_df.columns or 'Time' not in race_df.columns:
        logging.error("Required columns (LapNumber, DriverID, Position, Time) not found in DataFrame for delta calculation.")
        # Add empty columns to prevent downstream errors if they are expected
        race_df['TimeDeltaToDriverAhead'] = np.nan
        race_df['TimeDeltaToDriverTwoAhead'] = np.nan
        race_df['TimeDeltaToDriverBehind'] = np.nan
        race_df['TimeDeltaToDriverTwoBehind'] = np.nan
        return race_df

    # Initialize delta columns
    race_df['TimeDeltaToDriverAhead'] = np.nan
    race_df['TimeDeltaToDriverTwoAhead'] = np.nan
    race_df['TimeDeltaToDriverBehind'] = np.nan
    race_df['TimeDeltaToDriverTwoBehind'] = np.nan

    # Group by lap number to process each lap's standings
    for lap_num, lap_group in race_df.groupby('LapNumber'):
        # Sort by position within the lap to easily find drivers ahead/behind
        # Ensure Position is numeric and handle NaNs before sorting
        lap_group['Position'] = pd.to_numeric(lap_group['Position'], errors='coerce')
        lap_group_sorted = lap_group.dropna(subset=['Position']).sort_values(by='Position')
        
        num_drivers_in_lap = len(lap_group_sorted)

        for i, current_driver_row in enumerate(lap_group_sorted.iterrows()):
            current_driver_index = current_driver_row[0] # This is the original DataFrame index
            current_driver_time = current_driver_row[1]['Time']
            current_driver_pos = current_driver_row[1]['Position']

            if pd.isna(current_driver_time) or pd.isna(current_driver_pos):
                logging.debug(f"Skipping delta calculation for driver index {current_driver_index} on lap {lap_num} due to missing Time or Position.")
                continue
            
            # --- Deltas to drivers ahead ---
            if i > 0: # If not the first driver (P1)
                driver_ahead_row = lap_group_sorted.iloc[i-1]
                if pd.notna(driver_ahead_row['Time']):
                    race_df.loc[current_driver_index, 'TimeDeltaToDriverAhead'] = current_driver_time - driver_ahead_row['Time']
                else:
                     race_df.loc[current_driver_index, 'TimeDeltaToDriverAhead'] = np.nan
                
                if i > 1: # If not P1 or P2
                    driver_two_ahead_row = lap_group_sorted.iloc[i-2]
                    if pd.notna(driver_two_ahead_row['Time']):
                        race_df.loc[current_driver_index, 'TimeDeltaToDriverTwoAhead'] = current_driver_time - driver_two_ahead_row['Time']
                    else:
                        race_df.loc[current_driver_index, 'TimeDeltaToDriverTwoAhead'] = np.nan
                else: # P2
                    race_df.loc[current_driver_index, 'TimeDeltaToDriverTwoAhead'] = 0.0 # Or np.nan if preferred for "no one two ahead"
            else: # P1
                race_df.loc[current_driver_index, 'TimeDeltaToDriverAhead'] = 0.0
                race_df.loc[current_driver_index, 'TimeDeltaToDriverTwoAhead'] = 0.0

            # --- Deltas to drivers behind ---
            if i < num_drivers_in_lap - 1: # If not the last driver
                driver_behind_row = lap_group_sorted.iloc[i+1]
                if pd.notna(driver_behind_row['Time']):
                    race_df.loc[current_driver_index, 'TimeDeltaToDriverBehind'] = driver_behind_row['Time'] - current_driver_time
                else:
                    race_df.loc[current_driver_index, 'TimeDeltaToDriverBehind'] = np.nan

                if i < num_drivers_in_lap - 2: # If not last or second to last
                    driver_two_behind_row = lap_group_sorted.iloc[i+2]
                    if pd.notna(driver_two_behind_row['Time']):
                        race_df.loc[current_driver_index, 'TimeDeltaToDriverTwoBehind'] = driver_two_behind_row['Time'] - current_driver_time
                    else:
                        race_df.loc[current_driver_index, 'TimeDeltaToDriverTwoBehind'] = np.nan
                else: # Second to last
                     race_df.loc[current_driver_index, 'TimeDeltaToDriverTwoBehind'] = 0.0 # Or np.nan
            else: # Last driver
                race_df.loc[current_driver_index, 'TimeDeltaToDriverBehind'] = 0.0
                race_df.loc[current_driver_index, 'TimeDeltaToDriverTwoBehind'] = 0.0
                
    return race_df

# main function
def main():
    """Main function to orchestrate data extraction and processing."""
    args = parse_arguments()
    
    # Setup logging first, using a temporary basic config if setup_logging itself fails
    try:
        # Attempt to load config to get log file path and level for proper setup
        temp_config = load_config(args.config) # Load config to get log settings
        setup_logging(temp_config['log_file'], temp_config['log_level'])
        config = temp_config # Use the loaded config
    except Exception as e:
        # Fallback basic logging if config loading or initial setup fails
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.error(f"Failed to setup logging using config file {args.config}: {e}. Using basic logging.")
        # Try to load config again, or exit if essential config is missing
        try:
            config = load_config(args.config)
        except Exception:
            logging.error("Could not load configuration. Exiting.")
            return

    logging.info("Starting Formula 1 data extraction process...")
    logging.info(f"Using configuration file: {args.config}")
    logging.debug(f"Full configuration: {json.dumps(config, indent=2)}")

    # Enable FastF1 cache
    try:
        cache_path = config.get('cache_directory', 'ff1_cache') # Default if not in config
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
            logging.info(f"Created cache directory: {cache_path}")
        ff1.Cache.enable_cache(cache_path)
        logging.info(f"FastF1 cache enabled at: {cache_path}")
    except Exception as e:
        logging.error(f"Failed to enable FastF1 cache at {cache_path}: {e}")
        # Decide if this is a critical error; for now, we'll log and continue

    # Load circuit information
    circuits_data = get_circuit_info(config.get('circuits_info_file'))
    if not circuits_data:
        logging.warning("Circuit data is empty or could not be loaded. TireMeters/DriverMeters might be inaccurate.")

    all_races_dataframes = []
    years_to_process = config.get('years_to_process', [])
    races_per_year_test_limit = config.get('races_per_year_test_limit') # Might be None

    for year in years_to_process:
        logging.info(f"--- Processing Year: {year} ---")
        try:
            event_schedule = ff1.get_event_schedule(year, include_testing=False)
            if event_schedule.empty:
                logging.warning(f"No event schedule found for year {year}. Skipping.")
                continue
        except Exception as e:
            logging.error(f"Failed to get event schedule for year {year}: {e}")
            continue

        # Filter for conventional race events
        official_races = event_schedule[event_schedule['EventFormat'] == 'conventional']
        
        races_processed_this_year = 0
        for index, event in official_races.iterrows():
            if races_per_year_test_limit is not None and races_processed_this_year >= races_per_year_test_limit:
                logging.info(f"Reached test limit of {races_per_year_test_limit} races for year {year}. Skipping remaining races.")
                break

            race_name = event['EventName']
            race_location = event['Location'] # Used for circuit length lookup
            logging.info(f"Processing Race: {race_name} ({year}) at {race_location}")

            try:
                session = ff1.get_session(year, race_name, 'R') # 'R' for Race
            except Exception as e:
                logging.error(f"Failed to get session for {race_name} ({year}): {e}")
                continue

            try:
                data_to_load_config = config.get('data_to_load', {})
                data_to_load_config = config.get('data_to_load', {})
                # Filter for known valid parameters for session.load to avoid TypeError
                # Common valid params for session.load(): laps, weather, telemetry, messages.
                # 'position' is not a direct load parameter; position data comes with laps or via session.pos_data.
                # 'car_data' also needs to be loaded carefully, often with specific telemetry channels.
                valid_param_keys = ['laps', 'weather', 'telemetry', 'messages']
                actual_load_params = {k: v for k, v in data_to_load_config.items() if k in valid_param_keys}
                
                logging.info(f"Loading session data for {race_name} ({year}) with actual params: {actual_load_params}")
                session.load(**actual_load_params)
                logging.info(f"Session data loaded successfully for {race_name} ({year}).")
            # except ErgastConnectionError as e: # Example if we find the correct import
            #     logging.error(f"Ergast connection error loading session for {race_name} ({year}): {e}. Skipping race.")
            #     continue
            except SessionNotAvailableError as e:
                logging.error(f"Session data not available for {race_name} ({year}): {e}. Skipping race.")
                continue
            except DataNotLoadedError as e:
                logging.error(f"Specific data could not be loaded for {race_name} ({year}): {e}. Skipping race.")
                continue
            except TimeoutError as e: # Catching generic TimeoutError
                logging.error(f"Timeout error loading session for {race_name} ({year}): {e}. Skipping race.")
                continue
            except TypeError as e: # Specifically catch TypeErrors from load() due to bad params
                logging.error(f"TypeError during session.load() for {race_name} ({year}): {e}. Check 'data_to_load' in config. Skipping race.")
                continue
            except Exception as e:
                logging.error(f"An unexpected error occurred loading session for {race_name} ({year}): {e} (Type: {type(e)}). Skipping race.")
                continue

            race_df = extract_lap_data_from_session(session, year, race_name, race_location, circuits_data)
            
            if race_df is not None and not race_df.empty:
                logging.info(f"Calculating time deltas for {race_name} ({year})...")
                # Pass a copy to avoid SettingWithCopyWarning if calculate_time_deltas modifies it directly,
                # though current implementation modifies in place via .loc, it's safer.
                race_df_with_deltas = calculate_time_deltas(race_df.copy()) 
                all_races_dataframes.append(race_df_with_deltas)
                logging.info(f"Successfully processed and added data (including deltas) for {race_name} ({year}).")
            else:
                logging.warning(f"No lap data extracted by extract_lap_data_from_session for {race_name} ({year}), skipping delta calculation.")
            
            races_processed_this_year += 1

    if all_races_dataframes:
        logging.info("Concatenating all race dataframes...")
        final_df = pd.concat(all_races_dataframes, ignore_index=True)
        output_file = config.get('output_parquet_file', 'Vincenzo/all_races_data_raw.parquet')
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            final_df.to_parquet(output_file, index=False)
            logging.info(f"Final dataset saved to {output_file}")
        except Exception as e:
            logging.error(f"Failed to save final dataset to {output_file}: {e}")
    else:
        logging.warning("No data was processed. Final dataset not saved.")

    logging.info("Formula 1 data extraction process finished.")

if __name__ == "__main__":
    main()
