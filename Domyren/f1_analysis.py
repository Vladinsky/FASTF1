import fastf1
import pandas as pd

def get_lap_data(driver, event_name, year=2023):
    """
    Estrae i dati del giro per un pilota specifico in una gara specifica
    
    Args:
        driver (str): Nome abbreviato del pilota (es. 'HAM')
        event_name (str): Nome della gara (es. 'Italian Grand Prix')
        year (int): Anno della stagione (default 2023)
    
    Returns:
        DataFrame: Dati dei giri con le colonne richieste
    """
    try:
        # Carica la sessione
        session = fastf1.get_session(year, event_name, 'R')
        session.load(telemetry=False, weather=False)
        
        # Ottieni i dati del pilota
        driver_laps = session.laps.pick_driver(driver)
        
        if len(driver_laps) == 0:
            raise ValueError(f"Nessun dato disponibile per il pilota {driver}")
        
        # Identifica i pit stop
        pit_stops = (driver_laps['Compound'] != driver_laps['Compound'].shift(1)) | \
                   (driver_laps['LapTime'] > driver_laps['LapTime'].mean() * 1.5)
        
        # Crea DataFrame con i dati richiesti
        data = {
            'LapNumber': driver_laps['LapNumber'],
            'Compound': driver_laps['Compound'],
            'LapTimeDriver': driver_laps['LapTime'],
            'Sector1Time': driver_laps['Sector1Time'].dt.total_seconds(),
            'Sector2Time': driver_laps['Sector2Time'].dt.total_seconds(),
            'Sector3Time': driver_laps['Sector3Time'].dt.total_seconds(),
            'Position': driver_laps['Position']
        }
        
        # Calcola DriverAhead e DriverBehind usando la posizione giro per giro
        all_laps = session.laps
        driver_codes = {drv: session.get_driver(drv).Abbreviation for drv in session.drivers}
        driver_ahead = []
        driver_behind = []
        diff_ahead = []
        diff_behind = []
        for idx, lap in driver_laps.iterrows():
            lap_number = lap['LapNumber']
            pos = lap['Position']
            lap_time = lap['LapTime']
            same_lap = all_laps[all_laps['LapNumber'] == lap_number]
            ahead_abbr = ''
            behind_abbr = ''
            lap_time_ahead = None
            lap_time_behind = None
            if not same_lap.empty and pd.notna(pos):
                if pos > 1:
                    ahead_row = same_lap[same_lap['Position'] == pos - 1]
                    if not ahead_row.empty:
                        ahead_abbr = ahead_row.iloc[0]['Driver']
                        ahead_abbr = driver_codes.get(ahead_abbr, ahead_abbr)
                        lap_time_ahead = ahead_row.iloc[0]['LapTime']
                if pos < same_lap['Position'].max():
                    behind_row = same_lap[same_lap['Position'] == pos + 1]
                    if not behind_row.empty:
                        behind_abbr = behind_row.iloc[0]['Driver']
                        behind_abbr = driver_codes.get(behind_abbr, behind_abbr)
                        lap_time_behind = behind_row.iloc[0]['LapTime']
            driver_ahead.append(ahead_abbr if ahead_abbr else 'N/A')
            driver_behind.append(behind_abbr if behind_abbr else 'N/A')
            # Calcolo differenze
            # Differenza con il pilota davanti
            if lap_time is not pd.NaT and lap_time_ahead is not None and pd.notna(lap_time_ahead):
                if lap_time < lap_time_ahead:
                    diff = (lap_time_ahead - lap_time).total_seconds()
                else:
                    diff = (lap_time - lap_time_ahead).total_seconds()
                diff_ahead.append(diff)
            else:
                diff_ahead.append('N/A')
            # Differenza con il pilota dietro
            if lap_time is not pd.NaT and lap_time_behind is not None and pd.notna(lap_time_behind):
                if lap_time < lap_time_behind:
                    diff = (lap_time_behind - lap_time).total_seconds()
                else:
                    diff = (lap_time - lap_time_behind).total_seconds()
                diff_behind.append(diff)
            else:
                diff_behind.append('N/A')
        data['DriverAhead'] = driver_ahead
        data['DriverBehind'] = driver_behind
        data['DiffToAhead'] = diff_ahead
        data['DiffToBehind'] = diff_behind
        
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Errore durante l'estrazione dei dati: {str(e)}")
        return pd.DataFrame()

def get_available_events(year):
    """Restituisce la lista delle gare disponibili per un anno specifico"""
    try:
        schedule = fastf1.get_event_schedule(year)
        return schedule['EventName'].tolist()
    except Exception as e:
        print(f"Errore nel recupero delle gare: {str(e)}")
        return []

def get_available_drivers(year, event_name):
    """Restituisce la lista dei piloti disponibili per una gara specifica"""
    try:
        session = fastf1.get_session(year, event_name, 'R')
        session.load(telemetry=False, weather=False)
        return session.drivers
    except Exception as e:
        print(f"Errore nel recupero dei piloti: {str(e)}")
        return []

if __name__ == "__main__":
    print("\n=== Analisi dati F1 ===\n")
    
    # Input anno con controllo
    while True:
        year_input = input("Inserisci l'anno della stagione (premere Invio per 2023): ").strip()
        try:
            year = int(year_input) if year_input else 2023
            events = get_available_events(year)
            if events:
                break
            print(f"Nessuna gara disponibile per l'anno {year}. Riprova.\n")
        except ValueError:
            print("Anno non valido. Inserisci un numero intero.\n")
    
    # Selezione gara
    print(f"\nGare disponibili per il {year}:")
    for i, event in enumerate(events, 1):
        print(f"{i}. {event}")
    
    while True:
        event_choice = input("\nSeleziona il numero della gara: ").strip()
        try:
            if event_choice.isdigit() and 1 <= int(event_choice) <= len(events):
                event = events[int(event_choice)-1]
                break
            print("Scelta non valida. Riprova.\n")
        except ValueError:
            print("Input non valido. Riprova.\n")
    
    # Selezione pilota
    drivers = get_available_drivers(year, event)
    print(f"\nPiloti disponibili per {event}:")
    for driver in drivers:
        print(f"- {driver}")
    
    while True:
        driver = input("\nInserisci il codice pilota (es. '44' per Hamilton): ").strip()
        if driver in drivers:
            break
        print("Codice pilota non valido o non presente in questa gara. Riprova.\n")
    
    # Esegui analisi
    try:
        df = get_lap_data(driver, event, year)
        if not df.empty:
            if df[['LapNumber', 'Compound', 'LapTimeDriver']].isnull().values.any():
                print("\nAttenzione: alcuni dati mancanti nei giri registrati")
            
            filename = f"{driver}_{event.replace(' ', '_')}_laps.csv"
            df.to_csv(filename, index=False)
            print(f"\nFile CSV creato con successo: {filename}")
            print(f"Numero di giri registrati: {len(df)}")
        else:
            print("\nNessun dato disponibile per i parametri specificati")
    except Exception as e:
        print(f"\nErrore durante l'elaborazione: {str(e)}")