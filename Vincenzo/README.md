# Script di Estrazione Dati Formula 1 (`data_extraction.py`)

Questo script Python è progettato per estrarre dati dettagliati delle gare di Formula 1 utilizzando la libreria FastF1, arricchirli con feature calcolate e salvarli in formato Parquet per analisi successive.

## Caratteristiche Principali

-   **Configurabilità:** Utilizza file JSON per specificare i parametri di esecuzione.
-   **Estrazione Dati Completa:** Recupera dati dei giri, tempi, posizioni, stint, mescole degli pneumatici e dati meteorologici per ogni pilota in ogni gara.
-   **Feature Engineering:** Calcola automaticamente i distacchi temporali (in secondi) dai piloti immediatamente davanti/dietro e due posizioni davanti/dietro.
-   **Checkpoint e Ripresa:** Salva i dati di ogni gara elaborata in file Parquet individuali in una directory intermedia. Se lo script viene interrotto, può riprendere l'elaborazione dall'ultima gara non completata, evitando di riprocessare dati già salvati (questa funzione è attiva solo se non si usa `--simulate-writes`).
-   **Modalità di Test Flessibile:**
    -   Utilizza un file di configurazione separato (`config_test.json`) per limitare il numero di anni/gare da processare durante i test.
    -   Supporta un argomento da riga di comando `--simulate-writes` che esegue l'intera pipeline di elaborazione dati (incluse chiamate API a FastF1) ma **non scrive alcun file Parquet** su disco. Utile per testare la logica senza effetti collaterali di I/O.
-   **Logging Dettagliato:** Registra l'avanzamento, gli warning e gli errori in un file di log.
-   **Output Consolidato:** Al termine, tutti i dati delle gare elaborate (dai file intermedi, se non in modalità `--simulate-writes`) vengono consolidati in un unico file Parquet finale.

## Prerequisiti

-   Python 3.x
-   Librerie Python specificate nel file `requirements.txt` del progetto principale (assicurarsi che includa `fastf1`, `pandas`, `numpy`, `pyarrow`).
    ```bash
    pip install -r requirements.txt 
    ```
    (Se `requirements.txt` non esiste o è incompleto, installare almeno: `pip install fastf1 pandas numpy pyarrow`)
-   Una connessione internet per scaricare i dati da FastF1 (i dati vengono cachati localmente dopo il primo download).

## Struttura dei File

-   `Vincenzo/data_extraction.py`: Lo script principale.
-   `Vincenzo/config.json`: File di configurazione per l'elaborazione completa dei dati.
-   `Vincenzo/config_test.json`: File di configurazione per esecuzioni di test (con un numero ridotto di anni/gare).
-   `Vincenzo/circuits_length.json`: Contiene le lunghezze dei circuiti in metri.
-   `Vincenzo/intermediate_race_data/` (o `Vincenzo/intermediate_race_data_test/`): Directory creata dallo script per i file Parquet intermedi (uno per gara).
-   `Vincenzo/data_extraction.log` (o `Vincenzo/data_extraction_test.log`): File di log.
-   `Vincenzo/all_races_data_raw.parquet` (o `Vincenzo/test_races_data_raw.parquet`): File Parquet finale consolidato.

## File di Configurazione

I file `config.json` e `config_test.json` contengono i seguenti parametri principali:

-   `years_to_process`: Lista degli anni da elaborare (es. `[2020, 2021, 2022]`).
-   `races_per_year_test_limit`: (Solo in `config_test.json`) Numero massimo di gare da processare per ogni anno durante i test. `null` per nessun limite.
-   `cache_directory`: Path per la cache di FastF1 (es. `"ff1_cache"`).
-   `output_parquet_file`: Nome del file Parquet finale consolidato.
-   `log_file`: Nome del file di log.
-   `log_level`: Livello di logging (es. `"INFO"`, `"DEBUG"`).
-   `data_to_load`: Dizionario per specificare quali dati caricare con `session.load()` di FastF1 (es. `{"laps": true, "weather": true}`).
-   `circuits_info_file`: Path al file JSON con le lunghezze dei circuiti.
-   `intermediate_parquet_dir`: Path alla directory per i file Parquet intermedi.

## Come Eseguire lo Script

Aprire un terminale nella directory principale del progetto (`FASTF1`).

1.  **Esecuzione Standard (Elaborazione Completa):**
    Utilizza `config.json` per processare tutti gli anni e le gare specificate, con salvataggio effettivo dei file.
    ```bash
    python Vincenzo/data_extraction.py --config Vincenzo/config.json
    ```
    Oppure, dato che `Vincenzo/config.json` è il default:
    ```bash
    python Vincenzo/data_extraction.py
    ```

2.  **Esecuzione di Test (con Scrittura File Intermedi e Finali):**
    Utilizza `config_test.json` (che di solito specifica meno anni/gare) e scrive i file Parquet. Utile per testare la logica di checkpoint e ripresa.
    ```bash
    python Vincenzo/data_extraction.py --config Vincenzo/config_test.json
    ```

3.  **Esecuzione di Test con Simulazione delle Scritture (Nessuna Scrittura su Disco):**
    Utilizza `config_test.json` ed esegue tutta la pipeline di elaborazione dati (incluse chiamate API a FastF1) ma **non scrive alcun file Parquet**. Utile per testare rapidamente la logica di elaborazione senza effetti collaterali di I/O. La logica di ripresa basata su file non è attiva in questa modalità.
    ```bash
    python Vincenzo/data_extraction.py --config Vincenzo/config_test.json --simulate-writes
    ```

## Output

-   **File Parquet Intermedi:** Durante un'esecuzione normale (non `--simulate-writes`), i dati di ogni gara vengono salvati in file `.parquet` individuali nella directory specificata da `intermediate_parquet_dir`. Questi file servono come checkpoint.
-   **File Parquet Finale:** Al termine di un'esecuzione normale, tutti i file Parquet intermedi vengono consolidati in un unico file specificato da `output_parquet_file`.
-   **File di Log:** L'esecuzione dello script produce un file di log (es. `Vincenzo/data_extraction.log`) che contiene informazioni dettagliate, warning ed errori.

## Feature Estratte Principali

Oltre ai dati standard forniti da FastF1 (tempi, stint, gomme, meteo, ecc.), lo script calcola e include le seguenti feature per ogni pilota in ogni giro:
-   `TimeDeltaToDriverAhead`: Distacco temporale (secondi) dal pilota P-1.
-   `TimeDeltaToDriverTwoAhead`: Distacco temporale (secondi) dal pilota P-2.
-   `TimeDeltaToDriverBehind`: Distacco temporale (secondi) dal pilota P+1.
-   `TimeDeltaToDriverTwoBehind`: Distacco temporale (secondi) dal pilota P+2.

Questi dati sono pronti per essere utilizzati in analisi successive o per l'addestramento di modelli.
