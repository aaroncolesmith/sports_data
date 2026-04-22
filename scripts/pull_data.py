import pandas as pd
import datetime
import os
import sys
import time

# Ensure we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_todays_games, get_complete_game_results, filter_data_on_change, SPORT_INFO

HEADERS = {
    'Authority': 'api.actionnetwork',
    'Accept': 'application/json',
    'Origin': 'https://www.actionnetwork.com',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'
}

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/bets_db'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/results'))

def process_sport(sport, dates_or_weeks):
    print(f"\nProcessing {sport}...")
    
    # ensure data dir exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    db_path = os.path.join(DATA_DIR, f'{sport}_bets_db.csv')
    
    if os.path.exists(db_path):
        try:
            df_all = pd.read_csv(db_path)
            # Ensure date_scraped is datetime
            if 'date_scraped' in df_all.columns:
                df_all['date_scraped'] = pd.to_datetime(df_all['date_scraped'])
        except Exception as e:
            print(f"Error reading existing DB for {sport}: {e}")
            df_all = pd.DataFrame()
    else:
        df_all = pd.DataFrame()
        
    print(f"Fetching data for {sport}...")
    try:
        df_new = get_todays_games(sport, dates_or_weeks, HEADERS)
    except Exception as e:
        print(f"Error fetching data for {sport}: {e}")
        return

    if df_new is None or df_new.empty:
        print(f"No new data found for {sport}.")
        return

    # Add timestamp
    df_new['date_scraped'] = datetime.datetime.now()
    print(f"Found {len(df_new)} new rows.")
    
    # Concatenate
    df_combined = pd.concat([df_all, df_new], ignore_index=True)
    
    # Filter on change
    # check available columns
    dimension_cols = ['game_id', 'home_team', 'away_team']
    
    # Potential metric columns
    possible_metrics = [
        'home_money_line', 'away_money_line', 'total_score',
        'home_spread', 'away_spread', 
        'home_spread_odds', 'away_spread_odds',
        'over_odds', 'under_odds'
    ]
    
    metric_cols = [c for c in possible_metrics if c in df_combined.columns]
    
    if not metric_cols:
        print(f"Warning: No metric columns found for {sport}. Saving all data.")
        filtered_df = df_combined
    else:
        print(f"Filtering on change using metrics: {metric_cols}")
        # Ensure sorting by date_scraped to keep change logic correct
        if 'date_scraped' in df_combined.columns:
            df_combined = df_combined.sort_values('date_scraped')
            
        try:
            filtered_df = filter_data_on_change(df_combined, dimension_cols, metric_cols)
        except Exception as e:
            print(f"Error filtering data: {e}. Saving all.")
            filtered_df = df_combined

    print(f"Saving {len(filtered_df)} rows to {db_path} (was {len(df_all)})")
    filtered_df.to_csv(db_path, index=False)


def process_results(sport, dates_or_weeks):
    """Fetch completed game results and upsert into data/results/{sport}_results_db.csv."""
    print(f"\nFetching completed results for {sport}...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_path = os.path.join(RESULTS_DIR, f'{sport}_results_db.csv')

    if os.path.exists(results_path):
        try:
            df_existing = pd.read_csv(results_path)
        except Exception as e:
            print(f"Error reading existing results for {sport}: {e}")
            df_existing = pd.DataFrame()
    else:
        df_existing = pd.DataFrame()

    try:
        df_new = get_complete_game_results(sport, dates_or_weeks, HEADERS)
    except Exception as e:
        print(f"Error fetching results for {sport}: {e}")
        return

    if df_new is None or df_new.empty:
        print(f"No completed games found for {sport}.")
        return

    df_new['date_scraped'] = datetime.datetime.now()
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    # Keep the most recent scrape for each game
    df_combined = df_combined.drop_duplicates(subset=['game_id'], keep='last')

    print(f"Saving {len(df_combined)} result rows for {sport} (was {len(df_existing)})")
    df_combined.to_csv(results_path, index=False)


def main():
    today = datetime.date.today()
    date_format = '%Y%m%d'

    # Odds tracking: look back 7 days (catches recently completed games) and forward 3 days
    odds_dates = [(today + datetime.timedelta(days=i)).strftime(date_format) for i in range(-7, 4)]
    # Results: look back 14 days to ensure we don't miss any completions
    results_dates = [(today + datetime.timedelta(days=i)).strftime(date_format) for i in range(-14, 1)]

    date_sports = ['nba', 'ncaab', 'soccer', 'mlb']

    for sport in date_sports:
        if sport in SPORT_INFO:
            process_sport(sport, odds_dates)
            process_results(sport, results_dates)
        else:
            print(f"Skipping {sport} (not in SPORT_INFO)")

    # Week-based sports: cover full season range (reg season + playoffs)
    all_weeks = list(range(1, 23))
    week_sports = ['nfl', 'ncaaf']

    for sport in week_sports:
        if sport in SPORT_INFO:
            process_sport(sport, all_weeks)
            process_results(sport, all_weeks)
        else:
            print(f"Skipping {sport} (not in SPORT_INFO)")

if __name__ == "__main__":
    main()
