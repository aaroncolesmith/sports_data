import pandas as pd
import datetime
import os
import sys
import time

# Ensure we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_todays_games, filter_data_on_change, SPORT_INFO

HEADERS = {
    'Authority': 'api.actionnetwork',
    'Accept': 'application/json',
    'Origin': 'https://www.actionnetwork.com',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'
}

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/bets_db'))

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
        
    print(f"Fetching data for {sport} ({len(dates_or_weeks)} items)...")
    
    # Chunking requests to avoid potential issues with too many dates in one request URL if using GET
    # actually get_todays_games calls fetch_all_games_data which calls fetch_and_process_data PER date/week
    # so we can pass the whole list, but let's be safe and catch errors in the loop there.
    # The existing get_todays_games iterates over the list passed to it.
    
    try:
        # We don't filter for market_id '15' here (get_todays_games does)
        # If we want historical data, we likely want completed games too?
        # The user said "pulling data ... saving it off ... when there have been changes".
        # For historical data, "changes" in odds might not be as relevant as just getting the final line or closing line.
        # However, the user request says "do an adhoc run of historical data... only need to run it ad-hoc".
        # AND "I want to only save it when there have been changes in the dateset (like I am doing before)"
        # This implies using the same logic.
        
        # NOTE: get_todays_games filters by market_id='15' if present. 
        # Historical games might not have market_id='15' active? Or maybe they do.
        # Let's trust the util function.
        
        df_new = get_todays_games(sport, dates_or_weeks, HEADERS)
        
    except Exception as e:
        print(f"Error fetching data for {sport}: {e}")
        return

    if df_new is None or df_new.empty:
        print(f"No new data found for {sport}.")
        return

    # Add timestamp
    df_new['date_scraped'] = datetime.datetime.now()
    
    # For historical data, we probably want EVERYTHING, not just scheduled?
    # But pull_data filters for 'scheduled'.
    # If the user wants historical data, they probably want completed games content.
    # The user said "pulling data from action network and saving it off".
    # And "historical data".
    # Scheduled means future. Completed means history.
    # So I MUST NOT filter by 'scheduled' for this historical pull.
    # I will save everything.
    
    print(f"Found {len(df_new)} new rows.")
    
    # Concatenate
    df_combined = pd.concat([df_all, df_new], ignore_index=True)
    
    # Filter on change logic is still useful to dedup if we run this multiple times
    # check available columns
    dimension_cols = ['game_id', 'home_team', 'away_team']
    
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
        if 'date_scraped' in df_combined.columns:
            df_combined = df_combined.sort_values('date_scraped')
            
        try:
            filtered_df = filter_data_on_change(df_combined, dimension_cols, metric_cols)
        except Exception as e:
            print(f"Error filtering data: {e}. Saving all.")
            filtered_df = df_combined

    print(f"Saving {len(filtered_df)} rows to {db_path} (was {len(df_all)})")
    filtered_df.to_csv(db_path, index=False)


def main():
    print("Starting historical data pull (last 365 days)...")
    
    today = datetime.date.today()
    date_format = '%Y%m%d'
    
    # Last 365 days
    dates = [(today - datetime.timedelta(days=i)).strftime(date_format) for i in range(365)]
    # Reverse to start from oldest (optional, but nice for timeline)
    dates.reverse()

    # Split into chunks of 30 days to avoid overwhelming logs/memory or timeouts if utils doesn't handle it gracefully
    # Although utils iterates, one big DataFrame might be heavy.
    # But 365 days isn't crazy for text data.
    
    date_sports = ['nba', 'ncaab', 'soccer', 'mlb']
    
    for sport in date_sports:
        if sport in SPORT_INFO:
            process_sport(sport, dates)
        else:
            print(f"Skipping {sport} (not in SPORT_INFO)")

    # Week-based sports
    # 52 weeks covers a year
    weeks = list(range(1, 53)) 
    # NOTE: Action Network might use specific week numbers for seasons.
    # But usually 1-18 is reg season, then chunks for playoffs.
    # Safest is probably just request 1-52? Or maybe standard NFL is 1-22.
    # Let's try 1-30 just to be safe and cover playoffs.
    weeks = list(range(1, 40)) # 1-18 reg, 19-22 playoffs?
    
    week_sports = ['nfl', 'ncaaf']
    
    for sport in week_sports:
        if sport in SPORT_INFO:
            process_sport(sport, weeks)
        else:
            print(f"Skipping {sport} (not in SPORT_INFO)")

if __name__ == "__main__":
    main()
