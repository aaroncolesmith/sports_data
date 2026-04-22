import pandas as pd
import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_complete_game_results, SPORT_INFO

HEADERS = {
    'Authority': 'api.actionnetwork',
    'Accept': 'application/json',
    'Origin': 'https://www.actionnetwork.com',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'
}

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/results'))


def process_results_sport(sport, dates_or_weeks):
    print(f"\nBackfilling results for {sport}...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_path = os.path.join(RESULTS_DIR, f'{sport}_results_db.csv')

    if os.path.exists(results_path):
        try:
            df_existing = pd.read_csv(results_path)
            print(f"Loaded {len(df_existing)} existing result rows.")
        except Exception as e:
            print(f"Error reading existing results: {e}")
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
    df_combined = df_combined.drop_duplicates(subset=['game_id'], keep='last')

    print(f"Saving {len(df_combined)} result rows for {sport} (was {len(df_existing)})")
    df_combined.to_csv(results_path, index=False)


def main():
    print("Starting historical results backfill (last 365 days)...")

    today = datetime.date.today()
    date_format = '%Y%m%d'

    dates = [(today - datetime.timedelta(days=i)).strftime(date_format) for i in range(365)]
    dates.reverse()  # oldest first

    date_sports = ['nba', 'ncaab', 'soccer', 'mlb']
    for sport in date_sports:
        if sport in SPORT_INFO:
            process_results_sport(sport, dates)
        else:
            print(f"Skipping {sport} (not in SPORT_INFO)")

    # NFL/NCAAF: cover full season (reg season weeks 1-18 + playoffs 19-22)
    weeks = list(range(1, 23))
    week_sports = ['nfl', 'ncaaf']
    for sport in week_sports:
        if sport in SPORT_INFO:
            process_results_sport(sport, weeks)
        else:
            print(f"Skipping {sport} (not in SPORT_INFO)")


if __name__ == "__main__":
    main()
