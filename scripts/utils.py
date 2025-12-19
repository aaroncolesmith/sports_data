import requests
import pandas as pd
import numpy as np
import os


SPORT_INFO = {
        'soccer': {'prefix': 'soccer', 'full_name': 'soccer'},
        'ncaaf': {'prefix': 'ncaaf', 'full_name': 'college football'},
        'nfl': {'prefix': 'nfl', 'full_name': 'professional american football'},
        'mlb': {'prefix': 'mlb', 'full_name': 'baseball'},
        'nba': {'prefix': 'nba', 'full_name': 'basketball'},
        'ncaab':{'prefix': 'ncaab', 'full_name': 'college basketball'},
    }

def aggregate_betting_data(df: pd.DataFrame, group_by_cols: list, metric_cols: list) -> pd.DataFrame:
    """
    Aggregates a DataFrame by specified dimensions to get the first, average, 
    and last value for a list of metric columns.

    The function assumes the DataFrame is already sorted chronologically 
    for 'first' and 'last' to be meaningful.

    Args:
        df (pd.DataFrame): The input DataFrame. Must be sorted by time or sequence.
        group_by_cols (list): A list of column names to group the data by.
        metric_cols (list): A list of column names to be aggregated.

    Returns:
        pd.DataFrame: A new, aggregated DataFrame.
        
    Raises:
        ValueError: If any of the provided group-by or metric columns
                    do not exist in the DataFrame.
    """
    # --- Input Validation ---
    all_cols = group_by_cols + metric_cols
    if not all(col in df.columns for col in all_cols):
        missing_cols = [col for col in all_cols if col not in df.columns]
        raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")

    # --- Build Aggregation Dictionary ---
    # This dictionary will define the aggregations for each metric column.
    # We will rename the columns to include the aggregation type (e.g., 'num_bets_first').
    agg_config = {}
    for col in metric_cols:
        agg_config[f'{col}_first'] = (col, 'first')
        agg_config[f'{col}_avg'] = (col, 'mean')
        agg_config[f'{col}_last'] = (col, 'last')
    
    # --- Perform Grouping and Aggregation ---
    # The **agg_config unpacks the dictionary to pass its contents as arguments.
    # .reset_index() converts the grouped output back into a DataFrame.
    aggregated_df = df.groupby(group_by_cols).agg(**agg_config).reset_index()

    return aggregated_df


def filter_data_on_change(df: pd.DataFrame, dimensions: list, metrics: list) -> pd.DataFrame:
    """
    Filters a DataFrame to keep the first row, the last row, and any rows
    where metric values have changed for each group of dimensions.

    The function assumes the DataFrame is already sorted in the desired
    chronological order.

    Args:
        df (pd.DataFrame): The input DataFrame. Must be sorted by time or sequence.
        dimensions (list): A list of column names to use for grouping the data.
        metrics (list): A list of column names to check for value changes.

    Returns:
        pd.DataFrame: A new, filtered DataFrame.
    
    Raises:
        ValueError: If any of the provided dimension or metric columns
                    do not exist in the DataFrame.
    """
    # --- Input Validation ---
    if not all(col in df.columns for col in dimensions):
        raise ValueError("One or more dimension columns are not in the DataFrame.")
    if not all(col in df.columns for col in metrics):
        raise ValueError("One or more metric columns are not in the DataFrame.")

    # --- Group Processing Function ---
    def process_group(group: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a single group to identify which rows should be kept based on the rules.
        """
        # If a group has 2 or fewer rows, it already represents the first, last,
        # and any intermediate changes, so we keep all of them.
        if len(group) <= 2:
            return group

        # Identify rows where any metric has changed compared to the previous row.
        # .shift() gets the previous row's data.
        # .ne() compares for non-equality.
        # .any(axis=1) checks if at least one metric changed in a given row.
        # The first row is always kept because comparing it to a shifted (NaN) row yields True.
        has_changed = group[metrics].ne(group[metrics].shift()).any(axis=1)

        # Create a boolean mask to explicitly mark the last row for keeping.
        is_last = pd.Series(False, index=group.index)
        is_last.iloc[-1] = True

        # Combine the conditions: a row is kept if a metric has changed OR it's the last row.
        mask_to_keep = has_changed | is_last

        return group[mask_to_keep]

    # Group the DataFrame by the specified dimensions and apply the filtering logic.
    # group_keys=False prevents the group labels from being added as an index.
    filtered_df = df.groupby(dimensions, sort=False, group_keys=False).apply(process_group)

    return filtered_df


def fetch_and_process_data(url, headers):
    """
    Fetches detailed sports betting data from the API and processes it into a pandas DataFrame.
    
    This function handles nested JSON structures and potential missing data points
    by using defensive programming practices. It extracts moneyline, spread, and total
    betting information, including ticket and money percentages, for each market.

    Args:
        url (str): The API endpoint URL.
        headers (dict): The HTTP headers to include in the request.

    Returns:
        pandas.DataFrame or None: A DataFrame containing the extracted betting data, 
                                  or None if the request fails.
    """
    try:
        # Make the GET request to the API with the specified headers
        print('global version')
        print("Fetching data from the Action Network API...")
        print(url)
        response = requests.get(url, headers=headers)
        
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        
        # Parse the JSON data from the response
        response_json = response.json()
        print("Data successfully fetched.")

        # Check for the top-level 'games' key
        if 'games' not in response_json:
            print("No 'games' key found in the API response. Exiting.")
            return None
        
        # Initialize an empty list to store dictionaries for each market permutation
        all_market_data = []

        # Helper function to safely get a value from a DataFrame
        def get_value_from_df(df, side, column, default='N/A'):
            try:
                return df.loc[df['side'] == side][column].values[0]
            except (KeyError, IndexError):
                return default

        # Iterate through each game
        for game in response_json['games']:
            # Use .get() to safely retrieve nested data with a default value
            game_id = game.get('id', 'N/A')

            num_bets = game.get('num_bets', 0)
            
            # Safely get team data and determine home/away based on the 'home_away_team_id'
            teams_data = game.get('teams', [])
            home_team_data = next((team for team in teams_data if team.get('id') == game.get('home_team_id')), {})
            away_team_data = next((team for team in teams_data if team.get('id') == game.get('away_team_id')), {})

            # Extract general game information
            league_name = game.get('league_name', 'N/A')
            home_team = home_team_data.get('display_name', 'N/A')
            away_team = away_team_data.get('display_name', 'N/A')
            home_team_id = home_team_data.get('id', 'N/A')
            away_team_id = away_team_data.get('id', 'N/A')
            status = game.get('status', 'N/A')
            start_time = game.get('start_time', 'N/A')
            
            # Extract score data from the 'boxscore' object
            boxscore_data = game.get('boxscore', {})
            home_score = boxscore_data.get('total_home_points', 'N/A')
            away_score = boxscore_data.get('total_away_points', 'N/A')

            # Get the home pitcher's stats
            home_pitcher_data = game.get('player_stats', {}).get('home', [])
            if home_pitcher_data:
                home_pitcher_stats = home_pitcher_data[0]
                home_pitcher = home_pitcher_stats.get('player_id')
                home_pitcher_era = home_pitcher_stats.get('pitching', {}).get('era')
                home_pitcher_k9 = home_pitcher_stats.get('pitching', {}).get('k9')
                home_pitcher_ip = home_pitcher_stats.get('pitching', {}).get('ip_2')
                home_pitcher_starts = home_pitcher_stats.get('pitching', {}).get('games', {}).get('start')
                home_pitcher_win = home_pitcher_stats.get('pitching', {}).get('games', {}).get('win')
                home_pitcher_loss = home_pitcher_stats.get('pitching', {}).get('games', {}).get('loss')
            else:
                home_pitcher = home_pitcher_era = home_pitcher_k9 = home_pitcher_ip = home_pitcher_starts = home_pitcher_win = home_pitcher_loss = None

            # Handle the away pitcher's stats with a check
            away_pitcher_data = game.get('player_stats', {}).get('away', [])
            if away_pitcher_data:
                away_pitcher_stats = away_pitcher_data[0]
                away_pitcher = away_pitcher_stats.get('player_id')
                away_pitcher_era = away_pitcher_stats.get('pitching', {}).get('era')
                away_pitcher_k9 = away_pitcher_stats.get('pitching', {}).get('k9')
                away_pitcher_ip = away_pitcher_stats.get('pitching', {}).get('ip_2')
                away_pitcher_starts = away_pitcher_stats.get('pitching', {}).get('games', {}).get('start')
                away_pitcher_win = away_pitcher_stats.get('pitching', {}).get('games', {}).get('win')
                away_pitcher_loss = away_pitcher_stats.get('pitching', {}).get('games', {}).get('loss')
            else:
                # Assign 'None' or a default value if the away pitcher data is missing
                away_pitcher = away_pitcher_era = away_pitcher_k9 = away_pitcher_ip = away_pitcher_starts = away_pitcher_win = away_pitcher_loss = None

            # Print a quick summary to show what was extracted for the game
            print(f"\n--- Game Details ---")
            print(f"Game ID: {game_id}")
            print(f"League: {league_name}")
            print(f"Matchup: {home_team} vs {away_team}")
            print(f"Home Team ID: {home_team_id}")
            print(f"Away Team ID: {away_team_id}")
            print(f"Status: {status}")
            print(f"Start Time: {start_time}")
            print(f"Home Score: {home_score}")
            print(f"Away Score: {away_score}")
            print(f"--------------------")

            # Initialize a dictionary with all the base game data and default 'N/A' for market data.
            # This is done for every game, regardless of whether it has markets or not.
            base_row_data = {
                'game_id': game_id,
                'league_name': league_name,
                'home_team': home_team,
                'away_team': away_team,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'status': status,
                'home_score': home_score,
                'away_score': away_score,
                'home_pitcher': home_pitcher,
                'home_pitcher_era': home_pitcher_era,
                'home_pitcher_k9': home_pitcher_k9,
                'home_pitcher_ip': home_pitcher_ip,
                'home_pitcher_starts': home_pitcher_starts,
                'home_pitcher_win': home_pitcher_win,
                'home_pitcher_loss': home_pitcher_loss,
                'away_pitcher': away_pitcher,
                'away_pitcher_era': away_pitcher_era,
                'away_pitcher_k9': away_pitcher_k9,
                'away_pitcher_ip': away_pitcher_ip,
                'away_pitcher_starts': away_pitcher_starts,
                'away_pitcher_win': away_pitcher_win,
                'away_pitcher_loss': away_pitcher_loss,
                'start_time': start_time,
                'market_id': 'N/A',
                'book_id': 'N/A',
                'event_id': 'N/A',
                'num_bets': num_bets,
                'home_money_line': 'N/A',
                'home_ml_ticket_pct': 'N/A',
                'home_ml_money_pct': 'N/A',
                'away_money_line': 'N/A',
                'away_ml_ticket_pct': 'N/A',
                'away_ml_money_pct': 'N/A',
                'tie_money_line': 'N/A',
                'tie_ml_ticket_pct': 'N/A',
                'tie_ml_money_pct': 'N/A',
                'total_score': 'N/A',
                'over_odds': 'N/A',
                'under_odds': 'N/A',
                'over_ticket_pct': 'N/A',
                'over_money_pct': 'N/A',
                'under_ticket_pct': 'N/A',
                'under_money_pct': 'N/A',
                'home_spread': 'N/A',
                'home_spread_odds': 'N/A',
                'home_spread_ticket_pct': 'N/A',
                'home_spread_money_pct': 'N/A',
                'away_spread': 'N/A',
                'away_spread_odds': 'N/A',
                'away_spread_ticket_pct': 'N/A',
                'away_spread_money_pct': 'N/A'
            }

            # Check for the 'markets' key within the game data
            if 'markets' in game:
                # Use a flag to check if any market data was successfully processed
                market_found = False
                # Iterate through each market within the game
                for market_id, market in game['markets'].items():
                    # Check if the market has an 'event' and 'moneyline' data
                    if 'event' in market and 'moneyline' in market['event']:
                        market_found = True
                        print(f"Processing Game ID: {game_id}, Market ID: {market_id}")
                        
                        # Start with a copy of the base row data
                        row_data = base_row_data.copy()
                        row_data['market_id'] = market_id

                        # Use pandas.json_normalize to flatten the nested data
                        moneyline_data = pd.json_normalize(market['event'].get('moneyline', []))
                        total_data = pd.json_normalize(market['event'].get('total', []))
                        spread_data = pd.json_normalize(market['event'].get('spread', []))

                        # Populate common data points from moneyline DataFrame
                        if not moneyline_data.empty:
                            row_data['book_id'] = get_value_from_df(moneyline_data, 'home', 'book_id')
                            row_data['event_id'] = get_value_from_df(moneyline_data, 'home', 'event_id')
                            
                            # Moneyline data
                            row_data['home_money_line'] = get_value_from_df(moneyline_data, 'home', 'odds')
                            row_data['home_ml_ticket_pct'] = get_value_from_df(moneyline_data, 'home', 'bet_info.tickets.percent')
                            row_data['home_ml_money_pct'] = get_value_from_df(moneyline_data, 'home', 'bet_info.money.percent')
                            row_data['away_money_line'] = get_value_from_df(moneyline_data, 'away', 'odds')
                            row_data['away_ml_ticket_pct'] = get_value_from_df(moneyline_data, 'away', 'bet_info.tickets.percent')
                            row_data['away_ml_money_pct'] = get_value_from_df(moneyline_data, 'away', 'bet_info.money.percent')
                            row_data['tie_money_line'] = get_value_from_df(moneyline_data, 'draw', 'odds')
                            row_data['tie_ml_ticket_pct'] = get_value_from_df(moneyline_data, 'draw', 'bet_info.tickets.percent')
                            row_data['tie_ml_money_pct'] = get_value_from_df(moneyline_data, 'draw', 'bet_info.money.percent')

                        # Total data
                        if not total_data.empty:
                            row_data['total_score'] = get_value_from_df(total_data, 'over', 'value')
                            row_data['over_odds'] = get_value_from_df(total_data, 'over', 'odds')
                            row_data['under_odds'] = get_value_from_df(total_data, 'under', 'odds')
                            row_data['over_ticket_pct'] = get_value_from_df(total_data, 'over', 'bet_info.tickets.percent')
                            row_data['over_money_pct'] = get_value_from_df(total_data, 'over', 'bet_info.money.percent')
                            row_data['under_ticket_pct'] = get_value_from_df(total_data, 'under', 'bet_info.tickets.percent')
                            row_data['under_money_pct'] = get_value_from_df(total_data, 'under', 'bet_info.money.percent')

                        # Spread data
                        if not spread_data.empty:
                            row_data['home_spread'] = get_value_from_df(spread_data, 'home', 'value')
                            row_data['home_spread_odds'] = get_value_from_df(spread_data, 'home', 'odds')
                            row_data['home_spread_ticket_pct'] = get_value_from_df(spread_data, 'home', 'bet_info.tickets.percent')
                            row_data['home_spread_money_pct'] = get_value_from_df(spread_data, 'home', 'bet_info.money.percent')
                            row_data['away_spread'] = get_value_from_df(spread_data, 'away', 'value')
                            row_data['away_spread_odds'] = get_value_from_df(spread_data, 'away', 'odds')
                            row_data['away_spread_ticket_pct'] = get_value_from_df(spread_data, 'away', 'bet_info.tickets.percent')
                            row_data['away_spread_money_pct'] = get_value_from_df(spread_data, 'away', 'bet_info.money.percent')

                        # Append the collected data to the list
                        all_market_data.append(row_data)

                    else:
                        print(f"Game ID: {game_id}, Market ID: {market_id} - No moneyline data available. Skipping...")
                
                # If no valid markets were found for the game, still add the base row
                if not market_found:
                    print(f"Game ID: {game_id} - No valid markets found. Appending game data without market info.")
                    all_market_data.append(base_row_data)
            else:
                print(f"Game ID: {game_id} - No 'markets' key found. Appending game data without market info.")
                all_market_data.append(base_row_data)

        # Create the final pandas DataFrame
        df = pd.DataFrame(all_market_data)
        df['start_time_pt'] = pd.to_datetime(df['start_time']).dt.tz_convert('America/Los_Angeles')
        
        return df

    except requests.exceptions.RequestException as e:
        # Handle network or HTTP errors
        print(f"An error occurred: {e}")
        return None
    except Exception as e:
        # Handle any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        print(e)
        return None



def evaluate_bets(df_picks, df_result):
    df = df_picks.merge(df_result, on=["game_id", "start_time"], how="inner", suffixes=("", "_res"))


    # Payout function for both American and Decimal odds
    def calculate_payout(units, odds):
        """Calculates the payout based on units and American odds."""
        if odds > 0:
            return units * (odds / 100.0)
        else:
            return units * (100.0 / abs(odds))


    status_complete = df['status'] == 'complete'


    df['home_score'] = pd.to_numeric(df['home_score'])
    df['away_score'] = pd.to_numeric(df['away_score'])
    df['home_spread'] = pd.to_numeric(df['home_spread'])
    df['away_spread'] = pd.to_numeric(df['away_spread'])  

    # Conditions for bet_result
    conditions_result = [
        # Moneyline bets
        (status_complete) & (df['bet_home_ml'] == 1) & (df['home_score'] > df['away_score']),
        (status_complete) & (df['bet_away_ml'] == 1) & (df['away_score'] > df['home_score']),
        # (status_complete) & ((df['bet_home_ml'] == 1) | (df['bet_away_ml'] == 1)) & (df['home_score'] == df['away_score']),

        # Spread bets
        (status_complete) & (df['bet_home_spread'] == 1) & (df['home_score'] + df['home_spread'] > df['away_score']),
        (status_complete) & (df['bet_away_spread'] == 1) & (df['away_score'] + df['away_spread'] > df['home_score']),
        (status_complete) & (df['bet_home_spread'] == 1) & (df['home_score'] + df['home_spread'] == df['away_score']),
        (status_complete) & (df['bet_away_spread'] == 1) & (df['away_score'] + df['away_spread'] == df['home_score']),

        # Over/Under bets
        (status_complete) & (df['bet_over'] == 1) & (df['home_score'] + df['away_score'] > df['total_score']),
        (status_complete) & (df['bet_under'] == 1) & (df['home_score'] + df['away_score'] < df['total_score']),
        (status_complete) & (df['bet_over'] == 1) & (df['home_score'] + df['away_score'] == df['total_score']),
        (status_complete) & (df['bet_under'] == 1) & (df['home_score'] + df['away_score'] == df['total_score'])
    ]

    choices_result = [
        'win', 'win', 
        'win', 'win', 'push', 'push',
        'win', 'win', 'push', 'push'
    ]

    

    # Apply the conditions to get the bet results
    df['bet_result'] = np.select(conditions_result, choices_result, default='loss')

    # Conditions for bet_payout based on the result
    conditions_payout = [
        df['bet_result'] == 'win',
        df['bet_result'] == 'push',
        df['bet_result'] == 'loss'
    ]

    choices_payout = [
        df.apply(lambda row: calculate_payout(row['units'], row['odds']), axis=1),
        0.0,
        -df['units']
    ]

    # Apply the conditions to calculate the payout
    df['bet_payout'] = np.select(conditions_payout, choices_payout, default=0.0)


    ## create a date field based on the start_time in eastern time zone
    try:
        df['date'] = pd.to_datetime(df['start_time']).dt.tz_convert('America/New_York').dt.date
    except:
        df['date'] = pd.to_datetime(df['start_time'], utc=True).dt.tz_convert('America/New_York').dt.date

    return df[[
    "rank","model","date","game_id","match","home_score","away_score","pick","odds","units","bet_result","bet_payout"
    ]]






def fetch_all_games_data(sport, dates_or_weeks, HEADERS):
    """
    Fetches and concatenates all game data for a given sport.

    Args:
        sport (str): The sport to fetch data for (e.g., 'soccer', 'mlb', 'nfl').
        dates_or_weeks (list): A list of date strings (for soccer, mlb) or week numbers (for nfl, ncaaf).
        HEADERS (dict): The HTTP headers to use for the API request.

    Returns:
        pd.DataFrame: A DataFrame containing all fetched game data.
    """
    # Map sports to their base API URLs and parameter names
    API_MAPPING = {
        'soccer': {'base_url': 'https://api.actionnetwork.com/web/v2/scoreboard/soccer', 'param': 'date'},
        'mlb': {'base_url': 'https://api.actionnetwork.com/web/v2/scoreboard/mlb', 'param': 'date'},
        'ncaaf': {'base_url': 'https://api.actionnetwork.com/web/v2/scoreboard/ncaaf', 'param': 'week', 'division': 'FBS', 'seasonType': 'reg'},
        'nfl': {'base_url': 'https://api.actionnetwork.com/web/v2/scoreboard/nfl', 'param': 'week'},
        'nba': {'base_url': 'https://api.actionnetwork.com/web/v2/scoreboard/nba', 'param': 'date'},
        'ncaab': {'base_url': 'https://api.actionnetwork.com/web/v2/scoreboard/ncaab', 'param': 'date'}
    }

    if sport not in API_MAPPING:
        print(f"Error: Sport '{sport}' not supported.")
        return pd.DataFrame()

    sport_info = API_MAPPING[sport]
    base_url = sport_info['base_url']
    param_name = sport_info['param']

    all_games_df = pd.DataFrame()

    for value in dates_or_weeks:
        print(f"Processing data for {sport} ({param_name}: {value})")

        # Construct the API URL dynamically
        API_URL = f"{base_url}?bookIds=15,30,79,2988,75,123,71,68,69&periods=event&{param_name}={value}"

        ## if sport = ncaab, append &division=D1' to the API_URL
        if sport == 'ncaab':
             API_URL += '&division=D1'
        
        # Add additional parameters for specific sports (e.g., ncaaf)
        if 'division' in sport_info:
            API_URL += f"&division={sport_info['division']}"
        if 'seasonType' in sport_info:
            API_URL += f"&seasonType={sport_info['seasonType']}"

        # Assuming fetch_and_process_data_global() handles the API call and returns a DataFrame
        try:
            date_df = fetch_and_process_data(API_URL, HEADERS)
            all_games_df = pd.concat([all_games_df, date_df], ignore_index=True)
            print(f"Processed data for {sport} ({param_name}: {value})")
            print('-------')
        except Exception as e:
            print(f"Error fetching data for {sport} ({param_name}: {value}): {e}")
            continue

    return all_games_df




def get_complete_game_results(sport, dates_or_weeks, HEADERS):
    """
    Fetches and filters data to get complete game results for a given sport.
    """
    all_games_df = fetch_all_games_data(sport, dates_or_weeks, HEADERS)

    print(' getting complete game results banana')
    
    results_df = all_games_df.loc[all_games_df['status'] == 'complete']
    
    # Check if the required columns exist before trying to subset
    required_cols = ['game_id', 'start_time', 'league_name', 'home_team', 'away_team', 'home_score', 'away_score', 'status']
    available_cols = [col for col in required_cols if col in results_df.columns]
    
    results_df = results_df[available_cols].drop_duplicates()
    
    return results_df

def get_todays_games(sport, dates_or_weeks, HEADERS):
    """
    Fetches and filters data for today's games for a given sport based on market_id.
    """
    all_games_df = fetch_all_games_data(sport, dates_or_weeks, HEADERS)
    
    # Check if 'market_id' column exists before filtering
    if 'market_id' in all_games_df.columns:
        todays_games_df = all_games_df.loc[all_games_df['market_id'] == '15']
    else:
        print(f"Warning: 'market_id' not found for {sport}. Returning all games.")
        todays_games_df = all_games_df
        
    return todays_games_df



# def build_prompts(df, sport):
#     """
#     Constructs and saves prompt files for a list of models for a specific sport.

#     Args:
#         df (pd.DataFrame): DataFrame containing today's game data.
#         sport (str): The sport abbreviation (e.g., 'soccer', 'nfl').
#     """
#     # Dictionary to map sport abbreviations to file prefixes and full names

#     if sport not in SPORT_INFO:
#         print(f"Error: Sport '{sport}' not supported.")
#         return

#     if df.empty:
#         print(f"Error: The input DataFrame 'df' for today's games in {sport} is empty. Cannot generate prompts.")
#         return

#     sport_info = SPORT_INFO[sport]
#     file_prefix = sport_info['prefix']
#     sport_name = sport_info['full_name']

#     # input_todays_games=''
#     # input_historical_performance=''

#     prompt = """
#     I am providing you with a dataset of upcoming {sport_name} games and betting odds. 
#     Your task is to analyze the odds and recommend the top 10 picks to maximize return on investment. 
#     I am also providing you with the results of previous games so that you can evaluate your performance and adjust accordingly. 
#     It is important to note that I am evaluating your performance against a number of different other services in order to determine which model I will pay for a premium membership. 

#     **Instructions:**

#     1. **Validate every pick against the dataset**:
#     - Every Moneyline, spread, and total must match exactly what is in the dataset. No approximations or “best guesses.”
#     - Do not invent odds; use only the numbers provided.

#     2. **Pick Types**:
#     - You can pick:
#         - Home team ML (moneyline)
#         - Away team ML
#         - Home team spread
#         - Away team spread
#         - Over
#         - Under
#     - Include a **binary 1/0 for bet type**:
#         - ML = 1 if moneyline, 0 otherwise
#         - Spread = 1 if spread, 0 otherwise
#         - Over/Under = 1 if total bet, 0 otherwise

#     3. **Confidence & Units**:
#     - Rank picks by confidence.
#     - Provide **confidence %**.
#     - The confidence should be an integer between 0 and 100
#     - Assign units (1, 2, 3) based on your confidence.

#     4. **Predicted Score**:
#     - Include a predicted score for each match.
#     - The prediceted score should be in the format "HomeScore-AwayScore" (e.g., "3-2"). BE SURE TO HAVE THE HOME SCORE LISTED FIRST!
#     - Double check that the Home Score is listed first and the Away Score is second!

#     5. **Output**:
#     - **Human-readable table** first:
#         - Columns: Rank, Match, Pick, Odds, Units, Confidence %, Reason, Predicted Score
#     - **CSV block** second (ready to copy/paste):
#         - Exact structure:  
#         rank,game_id,start_time,match,pick,odds,units,confidence_pct,reason,predicted_score,bet_home_spread,bet_home_ml,bet_away_spread,bet_away_ml,bet_over,bet_under,home_money_line,away_money_line,tie_money_line,total_score,over_odds,under_odds,home_spread,home_spread_odds,away_spread,away_spread_odds
#         - Strings with commas/spaces must be double-quoted.
#         - All original betting odds fields must be included exactly as in the dataset.

#     6. **Top 10 Picks Only**:
#     - Output **exactly 10 picks** based on your analysis.

#     7. **Reasoning**:
#     - Include a brief explanation of why each pick was made.

#     **Dataset - Today's Games:** 
    
#     {input_todays_games}

#     **Dataset - Historical Performance:** 
    
#     {input_historical_performance}

#     **Deliverables:**
#     1. Table of top 10 picks in order of confidence.  
#     2. CSV block ready to use in a DataFrame, matching the exact column format above.  
#     3. All odds must match the dataset exactly.
#     """
#     model_list = ['Charlie', 'Cliff', 'David', 'Gary', 'Grover']

#     for model_name in model_list:
#         file_name = f'{file_prefix}_bet_picks_evaluated_{model_name}.csv'
#         prompt_file_name = f'{file_prefix}_bet_picks_prompt_{model_name}.txt'

#         print(file_name)

#         try:
#             with open(file_name, 'r') as f:
#                 historical_data = f.read()
#         except FileNotFoundError:
#             print(f"Warning: Historical data file not found for {model_name}. Skipping...")
#             continue
        
#         todays_games_csv = df.to_csv(index=False, header=True, sep=',')
#         print(todays_games_csv)
#         print(historical_data)
#         prompt_text = prompt.replace('{input_historical_performance}', historical_data).replace('{input_todays_games}', todays_games_csv).replace('{sport_name}', sport_name)

#         with open(prompt_file_name, 'w') as f:
#             f.write(prompt_text)

#         print(f"Generated prompt for {model_name} in {prompt_file_name}")
#         print('--------------------------------')






def build_prompts(df, sport):
    print('building prompts !!!!!')
    """
    Constructs and saves prompt files for a list of models for a specific sport.

    Args:
        df (pd.DataFrame): DataFrame containing today's game data.
        sport (str): The sport abbreviation (e.g., 'ncaaf').
    """
    if sport not in SPORT_INFO:
        print(f"Error: Sport '{sport}' not supported.")
        return

    # Check for empty input DataFrame first
    if df.empty:
        print(f"Error: The input DataFrame 'df' for today's games in {sport} is empty. Cannot generate prompts.")
        return

    sport_info = SPORT_INFO[sport]
    file_prefix = sport_info['prefix']
    sport_name = sport_info['full_name']

    # Removed the 'f' from the beginning of the string, and now including sport_name replacement
    prompt = """
    I am providing you with a dataset of upcoming {sport_name} games and betting odds. 
    Your task is to analyze the odds and recommend the top 10 picks to maximize return on investment. 
    I am also providing you with the results of previous games so that you can evaluate your performance and adjust accordingly. 
    It is important to note that I am evaluating your performance against a number of different other services in order to determine which model I will pay for a premium membership. 

    **Instructions:**

    1. **Validate every pick against the dataset**:
    - Every Moneyline, spread, and total must match exactly what is in the dataset. No approximations or “best guesses.”
    - Do not invent odds; use only the numbers provided.

    2. **Pick Types**:
    - You can pick:
        - Home team ML (moneyline)
        - Away team ML
        - Home team spread
        - Away team spread
        - Over
        - Under
    - Include a **binary 1/0 for bet type**:
        - ML = 1 if moneyline, 0 otherwise
        - Spread = 1 if spread, 0 otherwise
        - Over/Under = 1 if total bet, 0 otherwise

    3. **Confidence & Units**:
    - Rank picks by confidence.
    - Provide **confidence %**.
    - The confidence should be an integer between 0 and 100
    - Assign units (1, 2, 3) based on your confidence.

    4. **Predicted Score**:
    - Include a predicted score for each match.
    - The prediceted score should be in the format "HomeScore-AwayScore" (e.g., "3-2"). BE SURE TO HAVE THE HOME SCORE LISTED FIRST!
    - Double check that the Home Score is listed first and the Away Score is second!

    5. **Output**:
    - **Human-readable table** first:
        - Columns: Rank, Match, Pick, Odds, Units, Confidence %, Reason, Predicted Score
    - **CSV block** second (ready to copy/paste):
        - Exact structure:  
        rank,game_id,start_time,match,pick,odds,units,confidence_pct,reason,predicted_score,bet_home_spread,bet_home_ml,bet_away_spread,bet_away_ml,bet_over,bet_under,home_money_line,away_money_line,tie_money_line,total_score,over_odds,under_odds,home_spread,home_spread_odds,away_spread,away_spread_odds
        - Strings with commas/spaces must be double-quoted.
        - All original betting odds fields must be included exactly as in the dataset.

    6. **Top 10 Picks Only**:
    - Output **exactly 10 picks** based on your analysis.

    7. **Reasoning**:
    - Include a brief explanation of why each pick was made.

    **Dataset - Today's Games:** {input_todays_games}

    **Dataset - Historical Performance:** {input_historical_performance}

    **Deliverables:**
    1. Table of top 10 picks in order of confidence.  
    2. CSV block ready to use in a DataFrame, matching the exact column format above.  
    3. All odds must match the dataset exactly.
    """
    
    # Replace the sport name in the prompt string ONCE outside the loop for efficiency
    prompt = prompt.replace('{sport_name}', sport_name)

    todays_games_csv = df.loc[df['status']=='scheduled'].to_csv(index=False, header=True, sep=',')
    print(f"Generated CSV for today's games ({df.shape[0]} rows, {len(todays_games_csv)} chars)")
    
    # New logging to confirm the content of the today's games data
    if df.shape[0] > 0:
        print(f"Today's Games Sample (first data row): {todays_games_csv.splitlines()[1]}")
    else:
        print("Warning: Today's Games CSV contains only headers or is empty.")


    model_list = ['Charlie', 'Cliff', 'David', 'Gary', 'Grover']

    for model_name in model_list:
        file_name = f'{file_prefix}_bet_picks_evaluated_{model_name}.csv'
        prompt_file_name = f'{file_prefix}_bet_picks_prompt_{model_name}.txt'

        print(f"\nProcessing model: {model_name}")

        historical_data = ""
        try:
            with open(file_name, 'r') as f:
                historical_data = f.read()
            
            if not historical_data.strip():
                 print(f"Warning: Historical data file '{file_name}' was found but is empty.")
                 # Provide a clear message for the prompt if the file is empty
                 historical_data = "--- NO HISTORICAL DATA FOUND FOR THIS MODEL ---"
            else:
                 print(f"Historical data successfully loaded ({len(historical_data)} chars).")
                 # Print first line of historical data for verification
                 print(f"Historical Data Sample (first line): {historical_data.splitlines()[0]}") 

        except FileNotFoundError:
            print(f"Warning: Historical data file not found for {model_name}. Skipping...")
            continue
        
        # The placeholders are now replaced correctly.
        prompt_text = prompt.replace('{input_historical_performance}', historical_data).replace('{input_todays_games}', todays_games_csv)

        with open(prompt_file_name, 'w') as f:
            f.write(prompt_text)

        print(f"Generated prompt for {model_name} in {prompt_file_name}")
        print('--------------------------------')







def load_consolidated_picks(path: str) -> pd.DataFrame:
    """
    Reads a text file with multiple model sections (e.g. Charlie, Cliff, etc.)
    and returns a single pandas DataFrame with an added 'model' column.
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]

    sections = []
    current_model = None
    buffer = []

    for line in lines:
        # A model heading is a single word line (e.g., Charlie)
        if line.strip() and ',' not in line:
            # flush any buffered CSV lines from the previous model
            if buffer and current_model:
                df = pd.read_csv(pd.io.common.StringIO('\n'.join(buffer)))
                df['model'] = current_model
                sections.append(df)
                buffer = []
            current_model = line.strip()
        else:
            # part of CSV data
            if line.strip():
                buffer.append(line)

    # flush the last model's data
    if buffer and current_model:
        df = pd.read_csv(pd.io.common.StringIO('\n'.join(buffer)))
        df['model'] = current_model
        sections.append(df)

    # Combine all into one DataFrame
    return pd.concat(sections, ignore_index=True)

def build_consolidator_prompts(df_picks: pd.DataFrame, sport_name: str = "Soccer"):
    """
    Generates a consolidated betting prompt based on picks from different models.

    Args:
        df_picks (pd.DataFrame): A DataFrame containing all the picks from the models.
        sport_name (str): The name of the sport for which to generate the prompt.
    """

    # Dictionary to map different sport names to a generic term for the prompt.
    # This allows for easy generalization of the prompt text.


    # Normalize the input sport name to match the dictionary keys.
    normalized_sport_name = sport_name.lower()
    
    # Get the generic sport name from the mapping, default to "sports betting" if not found.
    sport_data = SPORT_INFO.get(normalized_sport_name, {'prefix': 'sports', 'full_name': 'sports betting'})
    generic_sport_full_name = sport_data['full_name']
    generic_sport_prefix = sport_data['prefix']

    df_picks_csv=''

    consolidator_prompt = """
    Ok -- I want you to be my {generic_sport_full_name} consolidator and results dashboard

    I am going to provide you with the outputs from different models

    I want you to consolidate everything into one table for easy analysis and viewing

    Here is an example of what the consolidated table will look like: please include the rank and the number of units for each pick, ensure that we always have the predicted score for each model and it should sit as the first line after the games header:

    BE SURE TO INCLUDE EVERY BET THAT A MODEL HAS PREDICTED IN THIS TABLE every single bet from every model, even if only one model picked it, is included

    Match   Bet Option  Charlie (M1)    Cliff (M2)  Gerald (M3)
    Bayern vs Leipzig   
        Predicted Score 3-2 4-2 3-1
        Bayern ML   –   ✅ (10th pick)   –
        Bayern -1.5 Spread  ✅ (3rd, 4u) –   ✅ (2nd, 8/10)
        Leipzig +1.5 Spread ✅ (10th, 1u)    ✅ (7th) ✅ (9th, 3/10)
        Over 2.5    ❌ (too juiced)  ❌   ✅ (5th, 5/10)
        Over 3.5 (alt)  ✅ (8th, 2u) –   –
    Chelsea vs Liverpool    
        Predicted Score 3-2 4-2 3-1
        Chelsea ML  –   ✅ (10th pick)   –
        Chelsea -1.5 Spread ✅ (3rd, 4u) –   ✅ (2nd, 8/10)


    Based on the models picks and their confidence levels, I want you to create 5 consolidated best bets and provide them as well
    Please display them with a reason why
    Also, generate a csv output of the best bets that I can include for further analysis
    If the output includes commas or special characters, please wrap the string in quotations so it can be read as a CSV

    The format of the CSV should be like this:
    rank,game_id,start_time,match,pick,odds,units,confidence_pct,reason,predicted_score,bet_home_spread,bet_home_ml,bet_away_spread,bet_away_ml,bet_over,bet_under,home_money_line, away_money_line,tie_money_line,total_score,over_odds,under_odds,home_spread,home_spread_odds,away_spread,away_spread_odds


    Ok, below is a table with all of the picks generate from other models:

    {df_picks_csv}

    """

    consolidator_prompt_filled = consolidator_prompt.replace('{df_picks_csv}', df_picks.to_csv(index=False, header=True, sep=',')).replace('{generic_sport_full_name}', generic_sport_full_name)

    # Create a unique filename based on the sport prefix
    filename = f"{generic_sport_prefix}_bet_picks_consolidator_prompt.txt"

    with open(filename, 'w') as f:
        f.write(consolidator_prompt_filled)

    print(f"Prompt successfully written to '{filename}'")


def process_and_save_evaluated_bets(df_picks, df_result, sport_name):
    """
    Processes, evaluates, and saves betting data for a given sport.
    
    This function evaluates picks, handles duplicates, and saves the results to
    sport-specific history files.
    """
    
    normalized_sport_name = sport_name.lower()
    sport_data = SPORT_INFO.get(normalized_sport_name, {'prefix': 'sports'})
    generic_sport_prefix = sport_data['prefix']

    df_evaluated = evaluate_bets(df_picks, df_result)
    
    games_left_to_play = pd.merge(df_picks[['game_id', 'match']], df_result[['game_id', 'status']], on='game_id', how='left').sort_values(['game_id']).query('status!="complete"').drop_duplicates()

    print('Games left to play:')
    # display(games_left_to_play)

    print('Current evaluation summary:')
    # display(df_evaluated.groupby('model').agg(
    #     bet_payout=('bet_payout', 'sum'),
    #     units=('units', 'sum'),
    #     bets=('game_id', 'count')
    # ).assign(ROI=lambda x: x['bet_payout'] / x['units'] * 100).sort_values('bet_payout', ascending=False).reset_index())

    # Dynamically create filenames based on the sport
    picks_hist_file = f"./data/evaluated/{generic_sport_prefix}_bet_picks.csv"
    evaluated_hist_file = f"./data/evaluated/{generic_sport_prefix}_bet_picks_evaluated.csv"

    try:
        df_picks_hist = pd.read_csv(picks_hist_file)
        df_evaluated_hist = pd.read_csv(evaluated_hist_file)
    except FileNotFoundError:
        df_picks_hist = pd.DataFrame()
        df_evaluated_hist = pd.DataFrame()

    df_picks_hist = pd.concat([df_picks_hist, df_picks], ignore_index=True)
    df_evaluated_hist = pd.concat([df_evaluated_hist, df_evaluated], ignore_index=True)

    df_evaluated_hist = df_evaluated_hist.drop_duplicates(subset=['rank', 'game_id', 'model', 'pick', 'bet_result'])
    df_picks_hist = df_picks_hist.drop_duplicates(subset=['rank', 'game_id', 'model', 'pick'])

    df_picks_hist.to_csv(picks_hist_file, index=False)
    df_evaluated_hist.to_csv(evaluated_hist_file, index=False)

    total_bet_payout = df_evaluated_hist['bet_payout'].sum()
    total_units = df_evaluated_hist['units'].sum()
    total_bets = df_evaluated_hist['game_id'].count()

    roi = (total_bet_payout / total_units) * 100

    print(f"Total Bet Payout: {total_bet_payout}")
    print(f"Total Units: {total_units}")
    print(f"Total Bets: {total_bets}")
    print(f"Return on Investment (ROI): {roi:.2f}%")

    print('Historical evaluation summary:')
    try:
        display(df_evaluated_hist.groupby('model').agg(
            bet_payout=('bet_payout', 'sum'),
            units=('units', 'sum'),
            bets=('game_id', 'count')
        ).assign(ROI=lambda x: x['bet_payout'] / x['units'] * 100).sort_values('bet_payout', ascending=False).reset_index())
    except:
        pass

    return df_evaluated, df_evaluated_hist


def generate_evaluated_hist_data(df_evaluated_hist, sport):
    if sport not in SPORT_INFO:
        print(f"Error: Sport '{sport}' not supported.")
        return

    sport_info = SPORT_INFO[sport]
    file_prefix = sport_info['prefix']

    for model in df_evaluated_hist['model'].unique():

        df_model = df_evaluated_hist.loc[df_evaluated_hist['model']==model][['date','rank','game_id','match','home_score','away_score','pick','odds','units','bet_result','bet_payout']]
        print(f"Model: {model}")
        df_model.to_csv(f'{file_prefix}_bet_picks_evaluated_{model}.csv',index=False)
