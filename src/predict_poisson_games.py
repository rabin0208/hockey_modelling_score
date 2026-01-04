"""
Predict over/under probabilities for upcoming NHL games using Poisson regression.

This script:
1. Loads the trained Poisson regression model
2. Fetches upcoming games from the NHL API
3. Creates features for those games
4. Predicts total goals and calculates over/under probabilities
5. Displays results with odds for over 5.5 and over 6.5
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from nhlpy import NHLClient
from scipy.stats import poisson

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from create_poisson_features import (
    load_games,
    extract_game_info,
    calculate_team_goal_stats_up_to_date,
    calculate_recent_goal_form,
    calculate_head_to_head_goals,
    calculate_nst_stats_up_to_date,
    match_team_name_to_nst
)


def load_model(model_path, features_path):
    """Load the saved Poisson regression model and feature list."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Loading feature list from {features_path}...")
    with open(features_path, 'rb') as f:
        feature_columns = pickle.load(f)
    
    print("✓ Model and features loaded")
    return model, feature_columns


def fetch_upcoming_games(client, days_ahead=7):
    """Fetch upcoming games from NHL API."""
    print(f"\nFetching upcoming games (next {days_ahead} days)...")
    
    today = datetime.now().date()
    end_date = today + timedelta(days=days_ahead)
    
    all_games = []
    current_date = today
    
    while current_date <= end_date:
        try:
            date_str = current_date.strftime('%Y-%m-%d')
            schedule = client.schedule.daily_schedule(date=date_str)
            
            # Handle different response formats
            if isinstance(schedule, dict):
                if 'games' in schedule:
                    games = schedule['games']
                elif 'dates' in schedule and len(schedule['dates']) > 0:
                    games = schedule['dates'][0].get('games', [])
                else:
                    games = []
            elif isinstance(schedule, list):
                games = schedule
            else:
                games = []
            
            if len(games) > 0:
                print(f"  {date_str}: Found {len(games)} games")
            
            for game in games:
                # Get game status
                game_status = game.get('status', {})
                if isinstance(game_status, dict):
                    abstract_state = game_status.get('abstractGameState', '')
                else:
                    abstract_state = game.get('abstractGameState', '')
                
                # Only include scheduled/preview games (not finished or live)
                if abstract_state not in ['Final', 'Live', 'Official']:
                    if 'gameDate' not in game or not game.get('gameDate'):
                        game['gameDate'] = date_str
                    all_games.append(game)
        except Exception as e:
            print(f"  Warning: Could not fetch games for {current_date}: {e}")
        
        current_date += timedelta(days=1)
    
    print(f"  Found {len(all_games)} upcoming games")
    return all_games


def parse_game_data(game):
    """Parse game data from NHL API format."""
    try:
        game_id = game.get('gamePk') or game.get('id')
        game_date = game.get('gameDate', '')
        
        # Parse date
        if isinstance(game_date, str):
            if 'T' in game_date:
                game_date = game_date.split('T')[0]
            game_date = pd.to_datetime(game_date).date()
        elif isinstance(game_date, (datetime, pd.Timestamp)):
            game_date = pd.to_datetime(game_date).date()
        
        # Get teams
        home_team = game.get('homeTeam', {})
        away_team = game.get('awayTeam', {})
        
        # Extract team info
        home_team_id = home_team.get('id')
        away_team_id = away_team.get('id')
        
        home_team_name = (
            home_team.get('placeName', {}).get('default', '') + ' ' + 
            home_team.get('commonName', {}).get('default', '')
        ).strip()
        
        away_team_name = (
            away_team.get('placeName', {}).get('default', '') + ' ' + 
            away_team.get('commonName', {}).get('default', '')
        ).strip()
        
        if not home_team_id or not away_team_id:
            return None
        
        return {
            'game_id': game_id,
            'date': game_date,
            'home_team_id': home_team_id,
            'home_team_name': home_team_name,
            'away_team_id': away_team_id,
            'away_team_name': away_team_name,
            'season': f"{game_date.year}_{game_date.year+1}" if game_date.month >= 10 else f"{game_date.year-1}_{game_date.year}"
        }
    except Exception as e:
        print(f"  Warning: Could not parse game {game.get('gamePk', 'unknown')}: {e}")
        return None


def create_features_for_game(games_df, game_data, feature_columns):
    """Create Poisson features for a single upcoming game."""
    game_date = game_data['date']
    home_team_id = game_data['home_team_id']
    away_team_id = game_data['away_team_id']
    home_team_name = game_data['home_team_name']
    away_team_name = game_data['away_team_name']
    
    # Ensure game_date is a date object
    if isinstance(game_date, str):
        game_date = pd.to_datetime(game_date).date()
    elif isinstance(game_date, pd.Timestamp):
        game_date = game_date.date()
    
    # Ensure games_df date column is date type
    if games_df['date'].dtype != 'object' or isinstance(games_df['date'].iloc[0], str):
        games_df = games_df.copy()
        games_df['date'] = pd.to_datetime(games_df['date']).dt.date
    
    # Convert team IDs to int
    home_team_id = int(home_team_id) if home_team_id is not None else None
    away_team_id = int(away_team_id) if away_team_id is not None else None
    
    if games_df['home_team_id'].dtype != 'int64':
        games_df = games_df.copy()
        games_df['home_team_id'] = games_df['home_team_id'].astype('Int64')
        games_df['away_team_id'] = games_df['away_team_id'].astype('Int64')
    
    # Get team goal stats
    home_stats = calculate_team_goal_stats_up_to_date(games_df, home_team_id, game_date)
    away_stats = calculate_team_goal_stats_up_to_date(games_df, away_team_id, game_date)
    
    if not home_stats or not away_stats:
        home_count = len(games_df[
            ((games_df['home_team_id'] == home_team_id) | (games_df['away_team_id'] == home_team_id)) &
            (games_df['date'] < game_date) &
            (games_df['total_goals'].notna())
        ])
        away_count = len(games_df[
            ((games_df['home_team_id'] == away_team_id) | (games_df['away_team_id'] == away_team_id)) &
            (games_df['date'] < game_date) &
            (games_df['total_goals'].notna())
        ])
        print(f"  ⚠ Skipping {away_team_name} @ {home_team_name} (home: {home_count} games, away: {away_count} games, need 5+)")
        return None
    
    # Get recent form
    home_recent = calculate_recent_goal_form(games_df, home_team_id, game_date)
    away_recent = calculate_recent_goal_form(games_df, away_team_id, game_date)
    
    # Get head-to-head statistics
    h2h_stats = calculate_head_to_head_goals(games_df, home_team_id, away_team_id, game_date)
    
    # Get Natural Stat Trick advanced stats
    home_nst_name = match_team_name_to_nst(home_team_name)
    away_nst_name = match_team_name_to_nst(away_team_name)
    home_nst = calculate_nst_stats_up_to_date(home_nst_name, game_date) if home_nst_name else None
    away_nst = calculate_nst_stats_up_to_date(away_nst_name, game_date) if away_nst_name else None
    
    # Create feature vector (same format as training)
    features = {
        'home_goals_for_avg': home_stats['goals_for_avg'],
        'home_goals_against_avg': home_stats['goals_against_avg'],
        'home_home_goals_for_avg': home_stats['home_goals_for_avg'],
        'home_home_goals_against_avg': home_stats['home_goals_against_avg'],
        'home_total_goals_per_game_avg': home_stats['total_goals_per_game_avg'],
        'away_goals_for_avg': away_stats['goals_for_avg'],
        'away_goals_against_avg': away_stats['goals_against_avg'],
        'away_away_goals_for_avg': away_stats['away_goals_for_avg'],
        'away_away_goals_against_avg': away_stats['away_goals_against_avg'],
        'away_total_goals_per_game_avg': away_stats['total_goals_per_game_avg'],
        'combined_goals_for_avg': (home_stats['goals_for_avg'] + away_stats['goals_for_avg']) / 2,
        'combined_goals_against_avg': (home_stats['goals_against_avg'] + away_stats['goals_against_avg']) / 2,
        'expected_total_goals_simple': home_stats['goals_for_avg'] + away_stats['goals_for_avg'],
        'expected_total_goals_defensive': home_stats['goals_against_avg'] + away_stats['goals_against_avg'],
        'expected_total_goals_combined': (home_stats['goals_for_avg'] + away_stats['goals_for_avg'] + 
                                         home_stats['goals_against_avg'] + away_stats['goals_against_avg']) / 2,
        'home_recent_goals_for_avg': home_recent['recent_goals_for_avg'] if home_recent else home_stats['goals_for_avg'],
        'home_recent_goals_against_avg': home_recent['recent_goals_against_avg'] if home_recent else home_stats['goals_against_avg'],
        'home_recent_total_goals_avg': home_recent['recent_total_goals_avg'] if home_recent else home_stats['total_goals_per_game_avg'],
        'away_recent_goals_for_avg': away_recent['recent_goals_for_avg'] if away_recent else away_stats['goals_for_avg'],
        'away_recent_goals_against_avg': away_recent['recent_goals_against_avg'] if away_recent else away_stats['goals_against_avg'],
        'away_recent_total_goals_avg': away_recent['recent_total_goals_avg'] if away_recent else away_stats['total_goals_per_game_avg'],
        'h2h_games': h2h_stats['h2h_games'] if h2h_stats else 0,
        'h2h_total_goals_avg': h2h_stats['h2h_total_goals_avg'] if h2h_stats else (home_stats['total_goals_per_game_avg'] + away_stats['total_goals_per_game_avg']) / 2,
        'h2h_recent_total_goals_avg': h2h_stats['h2h_recent_total_goals_avg'] if h2h_stats else None,
        'home_xgf_avg': home_nst['xgf_avg'] if home_nst and home_nst.get('xgf_avg') is not None else None,
        'home_xga_avg': home_nst['xga_avg'] if home_nst and home_nst.get('xga_avg') is not None else None,
        'home_xgf_pct_avg': home_nst['xgf_pct_avg'] if home_nst and home_nst.get('xgf_pct_avg') is not None else None,
        'away_xgf_avg': away_nst['xgf_avg'] if away_nst and away_nst.get('xgf_avg') is not None else None,
        'away_xga_avg': away_nst['xga_avg'] if away_nst and away_nst.get('xga_avg') is not None else None,
        'away_xgf_pct_avg': away_nst['xgf_pct_avg'] if away_nst and away_nst.get('xgf_pct_avg') is not None else None,
        'combined_xgf_avg': (home_nst['xgf_avg'] + away_nst['xgf_avg']) if (home_nst and away_nst and 
                                                                              home_nst.get('xgf_avg') is not None and 
                                                                              away_nst.get('xgf_avg') is not None) else None,
        'combined_xga_avg': (home_nst['xga_avg'] + away_nst['xga_avg']) if (home_nst and away_nst and 
                                                                              home_nst.get('xga_avg') is not None and 
                                                                              away_nst.get('xga_avg') is not None) else None,
    }
    
    # Create DataFrame with features in correct order
    features_df = pd.DataFrame([features])
    
    # Fill missing values with median (same strategy as training)
    for col in feature_columns:
        if col not in features_df.columns:
            features_df[col] = None
    
    # Reorder to match training data
    features_df = features_df[feature_columns]
    
    # Fill missing NST features with 0 (they're optional)
    # Use infer_objects to avoid FutureWarning
    features_df = features_df.fillna(0).infer_objects(copy=False)
    
    return features_df


def calculate_poisson_probability(lambda_pred, over_line):
    """Calculate probability of total goals being over a given line."""
    threshold = int(np.floor(over_line))
    prob_over = 1 - poisson.cdf(threshold, lambda_pred)
    return prob_over


def probability_to_american_odds(prob):
    """Convert probability to American odds format.
    
    American odds:
    - Negative (e.g., -150): Favorite - bet $150 to win $100
    - Positive (e.g., +150): Underdog - bet $100 to win $150
    
    Formula:
    - Favorite (prob >= 0.5): odds = -100 * (prob / (1 - prob))
    - Underdog (prob < 0.5): odds = +100 * ((1 - prob) / prob)
    """
    if prob <= 0 or prob >= 1:
        return None
    
    if prob >= 0.5:
        # Favorite: negative odds
        # Bet $X to win $100, where X = 100 * (prob / (1 - prob))
        odds = 100 * (prob / (1 - prob))
        return f"-{int(odds)}"
    else:
        # Underdog: positive odds
        # Bet $100 to win $X, where X = 100 * ((1 - prob) / prob)
        odds = 100 * ((1 - prob) / prob)
        return f"+{int(odds)}"


def predict_games(model, upcoming_games, games_df, feature_columns):
    """Make predictions for upcoming games."""
    print(f"\nCreating features and making predictions...")
    
    predictions = []
    
    for game in upcoming_games:
        # Parse game data
        game_data = parse_game_data(game)
        if not game_data:
            continue
        
        # Create features
        features_df = create_features_for_game(games_df, game_data, feature_columns)
        
        if features_df is None:
            continue
        
        # Predict lambda (mean total goals)
        lambda_pred = model.predict(features_df)[0]
        lambda_pred = max(lambda_pred, 0.1)  # Ensure positive
        
        # Calculate probabilities for over 5.5 and over 6.5
        prob_over_55 = calculate_poisson_probability(lambda_pred, 5.5)
        prob_over_65 = calculate_poisson_probability(lambda_pred, 6.5)
        
        # Calculate odds
        odds_over_55 = probability_to_american_odds(prob_over_55)
        odds_under_55 = probability_to_american_odds(1 - prob_over_55)
        odds_over_65 = probability_to_american_odds(prob_over_65)
        odds_under_65 = probability_to_american_odds(1 - prob_over_65)
        
        predictions.append({
            'date': game_data['date'],
            'away_team': game_data['away_team_name'],
            'home_team': game_data['home_team_name'],
            'predicted_total': lambda_pred,
            'prob_over_55': prob_over_55,
            'prob_under_55': 1 - prob_over_55,
            'odds_over_55': odds_over_55,
            'odds_under_55': odds_under_55,
            'prob_over_65': prob_over_65,
            'prob_under_65': 1 - prob_over_65,
            'odds_over_65': odds_over_65,
            'odds_under_65': odds_under_65,
        })
    
    return predictions


def display_predictions(predictions):
    """Display predictions in a readable format."""
    print("\n" + "=" * 80)
    print("UPCOMING GAME PREDICTIONS - OVER/UNDER")
    print("=" * 80)
    
    if not predictions:
        print("\nNo predictions available (insufficient historical data for upcoming games)")
        return
    
    # Sort by date
    predictions.sort(key=lambda x: x['date'])
    
    current_date = None
    for pred in predictions:
        # Print date header if new date
        if pred['date'] != current_date:
            current_date = pred['date']
            print(f"\n📅 {current_date.strftime('%A, %B %d, %Y')}")
            print("-" * 80)
        
        # Print prediction
        print(f"\n  {pred['away_team']} @ {pred['home_team']}")
        print(f"  Predicted Total Goals: {pred['predicted_total']:.2f}")
        print(f"\n  Over/Under 5.5:")
        print(f"    Over 5.5:  {pred['prob_over_55']:.1%} probability  |  Odds: {pred['odds_over_55']}")
        print(f"    Under 5.5: {pred['prob_under_55']:.1%} probability  |  Odds: {pred['odds_under_55']}")
        print(f"\n  Over/Under 6.5:")
        print(f"    Over 6.5:  {pred['prob_over_65']:.1%} probability  |  Odds: {pred['odds_over_65']}")
        print(f"    Under 6.5: {pred['prob_under_65']:.1%} probability  |  Odds: {pred['odds_under_65']}")
    
    print("\n" + "=" * 80)


def load_historical_games():
    """Load historical games for feature calculation."""
    print("\nLoading historical game data...")
    
    games = load_games('data')
    
    if not games:
        raise ValueError("No historical games found. Please run src/fetch_historical_seasons.py first.")
    
    print(f"  Loaded {len(games)} historical games")
    
    # Convert to DataFrame
    game_info = [extract_game_info(game) for game in games]
    game_info = [g for g in game_info if g is not None]
    
    games_df = pd.DataFrame(game_info)
    games_df['date'] = pd.to_datetime(games_df['date']).dt.date
    
    if len(games_df) == 0:
        raise ValueError("No valid games found in historical data.")
    
    return games_df


def main():
    """Main function to predict upcoming games."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict over/under for upcoming NHL games')
    parser.add_argument(
        '--model',
        type=str,
        default='models/poisson_regression_model.pkl',
        help='Path to model file (default: models/poisson_regression_model.pkl)'
    )
    parser.add_argument(
        '--features',
        type=str,
        default='models/poisson_model_features.pkl',
        help='Path to features file (default: models/poisson_model_features.pkl)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days ahead to fetch games (default: 7)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("NHL Over/Under Prediction - Poisson Regression")
    print("=" * 80)
    
    # Load model
    model, feature_columns = load_model(args.model, args.features)
    
    # Load historical games
    games_df = load_historical_games()
    
    # Initialize NHL client
    print("\nInitializing NHL API client...")
    client = NHLClient(debug=False)
    
    # Fetch upcoming games
    upcoming_games = fetch_upcoming_games(client, days_ahead=args.days)
    
    if not upcoming_games:
        print("\nNo upcoming games found in the specified time period.")
        return
    
    # Make predictions
    predictions = predict_games(model, upcoming_games, games_df, feature_columns)
    
    # Display results
    display_predictions(predictions)
    
    print("\n✓ Predictions complete!")


if __name__ == '__main__':
    main()

