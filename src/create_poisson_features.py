"""
Create features for Poisson regression model to predict over/unders.

This script processes historical games and creates features for each game
based on team offensive/defensive performance up to that point in the season.
Features are designed for predicting total goals (over/under).
"""

import os
import json
import pandas as pd
from datetime import datetime
import unicodedata

# Try to load NST data (optional)
NST_DATA = None
try:
    nst_file = os.path.join('data', 'nst_processed.csv')
    if os.path.exists(nst_file):
        NST_DATA = pd.read_csv(nst_file)
        NST_DATA['date'] = pd.to_datetime(NST_DATA['date']).dt.date
        print(f"✓ Loaded Natural Stat Trick data: {len(NST_DATA)} records")
except Exception as e:
    print(f"Note: Natural Stat Trick data not available: {e}")


def normalize_team_name(name):
    """Normalize team name for matching (remove accents, periods, etc.)."""
    if name is None:
        return None
    # Remove accents (Montréal -> Montreal)
    name = unicodedata.normalize('NFD', str(name)).encode('ascii', 'ignore').decode('ascii')
    # Remove periods (St. Louis -> St Louis)
    name = name.replace('.', '')
    # Lowercase and strip
    return name.lower().strip()


def load_games(data_dir):
    """Load all games from JSON files."""
    games = []
    
    # Try to load combined file first
    combined_file = os.path.join(data_dir, "games_all_seasons.json")
    if os.path.exists(combined_file):
        print(f"Loading games from {combined_file}...")
        with open(combined_file, 'r') as f:
            games = json.load(f)
        print(f"  Loaded {len(games)} games")
        return games
    
    # Otherwise load individual season files
    season_files = [
        "games_2025_2026.json",
        "games_2024_2025.json",
        "games_2023_2024.json",
        "games_2022_2023.json",
        "games_2021_2022.json"
    ]
    
    for filename in season_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"Loading games from {filename}...")
            with open(filepath, 'r') as f:
                season_games = json.load(f)
                games.extend(season_games)
                print(f"  Loaded {len(season_games)} games")
    
    return games


def extract_game_info(game):
    """Extract basic information from a game."""
    try:
        # Get game date
        game_date = game.get('gameDate') or game.get('startTimeUTC', '')[:10]
        
        # Get teams
        away_team = game.get('awayTeam', {})
        home_team = game.get('homeTeam', {})
        
        away_team_id = away_team.get('id')
        home_team_id = home_team.get('id')
        
        away_team_name = away_team.get('placeName', {}).get('default', '') + ' ' + away_team.get('commonName', {}).get('default', '')
        home_team_name = home_team.get('placeName', {}).get('default', '') + ' ' + home_team.get('commonName', {}).get('default', '')
        
        # Get scores
        away_score = away_team.get('score')
        home_score = home_team.get('score')
        
        # Calculate total goals
        total_goals = None
        if away_score is not None and home_score is not None:
            total_goals = home_score + away_score
        
        # Get game state
        game_state = game.get('gameState', '')
        
        return {
            'game_id': game.get('id'),
            'date': game_date,
            'season': game.get('season'),
            'away_team_id': away_team_id,
            'away_team_name': away_team_name.strip(),
            'home_team_id': home_team_id,
            'home_team_name': home_team_name.strip(),
            'away_score': away_score,
            'home_score': home_score,
            'total_goals': total_goals,
            'game_state': game_state
        }
    except Exception as e:
        print(f"Error extracting game info: {e}")
        return None


def calculate_team_goal_stats_up_to_date(games_df, team_id, date, min_games=5):
    """
    Calculate team goal-scoring statistics up to (but not including) a given date.
    Focus on offensive and defensive capabilities for goal prediction.
    
    Args:
        games_df: DataFrame with all games
        team_id: Team ID to calculate stats for
        date: Date to calculate stats up to (exclusive)
        min_games: Minimum games required to return stats
    
    Returns:
        Dictionary with team goal statistics
    """
    # Get all games for this team before the given date
    team_games = games_df[
        (games_df['date'] < date) &
        ((games_df['home_team_id'] == team_id) | (games_df['away_team_id'] == team_id)) &
        (games_df['total_goals'].notna())  # Only completed games
    ].copy()
    
    if len(team_games) < min_games:
        return None
    
    # Calculate goal statistics
    goals_for = 0
    goals_against = 0
    total_goals_in_games = 0  # Total goals in games this team played
    home_goals_for = 0
    home_goals_against = 0
    home_games = 0
    away_goals_for = 0
    away_goals_against = 0
    away_games = 0
    
    for _, game in team_games.iterrows():
        is_home = game['home_team_id'] == team_id
        
        if is_home:
            team_score = game['home_score']
            opp_score = game['away_score']
            home_games += 1
            home_goals_for += team_score if pd.notna(team_score) else 0
            home_goals_against += opp_score if pd.notna(opp_score) else 0
        else:
            team_score = game['away_score']
            opp_score = game['home_score']
            away_games += 1
            away_goals_for += team_score if pd.notna(team_score) else 0
            away_goals_against += opp_score if pd.notna(opp_score) else 0
        
        goals_for += team_score if pd.notna(team_score) else 0
        goals_against += opp_score if pd.notna(opp_score) else 0
        total_goals_in_games += game['total_goals'] if pd.notna(game['total_goals']) else 0
    
    total_games = len(team_games)
    
    if total_games == 0:
        return None
    
    stats = {
        'games_played': total_games,
        'goals_for_avg': goals_for / total_games if total_games > 0 else 0,
        'goals_against_avg': goals_against / total_games if total_games > 0 else 0,
        'total_goals_per_game_avg': total_goals_in_games / total_games if total_games > 0 else 0,
        'home_goals_for_avg': home_goals_for / home_games if home_games > 0 else 0,
        'home_goals_against_avg': home_goals_against / home_games if home_games > 0 else 0,
        'away_goals_for_avg': away_goals_for / away_games if away_games > 0 else 0,
        'away_goals_against_avg': away_goals_against / away_games if away_games > 0 else 0,
        'home_games': home_games,
        'away_games': away_games
    }
    
    return stats


def calculate_recent_goal_form(games_df, team_id, date, lookback_games=10):
    """
    Calculate recent goal-scoring form (last N games) for a team.
    
    Args:
        games_df: DataFrame with all games
        team_id: Team ID
        date: Date to calculate form up to
        lookback_games: Number of recent games to consider
    
    Returns:
        Dictionary with recent form statistics
    """
    # Get recent games
    team_games = games_df[
        (games_df['date'] < date) &
        ((games_df['home_team_id'] == team_id) | (games_df['away_team_id'] == team_id)) &
        (games_df['total_goals'].notna())
    ].sort_values('date', ascending=False).head(lookback_games)
    
    if len(team_games) == 0:
        return None
    
    goals_for = 0
    goals_against = 0
    total_goals = 0
    
    for _, game in team_games.iterrows():
        is_home = game['home_team_id'] == team_id
        
        if is_home:
            team_score = game['home_score']
            opp_score = game['away_score']
        else:
            team_score = game['away_score']
            opp_score = game['home_score']
        
        goals_for += team_score if pd.notna(team_score) else 0
        goals_against += opp_score if pd.notna(opp_score) else 0
        total_goals += game['total_goals'] if pd.notna(game['total_goals']) else 0
    
    return {
        'recent_games': len(team_games),
        'recent_goals_for_avg': goals_for / len(team_games) if len(team_games) > 0 else 0,
        'recent_goals_against_avg': goals_against / len(team_games) if len(team_games) > 0 else 0,
        'recent_total_goals_avg': total_goals / len(team_games) if len(team_games) > 0 else 0
    }


def calculate_nst_stats_up_to_date(team_name, date, min_games=5):
    """
    Calculate Natural Stat Trick advanced stats for a team up to a given date.
    
    Args:
        team_name: Team name (must match NST data)
        date: Date to calculate stats up to (exclusive)
        min_games: Minimum games required
    
    Returns:
        Dictionary with NST statistics or None
    """
    if NST_DATA is None or len(NST_DATA) == 0:
        return None
    
    # Ensure date is a date object for comparison
    if isinstance(date, pd.Timestamp):
        date = date.date()
    elif isinstance(date, str):
        date = pd.to_datetime(date).date()
    
    # Filter to this team's games before the date
    team_games = NST_DATA[
        (NST_DATA['team_name'] == team_name) &
        (NST_DATA['date'] < date)
    ].copy()
    
    if len(team_games) < min_games:
        return None
    
    # Calculate averages (handle None values)
    def safe_mean(series):
        valid = series.dropna()
        return valid.mean() if len(valid) > 0 else None
    
    stats = {
        'xgf_avg': safe_mean(team_games['xGF']) if 'xGF' in team_games.columns else None,
        'xga_avg': safe_mean(team_games['xGA']) if 'xGA' in team_games.columns else None,
        'xgf_pct_avg': safe_mean(team_games['xGF%']) if 'xGF%' in team_games.columns else None,
        'cf_pct_avg': safe_mean(team_games['CF%']) if 'CF%' in team_games.columns else None,
        'ff_pct_avg': safe_mean(team_games['FF%']) if 'FF%' in team_games.columns else None,
        'sf_pct_avg': safe_mean(team_games['SF%']) if 'SF%' in team_games.columns else None,
        'hdcf_pct_avg': safe_mean(team_games['HDCF%']) if 'HDCF%' in team_games.columns else None,
        'scf_pct_avg': safe_mean(team_games['SCF%']) if 'SCF%' in team_games.columns else None,
        'pdo_avg': safe_mean(team_games['PDO']) if 'PDO' in team_games.columns else None,
    }
    
    return stats


def match_team_name_to_nst(our_team_name):
    """
    Match our team name format to NST team name format.
    
    Args:
        our_team_name: Team name from our data
    
    Returns:
        NST team name or None
    """
    if NST_DATA is None:
        return None
    
    our_normalized = normalize_team_name(our_team_name)
    nst_teams = NST_DATA['team_name'].unique()
    
    # Try exact match (normalized)
    for nst_team in nst_teams:
        if normalize_team_name(nst_team) == our_normalized:
            return nst_team
    
    # Try partial match (normalized)
    for nst_team in nst_teams:
        nst_normalized = normalize_team_name(nst_team)
        # Check if key words match
        our_words = set(our_normalized.split())
        nst_words = set(nst_normalized.split())
        if our_words & nst_words:  # If there's any overlap
            return nst_team
    
    # Special cases for teams that don't exist in NST data
    if 'utah' in our_normalized:
        return None
    
    return None


def calculate_head_to_head_goals(games_df, home_team_id, away_team_id, date, lookback_games=10):
    """
    Calculate head-to-head goal statistics between two teams.
    
    Args:
        games_df: DataFrame with all games
        home_team_id: Home team ID
        away_team_id: Away team ID
        date: Date to calculate stats up to
        lookback_games: Number of recent H2H games to consider
    
    Returns:
        Dictionary with head-to-head goal statistics
    """
    # Get all games between these two teams before the given date
    h2h_games = games_df[
        (games_df['date'] < date) &
        (games_df['total_goals'].notna()) &  # Only completed games
        (
            ((games_df['home_team_id'] == home_team_id) & (games_df['away_team_id'] == away_team_id)) |
            ((games_df['home_team_id'] == away_team_id) & (games_df['away_team_id'] == home_team_id))
        )
    ].sort_values('date', ascending=False)
    
    if len(h2h_games) == 0:
        return None
    
    home_team_goals = 0
    away_team_goals = 0
    total_goals = 0
    
    for _, game in h2h_games.iterrows():
        is_home = game['home_team_id'] == home_team_id
        
        if is_home:
            home_team_score = game['home_score']
            away_team_score = game['away_score']
        else:
            home_team_score = game['away_score']  # home_team_id was away in this game
            away_team_score = game['home_score']
        
        home_team_goals += home_team_score if pd.notna(home_team_score) else 0
        away_team_goals += away_team_score if pd.notna(away_team_score) else 0
        total_goals += game['total_goals'] if pd.notna(game['total_goals']) else 0
    
    total_h2h_games = len(h2h_games)
    
    # Get recent H2H games (last N games)
    recent_h2h = h2h_games.head(lookback_games)
    recent_home_goals = 0
    recent_away_goals = 0
    recent_total_goals = 0
    
    for _, game in recent_h2h.iterrows():
        is_home = game['home_team_id'] == home_team_id
        if is_home:
            recent_home_goals += game['home_score'] if pd.notna(game['home_score']) else 0
            recent_away_goals += game['away_score'] if pd.notna(game['away_score']) else 0
        else:
            recent_home_goals += game['away_score'] if pd.notna(game['away_score']) else 0
            recent_away_goals += game['home_score'] if pd.notna(game['home_score']) else 0
        recent_total_goals += game['total_goals'] if pd.notna(game['total_goals']) else 0
    
    return {
        'h2h_games': total_h2h_games,
        'h2h_home_team_goals_avg': home_team_goals / total_h2h_games if total_h2h_games > 0 else 0,
        'h2h_away_team_goals_avg': away_team_goals / total_h2h_games if total_h2h_games > 0 else 0,
        'h2h_total_goals_avg': total_goals / total_h2h_games if total_h2h_games > 0 else 0,
        'h2h_recent_games': len(recent_h2h),
        'h2h_recent_total_goals_avg': recent_total_goals / len(recent_h2h) if len(recent_h2h) > 0 else 0
    }


def create_poisson_features(games_df):
    """
    Create features for each game based on team goal-scoring performance up to that date.
    
    Args:
        games_df: DataFrame with all games
    
    Returns:
        DataFrame with features for each game
    """
    print("\nCreating features for each game...")
    
    features_list = []
    total_games = len(games_df)
    
    for idx, game in games_df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing game {idx+1}/{total_games}...")
        
        # Skip games without scores (future games)
        if pd.isna(game['total_goals']):
            continue
        
        game_date = game['date']
        home_team_id = game['home_team_id']
        away_team_id = game['away_team_id']
        
        # Get team goal stats up to this date
        home_stats = calculate_team_goal_stats_up_to_date(games_df, home_team_id, game_date)
        away_stats = calculate_team_goal_stats_up_to_date(games_df, away_team_id, game_date)
        
        # Get recent form
        home_recent = calculate_recent_goal_form(games_df, home_team_id, game_date)
        away_recent = calculate_recent_goal_form(games_df, away_team_id, game_date)
        
        # Get head-to-head statistics
        h2h_stats = calculate_head_to_head_goals(games_df, home_team_id, away_team_id, game_date)
        
        # Get Natural Stat Trick advanced stats
        home_nst_name = match_team_name_to_nst(game['home_team_name'])
        away_nst_name = match_team_name_to_nst(game['away_team_name'])
        home_nst = calculate_nst_stats_up_to_date(home_nst_name, game_date) if home_nst_name else None
        away_nst = calculate_nst_stats_up_to_date(away_nst_name, game_date) if away_nst_name else None
        
        # Skip if we don't have enough data
        if not home_stats or not away_stats:
            continue
        
        # Create feature vector focused on goal prediction
        features = {
            'game_id': game['game_id'],
            'date': game_date,
            'season': game['season'],
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_team_name': game['home_team_name'],
            'away_team_name': game['away_team_name'],
            
            # Home team offensive features
            'home_goals_for_avg': home_stats['goals_for_avg'],
            'home_goals_against_avg': home_stats['goals_against_avg'],
            'home_home_goals_for_avg': home_stats['home_goals_for_avg'],
            'home_home_goals_against_avg': home_stats['home_goals_against_avg'],
            'home_total_goals_per_game_avg': home_stats['total_goals_per_game_avg'],
            
            # Away team offensive features
            'away_goals_for_avg': away_stats['goals_for_avg'],
            'away_goals_against_avg': away_stats['goals_against_avg'],
            'away_away_goals_for_avg': away_stats['away_goals_for_avg'],
            'away_away_goals_against_avg': away_stats['away_goals_against_avg'],
            'away_total_goals_per_game_avg': away_stats['total_goals_per_game_avg'],
            
            # Combined offensive/defensive features (key for total goals prediction)
            'combined_goals_for_avg': (home_stats['goals_for_avg'] + away_stats['goals_for_avg']) / 2,
            'combined_goals_against_avg': (home_stats['goals_against_avg'] + away_stats['goals_against_avg']) / 2,
            'expected_total_goals_simple': home_stats['goals_for_avg'] + away_stats['goals_for_avg'],
            'expected_total_goals_defensive': home_stats['goals_against_avg'] + away_stats['goals_against_avg'],
            'expected_total_goals_combined': (home_stats['goals_for_avg'] + away_stats['goals_for_avg'] + 
                                             home_stats['goals_against_avg'] + away_stats['goals_against_avg']) / 2,
            
            # Recent form features
            'home_recent_goals_for_avg': home_recent['recent_goals_for_avg'] if home_recent else home_stats['goals_for_avg'],
            'home_recent_goals_against_avg': home_recent['recent_goals_against_avg'] if home_recent else home_stats['goals_against_avg'],
            'home_recent_total_goals_avg': home_recent['recent_total_goals_avg'] if home_recent else home_stats['total_goals_per_game_avg'],
            'away_recent_goals_for_avg': away_recent['recent_goals_for_avg'] if away_recent else away_stats['goals_for_avg'],
            'away_recent_goals_against_avg': away_recent['recent_goals_against_avg'] if away_recent else away_stats['goals_against_avg'],
            'away_recent_total_goals_avg': away_recent['recent_total_goals_avg'] if away_recent else away_stats['total_goals_per_game_avg'],
            
            # Head-to-head features
            'h2h_games': h2h_stats['h2h_games'] if h2h_stats else 0,
            'h2h_total_goals_avg': h2h_stats['h2h_total_goals_avg'] if h2h_stats else (home_stats['total_goals_per_game_avg'] + away_stats['total_goals_per_game_avg']) / 2,
            'h2h_recent_total_goals_avg': h2h_stats['h2h_recent_total_goals_avg'] if h2h_stats else None,
            
            # Natural Stat Trick advanced stats (if available)
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
            
            # Target variable
            'total_goals': game['total_goals'],
            'home_score': game['home_score'],
            'away_score': game['away_score']
        }
        
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    print(f"\n✓ Created features for {len(features_df)} games")
    
    return features_df


def main():
    """Main function to create features for Poisson regression."""
    print("=" * 60)
    print("NHL Game Features Creator for Poisson Regression (Over/Under)")
    print("=" * 60)
    
    data_dir = "data"
    
    # Load games
    games = load_games(data_dir)
    
    if len(games) == 0:
        print("\n✗ No games found. Please run fetch_historical_seasons.py first.")
        return
    
    # Convert to DataFrame
    print("\nProcessing games...")
    game_info = [extract_game_info(game) for game in games]
    game_info = [g for g in game_info if g is not None]
    
    games_df = pd.DataFrame(game_info)
    games_df['date'] = pd.to_datetime(games_df['date'])
    games_df = games_df.sort_values('date')
    
    print(f"  Total games: {len(games_df)}")
    print(f"  Date range: {games_df['date'].min()} to {games_df['date'].max()}")
    print(f"  Completed games: {games_df['total_goals'].notna().sum()}")
    
    # Create features
    features_df = create_poisson_features(games_df)
    
    # Save features
    output_file = os.path.join(data_dir, "poisson_features.csv")
    features_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved features to {output_file}")
    
    # Show summary
    print("\n" + "=" * 60)
    print("Feature Summary")
    print("=" * 60)
    print(f"Total games with features: {len(features_df)}")
    print(f"Average total goals per game: {features_df['total_goals'].mean():.2f}")
    print(f"Total goals distribution:")
    print(f"  Min: {features_df['total_goals'].min()}")
    print(f"  Max: {features_df['total_goals'].max()}")
    print(f"  Median: {features_df['total_goals'].median()}")
    print(f"  Std: {features_df['total_goals'].std():.2f}")
    
    print("\nFeature columns:")
    feature_cols = [col for col in features_df.columns if col not in [
        'game_id', 'date', 'season', 'home_team_id', 'away_team_id', 
        'home_team_name', 'away_team_name', 'total_goals', 'home_score', 'away_score'
    ]]
    for col in feature_cols:
        non_null_pct = (features_df[col].notna().sum() / len(features_df)) * 100
        print(f"  - {col} ({non_null_pct:.1f}% non-null)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

