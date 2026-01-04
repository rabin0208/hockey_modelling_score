"""
Hyperparameter optimization for Poisson regression model.

This script performs randomized search to find the best hyperparameters
for the Poisson regression model to predict over/unders.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import poisson


def load_data(data_dir="data"):
    """Load the Poisson features CSV."""
    filepath = os.path.join(data_dir, "poisson_features.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Features file not found: {filepath}\n"
            "Please run create_poisson_features.py first to generate features."
        )
    
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} games")
    return df


def prepare_features(df):
    """Prepare features and target variable."""
    # Feature columns (exclude identifiers and target)
    exclude_cols = [
        'game_id', 'date', 'season', 'home_team_id', 'away_team_id',
        'home_team_name', 'away_team_name', 'total_goals', 'home_score', 'away_score'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['total_goals'].copy()
    
    # Handle missing values (same strategy as train_poisson_model.py)
    missing_pct = X.isna().sum() / len(X) * 100
    high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
    
    if high_missing_cols:
        print(f"\nDropping {len(high_missing_cols)} features with >50% missing values:")
        for col in high_missing_cols:
            print(f"  - {col} ({missing_pct[col]:.1f}% missing)")
        X = X.drop(columns=high_missing_cols)
        feature_cols = [col for col in feature_cols if col not in high_missing_cols]
    
    # Update feature_cols to match actual columns in X
    feature_cols = [col for col in feature_cols if col in X.columns]
    
    # Fill missing values in non-critical features with median
    critical_features = [
        'home_goals_for_avg', 'home_goals_against_avg',
        'away_goals_for_avg', 'away_goals_against_avg',
        'expected_total_goals_simple'
    ]
    
    for col in X.columns:
        if col not in critical_features and X[col].isna().any():
            median_val = X[col].median()
            if pd.notna(median_val):
                X[col] = X[col].fillna(median_val)
    
    # Drop rows where critical features are missing
    before_drop = len(X)
    critical_mask = X[critical_features].notna().all(axis=1)
    X = X[critical_mask].copy()
    y = y[critical_mask].copy()
    after_drop = len(X)
    
    if before_drop != after_drop:
        print(f"\nDropped {before_drop - after_drop} games with missing critical features ({((before_drop - after_drop)/before_drop)*100:.1f}%)")
        print(f"  Using {after_drop} games with complete critical data")
    
    print(f"\nFeatures used ({len(feature_cols)}):")
    for col in feature_cols:
        missing = X[col].isna().sum()
        if missing > 0:
            print(f"  - {col} ({missing} missing, filled with median)")
        else:
            print(f"  - {col}")
    
    return X, y, feature_cols


def calculate_poisson_probability(lambda_pred, over_line):
    """Calculate probability of total goals being over a given line."""
    threshold = int(np.floor(over_line))
    prob_over = 1 - poisson.cdf(threshold, lambda_pred)
    return prob_over


def evaluate_over_under_predictions(y_true, lambda_pred, line):
    """Evaluate over/under predictions for a given line."""
    probs_over = calculate_poisson_probability(lambda_pred, line)
    pred_over = probs_over > 0.5
    actual_over = y_true > line
    accuracy = (pred_over == actual_over).mean()
    return accuracy


def optimize_hyperparameters(X, y, cv=5, n_iter=50, random_state=42):
    """
    Perform randomized search to find best hyperparameters.
    
    Args:
        X: Feature matrix
        y: Target variable (total goals)
        cv: Number of cross-validation folds
        n_iter: Number of parameter combinations to try (default 50)
        random_state: Random seed
    
    Returns:
        Best model and results
    """
    print(f"\nSplitting data: 80% train, 20% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    print(f"  Training set: {len(X_train)} games")
    print(f"  Test set: {len(X_test)} games")
    
    # Create pipeline
    pipeline = make_pipeline(
        StandardScaler(),
        PoissonRegressor(max_iter=2000)  # Higher max_iter for optimization
    )
    
    # Define hyperparameter search space for PoissonRegressor
    # Key hyperparameters:
    # - alpha: Regularization strength (L2 penalty)
    # - fit_intercept: Whether to fit intercept
    # - max_iter: Maximum iterations (already set in pipeline, but can override)
    param_distributions = {
        'poissonregressor__alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'poissonregressor__fit_intercept': [True, False],
        'poissonregressor__max_iter': [500, 1000, 2000, 3000],
        'poissonregressor__tol': [1e-4, 1e-5, 1e-6]
    }
    
    # Calculate total possible combinations
    total_combos = 1
    for key, values in param_distributions.items():
        total_combos *= len(values)
    
    print("\n" + "=" * 60)
    print("Starting Randomized Search with Cross-Validation...")
    print("=" * 60)
    print(f"  Cross-validation folds: {cv}")
    print(f"  Parameter combinations to try: {n_iter}")
    print(f"  Total possible combinations: {total_combos}")
    print(f"  (Randomized search samples {n_iter} random combinations)")
    print("\n  This may take several minutes...")
    
    # Create randomized search
    # Use negative MAE as scoring (higher is better, so we negate MAE)
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_absolute_error',  # Negative MAE (higher is better)
        n_jobs=-1,  # Use all available CPU cores
        verbose=1,
        random_state=random_state,
        return_train_score=True
    )
    
    # Perform randomized search
    random_search.fit(X_train, y_train)
    
    # Get best model
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    print("\n" + "=" * 60)
    print("Randomized Search Results")
    print("=" * 60)
    print(f"\nBest Cross-Validation Score (Negative MAE): {best_score:.4f}")
    print(f"Best Cross-Validation MAE: {-best_score:.4f} goals")
    print(f"\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating Best Model on Test Set")
    print("=" * 60)
    
    lambda_train = best_model.predict(X_train)
    lambda_test = best_model.predict(X_test)
    
    # Ensure predictions are positive
    lambda_train = np.maximum(lambda_train, 0.1)
    lambda_test = np.maximum(lambda_test, 0.1)
    
    # Regression metrics
    train_mae = mean_absolute_error(y_train, lambda_train)
    test_mae = mean_absolute_error(y_test, lambda_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, lambda_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, lambda_test))
    
    print(f"\nModel Performance (Regression):")
    print(f"  Training MAE: {train_mae:.2f} goals")
    print(f"  Test MAE: {test_mae:.2f} goals")
    print(f"  Training RMSE: {train_rmse:.2f} goals")
    print(f"  Test RMSE: {test_rmse:.2f} goals")
    
    # Evaluate over/under predictions
    print(f"\nOver/Under Performance:")
    for line in [5.5, 6.5]:
        train_acc = evaluate_over_under_predictions(y_train, lambda_train, line)
        test_acc = evaluate_over_under_predictions(y_test, lambda_test, line)
        print(f"  Over {line} Line:")
        print(f"    Training Accuracy: {train_acc:.2%}")
        print(f"    Test Accuracy: {test_acc:.2%}")
    
    # Show top 5 parameter combinations
    print("\n" + "=" * 60)
    print("Top 5 Parameter Combinations")
    print("=" * 60)
    results_df = pd.DataFrame(random_search.cv_results_)
    top_5 = results_df.nlargest(5, 'mean_test_score')[
        ['params', 'mean_test_score', 'std_test_score']
    ]
    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"\n  Rank {idx}:")
        print(f"    Score (Neg MAE): {row['mean_test_score']:.4f} (+/- {row['std_test_score']*2:.4f})")
        print(f"    MAE: {-row['mean_test_score']:.4f} goals")
        print(f"    Params: {row['params']}")
    
    return best_model, best_params, random_search, X_test, y_test, lambda_test


def save_model(model, feature_cols, best_params, model_dir="models"):
    """Save the optimized model as a pickle file."""
    # Create models directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"\nCreated {model_dir}/ folder")
    
    # Save model
    model_file = os.path.join(model_dir, "poisson_regression_optimized.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature list
    features_file = os.path.join(model_dir, "poisson_model_features_optimized.pkl")
    with open(features_file, 'wb') as f:
        pickle.dump(feature_cols, f)
    
    # Save best parameters
    params_file = os.path.join(model_dir, "poisson_model_best_params.pkl")
    with open(params_file, 'wb') as f:
        pickle.dump(best_params, f)
    
    print(f"\n✓ Optimized model saved to {model_file}")
    print(f"✓ Feature list saved to {features_file}")
    print(f"✓ Best parameters saved to {params_file}")
    return model_file


def main():
    """Main function to optimize and save the model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize Poisson regression hyperparameters')
    parser.add_argument(
        '--n_iter',
        type=int,
        default=50,
        help='Number of parameter combinations to try (default: 50)'
    )
    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NHL Poisson Regression Model - Hyperparameter Optimization")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Prepare features
    X, y, feature_cols = prepare_features(df)
    
    # Optimize hyperparameters
    best_model, best_params, random_search, X_test, y_test, lambda_test = optimize_hyperparameters(
        X, y, cv=args.cv, n_iter=args.n_iter
    )
    
    # Save model
    model_file = save_model(best_model, feature_cols, best_params)
    
    print("\n" + "=" * 60)
    print("Hyperparameter optimization complete!")
    print(f"Optimized model saved to: {model_file}")
    print("=" * 60)
    print("\nYou can now use this optimized model for predictions:")
    print("  python src/predict_poisson_games.py --model models/poisson_regression_optimized.pkl --features models/poisson_model_features_optimized.pkl")


if __name__ == "__main__":
    main()

