"""
Train a Poisson regression model for predicting NHL game total goals (over/under).

This script:
1. Loads features from poisson_features.csv
2. Trains a Poisson regression model to predict total goals
3. Evaluates model performance
4. Calculates probabilities for over/under 5.5 and 6.5
5. Saves the trained model
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
    
    # Handle missing values
    # For Poisson regression, we'll use a strategy that preserves as much data as possible
    # First, identify columns with many missing values
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
    
    # For remaining features, fill with median (for numeric) or drop rows
    # Strategy: Fill NST features with median, drop rows only if critical features are missing
    critical_features = [
        'home_goals_for_avg', 'home_goals_against_avg',
        'away_goals_for_avg', 'away_goals_against_avg',
        'expected_total_goals_simple'
    ]
    
    # Fill missing values in non-critical features with median
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
            print(f"  - {col} ({missing} missing, will be filled)")
        else:
            print(f"  - {col}")
    
    return X, y, feature_cols


def calculate_poisson_probabilities(lambda_pred, over_line):
    """
    Calculate probability of total goals being over a given line using Poisson distribution.
    
    Args:
        lambda_pred: Predicted mean (lambda) for total goals
        over_line: The line to check (e.g., 5.5 means over 5.5)
    
    Returns:
        Probability of over
    """
    # For over 5.5, we need P(X > 5.5) = 1 - P(X <= 5)
    # Since goals are integers, over 5.5 means 6 or more goals
    threshold = int(np.floor(over_line))
    prob_over = 1 - poisson.cdf(threshold, lambda_pred)
    return prob_over


def evaluate_over_under_predictions(y_true, lambda_pred, line):
    """
    Evaluate over/under predictions for a given line.
    
    Args:
        y_true: Actual total goals
        lambda_pred: Predicted mean total goals
        line: The line (e.g., 5.5)
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Calculate probabilities
    probs_over = calculate_poisson_probabilities(lambda_pred, line)
    
    # Predictions: over if probability > 0.5
    pred_over = probs_over > 0.5
    
    # Actual: over if total goals > line
    actual_over = y_true > line
    
    # Calculate metrics
    accuracy = (pred_over == actual_over).mean()
    
    # Calculate Brier score (probability calibration)
    brier_score = np.mean((probs_over - actual_over.astype(float)) ** 2)
    
    # Calculate log loss
    eps = 1e-15  # Small value to avoid log(0)
    probs_over_clipped = np.clip(probs_over, eps, 1 - eps)
    log_loss = -np.mean(actual_over * np.log(probs_over_clipped) + 
                       (1 - actual_over) * np.log(1 - probs_over_clipped))
    
    # Confusion matrix
    tp = ((pred_over == True) & (actual_over == True)).sum()
    fp = ((pred_over == True) & (actual_over == False)).sum()
    tn = ((pred_over == False) & (actual_over == False)).sum()
    fn = ((pred_over == False) & (actual_over == True)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'brier_score': brier_score,
        'log_loss': log_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'mean_prob_over': probs_over.mean()
    }


def train_model(X, y, test_size=0.2, random_state=42):
    """Train Poisson regression model."""
    print(f"\nSplitting data: {1-test_size:.0%} train, {test_size:.0%} test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"  Training set: {len(X_train)} games")
    print(f"  Test set: {len(X_test)} games")
    
    print(f"\nTarget variable statistics:")
    print(f"  Mean total goals: {y.mean():.2f}")
    print(f"  Std total goals: {y.std():.2f}")
    print(f"  Min: {y.min()}, Max: {y.max()}")
    
    print("\nTraining Poisson regression model with StandardScaler pipeline...")
    # Create pipeline with StandardScaler and PoissonRegressor
    # PoissonRegressor uses log-link by default, which is appropriate for count data
    model = make_pipeline(
        StandardScaler(),
        PoissonRegressor(
            alpha=0.1,  # L2 regularization
            max_iter=1000
        )
    )
    
    model.fit(X_train, y_train)
    
    # Predict
    print("\nEvaluating model...")
    lambda_train = model.predict(X_train)
    lambda_test = model.predict(X_test)
    
    # Ensure predictions are positive (Poisson requires lambda > 0)
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
    
    # Additional diagnostics
    print(f"\nPrediction Statistics:")
    print(f"  Mean predicted lambda (test): {lambda_test.mean():.2f} goals")
    print(f"  Mean actual total goals (test): {y_test.mean():.2f} goals")
    print(f"  Predicted lambda range: {lambda_test.min():.2f} - {lambda_test.max():.2f}")
    print(f"  Actual total goals range: {y_test.min():.0f} - {y_test.max():.0f}")
    
    # Show distribution of predictions vs actuals
    print(f"\nActual vs Predicted Distribution (Test Set):")
    actual_over_55 = (y_test > 5.5).sum()
    actual_under_55 = (y_test <= 5.5).sum()
    print(f"  Actual Over 5.5: {actual_over_55} ({actual_over_55/len(y_test)*100:.1f}%)")
    print(f"  Actual Under 5.5: {actual_under_55} ({actual_under_55/len(y_test)*100:.1f}%)")
    
    actual_over_65 = (y_test > 6.5).sum()
    actual_under_65 = (y_test <= 6.5).sum()
    print(f"  Actual Over 6.5: {actual_over_65} ({actual_over_65/len(y_test)*100:.1f}%)")
    print(f"  Actual Under 6.5: {actual_under_65} ({actual_under_65/len(y_test)*100:.1f}%)")
    
    # Evaluate over/under predictions for 5.5 and 6.5
    print(f"\n" + "=" * 60)
    print("Over/Under Evaluation")
    print("=" * 60)
    
    for line in [5.5, 6.5]:
        print(f"\nOver {line} Line:")
        train_metrics = evaluate_over_under_predictions(y_train, lambda_train, line)
        test_metrics = evaluate_over_under_predictions(y_test, lambda_test, line)
        
        print(f"  Training Set:")
        print(f"    Accuracy: {train_metrics['accuracy']:.2%}")
        print(f"    Precision: {train_metrics['precision']:.2%}")
        print(f"    Recall: {train_metrics['recall']:.2%}")
        print(f"    F1 Score: {train_metrics['f1']:.2%}")
        print(f"    Brier Score: {train_metrics['brier_score']:.4f}")
        print(f"    Log Loss: {train_metrics['log_loss']:.4f}")
        print(f"    Mean Predicted Prob: {train_metrics['mean_prob_over']:.2%}")
        
        print(f"  Test Set:")
        print(f"    Accuracy: {test_metrics['accuracy']:.2%}")
        print(f"    Precision: {test_metrics['precision']:.2%}")
        print(f"    Recall: {test_metrics['recall']:.2%}")
        print(f"    F1 Score: {test_metrics['f1']:.2%}")
        print(f"    Brier Score: {test_metrics['brier_score']:.4f}")
        print(f"    Log Loss: {test_metrics['log_loss']:.4f}")
        print(f"    Mean Predicted Prob: {test_metrics['mean_prob_over']:.2%}")
        
        print(f"    Confusion Matrix:")
        print(f"      True Positives (Over predicted correctly): {test_metrics['true_positives']}")
        print(f"      False Positives (Over predicted, but Under): {test_metrics['false_positives']}")
        print(f"      True Negatives (Under predicted correctly): {test_metrics['true_negatives']}")
        print(f"      False Negatives (Under predicted, but Over): {test_metrics['false_negatives']}")
    
    return model, X_test, y_test, lambda_test


def save_model(model, feature_cols, models_dir="models"):
    """Save the trained model and feature list."""
    os.makedirs(models_dir, exist_ok=True)
    
    model_file = os.path.join(models_dir, "poisson_regression_model.pkl")
    features_file = os.path.join(models_dir, "poisson_model_features.pkl")
    
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Saved model to {model_file}")
    
    with open(features_file, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"✓ Saved feature list to {features_file}")


def main():
    """Main function to train Poisson regression model."""
    print("=" * 60)
    print("NHL Poisson Regression Model Training (Over/Under Prediction)")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Prepare features
    X, y, feature_cols = prepare_features(df)
    
    # Train model
    model, X_test, y_test, lambda_test = train_model(X, y)
    
    # Save model
    save_model(model, feature_cols)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nModel saved. You can now use it to predict over/under probabilities for new games.")


if __name__ == "__main__":
    main()

