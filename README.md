# Hockey Over/Under Prediction Project

A Poisson regression model to predict NHL game total goals (over/under) for betting purposes.

## Overview

This project uses Poisson regression to predict the total number of goals in NHL games, allowing you to calculate probabilities for any over/under line (e.g., 5.5, 6.5). The model is trained on historical game data and uses team offensive/defensive statistics, recent form, head-to-head matchups, and advanced metrics.

## Features

- **Poisson Regression Model**: Predicts mean total goals using team statistics
- **Probability Calculations**: Calculates probabilities for any over/under line
- **American Odds**: Converts probabilities to American betting odds format
- **Feature Engineering**: Includes team stats, home/away splits, recent form, H2H, and advanced metrics
- **Hyperparameter Optimization**: Randomized search to find best model parameters

## Setup

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate hockey_modelling
```

### Using pip

```bash
pip install -r requirements.txt
```

## Data Requirements

### 1. Historical Game Data

Fetch historical game data from the NHL API:

```bash
python src/fetch_historical_seasons.py
```

This will download games from the last 5 seasons and save them to:
- `data/games_all_seasons.json` (combined file)
- `data/games_YYYY_YYYY.json` (individual season files)

### 2. Natural Stat Trick Data (Optional but Recommended)

Natural Stat Trick provides advanced team-level metrics like expected goals (xGF/xGA), Corsi, and High Danger Chances.

**How to download:**
1. Navigate to [Natural Stat Trick](https://www.naturalstattrick.com/)
2. Go to **"Games"** → **"All Teams"**
3. Select the time period and use the **"Export to CSV"** button
4. Save the file as `data/games.csv` (overwrite if exists)

**Process the NST data:**
```bash
python src/process_nst_data.py
```

This creates `data/nst_processed.csv` with processed advanced statistics.

## Workflow

### Step 1: Fetch Historical Data

```bash
python src/fetch_historical_seasons.py
```

### Step 2: Process NST Data (Optional)

```bash
python src/process_nst_data.py
```

### Step 3: Create Features

Generate features for the Poisson regression model:

```bash
python src/create_poisson_features.py
```

This creates `data/poisson_features.csv` with features for each game, including:
- Team offensive/defensive averages
- Home/away splits
- Recent form (last 10 games)
- Head-to-head statistics
- Advanced metrics (xGF, xGA, etc.)

### Step 4: Train Model

Train the Poisson regression model:

```bash
python src/train_poisson_model.py
```

This will:
- Train a Poisson regression model
- Evaluate performance on test set
- Show metrics for over/under 5.5 and 6.5
- Save model to `models/poisson_regression_model.pkl`

### Step 5: Optimize Hyperparameters (Optional)

Perform hyperparameter optimization to find the best model parameters:

```bash
python src/optimize_poisson_model.py
```

Or with custom settings:

```bash
python src/optimize_poisson_model.py --n_iter 100 --cv 10
```

This will:
- Test 50 random parameter combinations (default)
- Use 5-fold cross-validation (default)
- Save optimized model to `models/poisson_regression_optimized.pkl`

**Note:** This takes several minutes to run.

### Step 6: Make Predictions

Predict over/under probabilities for upcoming games:

```bash
python src/predict_poisson_games.py
```

Or with custom settings:

```bash
# Predict next 14 days
python src/predict_poisson_games.py --days 14

# Use optimized model
python src/predict_poisson_games.py --model models/poisson_regression_optimized.pkl --features models/poisson_model_features_optimized.pkl
```

## Model Performance

The model achieves:
- **MAE**: ~1.95 goals (mean absolute error)
- **RMSE**: ~2.43 goals (root mean squared error)
- **Over 5.5 Accuracy**: ~56-57%
- **Over 6.5 Accuracy**: ~53-54%

The model is well-calibrated, with predicted mean total goals matching actual mean (around 6.2-6.3 goals per game).

## Features Used

The model uses 32 features including:

### Team Statistics
- Goals for/against averages (overall and home/away splits)
- Total goals per game averages
- Recent form (last 10 games)

### Combined Features
- Combined offensive/defensive strength
- Expected total goals (multiple calculations)

### Advanced Metrics (if NST data available)
- Expected goals for/against (xGF/xGA)
- Expected goals percentage (xGF%)
- Corsi percentage (CF%)
- High danger chances (HDCF%)

### Contextual Features
- Head-to-head total goals averages
- Recent H2H performance

## Understanding the Output

### Prediction Format

For each upcoming game, the model outputs:
- **Predicted Total Goals**: Mean expected total goals (lambda)
- **Over/Under 5.5**: Probabilities and American odds
- **Over/Under 6.5**: Probabilities and American odds

### American Odds Format

- **Negative odds** (e.g., -150): Favorite - bet $150 to win $100
- **Positive odds** (e.g., +150): Underdog - bet $100 to win $150

### Example Output

```
📅 Sunday, January 04, 2026
--------------------------------------------------------------------------------

  Montréal Canadiens @ Dallas Stars
  Predicted Total Goals: 6.30

  Over/Under 5.5:
    Over 5.5:  60.1% probability  |  Odds: -151
    Under 5.5: 39.9% probability  |  Odds: +151

  Over/Under 6.5:
    Over 6.5:  44.2% probability  |  Odds: -79
    Under 6.5: 55.8% probability  |  Odds: +79
```

## Model Architecture

The model uses:
- **Poisson Regression**: Appropriate for count data (goals)
- **StandardScaler**: Normalizes features before training
- **L2 Regularization**: Prevents overfitting (alpha=2.0 in optimized model)
- **Log-link function**: Ensures predictions are positive

The model predicts the mean (lambda) of a Poisson distribution, then uses that distribution to calculate probabilities for any over/under line.

## File Structure

```
hockey_modelling_score/
├── data/
│   ├── games_all_seasons.json      # Historical game data
│   ├── games_YYYY_YYYY.json        # Individual season files
│   ├── games.csv                    # Raw NST data (download manually)
│   ├── nst_processed.csv            # Processed NST data
│   └── poisson_features.csv         # Features for training
├── models/
│   ├── poisson_regression_model.pkl           # Trained model
│   ├── poisson_model_features.pkl             # Feature list
│   ├── poisson_regression_optimized.pkl       # Optimized model
│   ├── poisson_model_features_optimized.pkl   # Optimized feature list
│   └── poisson_model_best_params.pkl          # Best parameters
├── src/
│   ├── fetch_historical_seasons.py  # Fetch game data from NHL API
│   ├── process_nst_data.py           # Process Natural Stat Trick data
│   ├── create_poisson_features.py    # Generate features for training
│   ├── train_poisson_model.py       # Train the model
│   ├── optimize_poisson_model.py    # Hyperparameter optimization
│   └── predict_poisson_games.py     # Predict upcoming games
├── environment.yml                   # Conda environment
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Updating Data

To update your dataset with the latest games:

1. **Fetch new games:**
   ```bash
   python src/fetch_historical_seasons.py
   ```

2. **Update features:**
   ```bash
   python src/create_poisson_features.py
   ```

3. **Optional: Retrain model** (if you want to include new data):
   ```bash
   python src/train_poisson_model.py
   ```

## Notes

- The model uses temporal features (stats calculated up to each game date) to avoid look-ahead bias
- Home/away effects are captured through team-specific home/away splits
- NST data is optional but recommended for better predictions
- The model is conservative and doesn't predict extreme outcomes well
- For over 5.5, the model tends to predict "over" for most games (since mean is ~6.2)
- For over 6.5, the model is very conservative, rarely predicting "over"

## Dependencies

- Python 3.10+
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- scipy >= 1.10.0
- numpy >= 1.24.0
- nhl-api-py >= 0.1.0

## License

This project is for educational and research purposes.
