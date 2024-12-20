import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb
from prepare import prepare_dataset
import joblib
import json

# Constants
RANDOM_STATE = 42
N_SPLITS = 5
RESULTS_DIR = Path("epmu/model/optimization_results")
MODELS_DIR = Path("epmu/model/saved_models")


def expected_profit_score(y_true, y_pred_proba, winning_odds, threshold):
    """
    Custom scoring function that simulates betting returns using actual odds.
    Assumes betting 1€ when model probability > threshold.

    Args:
        y_true: True labels (1 if favorite won, 0 otherwise)
        y_pred_proba: Predicted probabilities of favorite winning
        winning_odds: Array of actual winning odds for each race
        threshold: Threshold for betting

    Returns:
        Expected profit per race
    """
    # Only bet when probability > threshold
    bets = y_pred_proba[:, 1] > threshold

    if not any(bets):
        return 0

    # Fixed bet amount of 1€
    bet_amounts = np.ones(len(y_true))

    # Calculate returns using actual odds
    wins = (y_true == 1) & bets
    losses = ~wins & bets

    # Initialize returns array with zeros (no bets)
    returns = np.zeros(len(y_true))

    # Update returns only where we bet
    returns[wins] = bet_amounts[wins] * (winning_odds[wins] - 1)  # Winning bets
    returns[losses] = -bet_amounts[losses]  # Losing bets

    # Return average profit per race (including no-bet races)
    return np.mean(returns)


def make_profit_scorer(winning_odds, threshold=0.5):
    """Create a scorer that includes the winning odds."""

    def scorer(estimator, X, y):
        # Get indices from y to slice winning_odds
        indices = y.index if hasattr(y, "index") else np.arange(len(y))
        fold_winning_odds = winning_odds[indices]

        y_pred_proba = estimator.predict_proba(X)
        return expected_profit_score(y, y_pred_proba, fold_winning_odds, threshold)

    return scorer


def setup_directories():
    """Create necessary directories."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "plots").mkdir(exist_ok=True)


def get_models():
    """Get all models to evaluate."""
    base_models = {
        "logistic": LogisticRegression(random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "lgb": lgb.LGBMClassifier(random_state=RANDOM_STATE),
        "gb": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "svm": SVC(probability=True, random_state=RANDOM_STATE),
    }

    # Add ensemble models
    estimators = [(name, model) for name, model in base_models.items()]
    ensemble_models = {
        "voting_soft": VotingClassifier(estimators=estimators, voting="soft"),
        "stacking": StackingClassifier(
            estimators=estimators[:-1],  # Exclude SVM as final estimator
            final_estimator=base_models["rf"],
            cv=3,
        ),
    }

    return {**base_models, **ensemble_models}


def evaluate_model(model, X, y, winning_odds):
    """Evaluate a model using cross-validation."""
    # Ensure data types
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "profit_0.5": make_profit_scorer(winning_odds, 0.5),
        "profit_0.6": make_profit_scorer(winning_odds, 0.6),
        "profit_0.7": make_profit_scorer(winning_odds, 0.7),
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    scores = cross_validate(model, X, y, scoring=scoring, cv=cv, return_train_score=True, n_jobs=-1)

    return scores


def save_results(results):
    """Save evaluation results."""
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "model_evaluation_results.csv")

    # Create summary plots
    metrics = ["test_profit_0.5", "test_roc_auc"]
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    for i, metric in enumerate(metrics):
        sns.boxplot(data=pd.melt(results_df[["Model", metric]], id_vars=["Model"]), x="Model", y="value", ax=axes[i])
        axes[i].set_title(f'{metric.replace("test_", "").upper()} Distribution')
        axes[i].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plots" / "metrics_comparison.png")
    plt.close()


def save_best_model(results_df, X, y, winning_odds, models, scaler, timestamp):
    """Save the best performing model and its associated components."""
    # Add a column for max profit across thresholds
    results_df["max_profit"] = results_df[["test_profit_0.5", "test_profit_0.6", "test_profit_0.7"]].max(axis=1)
    results_df["best_threshold"] = (
        results_df[["test_profit_0.5", "test_profit_0.6", "test_profit_0.7"]]
        .idxmax(axis=1)
        .map({"test_profit_0.5": 0.5, "test_profit_0.6": 0.6, "test_profit_0.7": 0.7})
    )

    # Get the best model configuration
    best_model_row = results_df.nlargest(1, "max_profit").iloc[0]
    model_name = best_model_row["Model"]
    best_threshold = best_model_row["best_threshold"]

    print("\nBest model configuration:")
    print(f"Model: {model_name}")
    print(f"Best threshold: {best_threshold}")
    print(f"Expected Profit at threshold {best_threshold}: {best_model_row['max_profit']:.4f}")
    print(f"Profits at different thresholds:")
    print(f"- 0.5: {best_model_row['test_profit_0.5']:.4f}")
    print(f"- 0.6: {best_model_row['test_profit_0.6']:.4f}")
    print(f"- 0.7: {best_model_row['test_profit_0.7']:.4f}")
    print(f"ROC AUC: {best_model_row['test_roc_auc']:.4f}")

    # Get the model
    model = models[model_name]

    # Scale the features
    print("\nScaling features...")
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Train the final model on full dataset
    print("Training final model...")
    model.fit(X_scaled, y)

    # Save components
    print("\nSaving model components...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save model
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(model, model_path)
    print(f"- Model saved to: {model_path}")

    # Save scaler
    scaler_path = MODELS_DIR / "best_model_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"- Scaler saved to: {scaler_path}")

    # Save model info
    model_info = {
        "model_name": model_name,
        "best_threshold": float(best_threshold),  # Convert numpy float to Python float for JSON
        "timestamp": timestamp,
        "metrics": {
            "max_profit": float(best_model_row["max_profit"]),
            "profit_0.5": float(best_model_row["test_profit_0.5"]),
            "profit_0.6": float(best_model_row["test_profit_0.6"]),
            "profit_0.7": float(best_model_row["test_profit_0.7"]),
            "roc_auc": float(best_model_row["test_roc_auc"]),
            "precision": float(best_model_row["test_precision"]),
            "recall": float(best_model_row["test_recall"]),
            "f1": float(best_model_row["test_f1"]),
            "accuracy": float(best_model_row["test_accuracy"]),
        },
    }
    info_path = MODELS_DIR / "best_model_info.json"
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=4)
    print(f"- Model info saved to: {info_path}")


def optimize_models():
    """Main function to run model optimization."""
    print("Loading and preparing dataset...")
    X, y, winning_odds = prepare_dataset()
    feature_names = X.columns.tolist()

    # Generate timestamp for this optimization run
    timestamp = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

    print("\nGetting models...")
    models = get_models()

    results = []

    # Evaluate all models
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        scores = evaluate_model(model, X_scaled, y, winning_odds)
        results.append(
            {
                "Model": model_name,
                **{k: np.mean(v) for k, v in scores.items()},
            }
        )

    # Save results
    results_df = pd.DataFrame(results)
    save_results(results_df)

    # Print summary
    print("\nModel Performance Summary (sorted by max profit across thresholds):")
    results_df["max_profit"] = results_df[["test_profit_0.5", "test_profit_0.6", "test_profit_0.7"]].max(axis=1)
    results_df["best_threshold"] = (
        results_df[["test_profit_0.5", "test_profit_0.6", "test_profit_0.7"]]
        .idxmax(axis=1)
        .map({"test_profit_0.5": 0.5, "test_profit_0.6": 0.6, "test_profit_0.7": 0.7})
    )
    print(
        results_df[["Model", "max_profit", "best_threshold", "test_roc_auc", "test_precision"]].sort_values(
            "max_profit", ascending=False
        )
    )

    # Save the best model
    save_best_model(results_df, X, y, winning_odds, models, scaler, timestamp)

    return results_df, timestamp


if __name__ == "__main__":
    setup_directories()
    results_df, timestamp = optimize_models()

    print("\nOptimization completed! Results saved in:")
    print(f"- {RESULTS_DIR}/model_evaluation_results.csv")
    print(f"- {RESULTS_DIR}/plots/")
