import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
from scipy.stats import skew
import joblib

DATA_DIR = Path("data/2024")

# Load category statistics
try:
    CATEGORY_STATS = joblib.load("epmu/model/data/category_stats.joblib")
    HIPPODROME_RACE_COUNTS = joblib.load("epmu/model/data/hippodrome_race_counts.joblib")
except:
    CATEGORY_STATS = None
    HIPPODROME_RACE_COUNTS = None

# Predefined categories for one-hot encoding
CATEGORIES = {
    "discipline": [
        "ATTELE",
        "CROSS",
        "HAIE",
        "MONTE",
        "PLAT",
        "STEEPLECHASE",
        "UNKNOWN",
    ],
    "specialite": [
        "OBSTACLE",
        "PLAT",
        "TROT_ATTELE",
        "TROT_MONTE",
        "UNKNOWN",
    ],
    "conditionSexe": [
        "FEMELLES",
        "FEMELLES_ET_MALES",
        "MALES",
        "MALES_ET_HONGRES",
        "TOUS_CHEVAUX",
        "UNKNOWN",
    ],
    "nature": [
        "DIURNE",
        "NOCTURNE",
        "SEMINOCTURNE",
        "UNKNOWN",
    ],
    "nebulosite": [
        "P1",
        "P10",
        "P11",
        "P13",
        "P13a",
        "P15",
        "P18",
        "P18a",
        "P4",
        "P5",
        "P6",
        "P7",
        "P7d",
        "P8",
        "P8c",
        "UNKNOWN",
    ],
}

def get_all_feature_names() -> List[str]:
    """Get a list of all possible feature names in the correct order."""
    feature_names = [
        # Basic features
        "num_participants",
        "distance",
        "prize_money",
        # Time features
        "hour",
        "minute",
        "is_afternoon",
        # Prize money ratios
        "prize_1st_2nd_ratio",
        "prize_2nd_3rd_ratio",
        "prize_total_ratio",
        # Weather features
        "temperature",
        "wind_force",
        # Boolean features
        "is_grand_prix",
        # Odds-related features
        "favorite_second_diff",
        "favorite_second_ratio",
        "favorite_mean_ratio",
        "top_3_std",
        "min_odds",
        "max_odds",
        "mean_odds",
        "std_odds",
        "odds_range",
        # Betting amounts features
        "total_betting_amount",
        "max_betting_amount",
        "min_betting_amount",
        "mean_betting_amount",
        "std_betting_amount",
        "top_bet_ratio",
        "top_2_bet_ratio",
        # Hippodrome race count
        "hippodrome_race_count",
    ]
    
    # Add categorical features
    for feature, categories in CATEGORIES.items():
        for category in categories:
            feature_names.append(f"{feature}_{category}")
    
    return feature_names

def get_highest_prob_horse(race_data: Dict[str, Any]) -> str:
    """Get the horse number with lowest odds (highest probability)."""
    if "rapports" not in race_data:
        return None
    
    # Get the last odds for each horse
    latest_odds = {}
    for horse, odds_history in race_data["rapports"].items():
        if odds_history:  # Check if there are any odds
            latest_timestamp = max(odds_history.keys())
            latest_odds[horse] = odds_history[latest_timestamp]
    
    if not latest_odds:
        return None
    
    # Return horse number with lowest odds
    return min(latest_odds.items(), key=lambda x: x[1])[0]

def get_sorted_odds(race_data: Dict[str, Any]) -> List[float]:
    """Get sorted list of odds (ascending)."""
    if "rapports" not in race_data:
        return []
    
    odds_list = []
    for horse, odds_history in race_data["rapports"].items():
        if odds_history:
            latest_timestamp = max(odds_history.keys())
            odds_list.append(odds_history[latest_timestamp])
    
    return sorted(odds_list) if odds_list else []

def extract_time_features(timestamp: int) -> Dict[str, float]:
    """Extract time-related features from timestamp."""
    dt = datetime.fromtimestamp(timestamp / 1000)  # Convert from milliseconds
    return {
        "hour": dt.hour,
        "minute": dt.minute,
        "is_afternoon": 1 if dt.hour >= 12 else 0
    }

def odds_to_probability(odds: float) -> float:
    """Convert odds to probability."""
    return 1 / odds if odds > 0 else 0

def get_betting_amounts(race_data: Dict[str, Any]) -> List[float]:
    """Get list of betting amounts."""
    amounts = []
    if "enjeux" in race_data:
        simple_gagnant = race_data["enjeux"].get("E_SIMPLE_GAGNANT", {})
        if simple_gagnant:
            for horse_bets in simple_gagnant.values():
                if horse_bets:
                    latest_timestamp = max(horse_bets.keys())
                    amounts.append(horse_bets[latest_timestamp])
    return amounts

def extract_features(race_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract relevant features from a race."""
    features = {}
    
    # Basic race features
    features["num_participants"] = len(race_data.get("rapports", {}))
    features["distance"] = race_data.get("distance", 0)
    features["prize_money"] = race_data.get("montantPrix", 0)
    
    # Time features
    time_features = extract_time_features(race_data.get("heureDepart", 0))
    features.update(time_features)
    
    # Prize money distribution features
    montant_1er = race_data.get("montantOffert1er", 0)
    montant_2eme = race_data.get("montantOffert2eme", 0)
    montant_3eme = race_data.get("montantOffert3eme", 0)
    
    if montant_1er and montant_2eme:
        features["prize_1st_2nd_ratio"] = montant_1er / montant_2eme if montant_2eme != 0 else 0
    if montant_2eme and montant_3eme:
        features["prize_2nd_3rd_ratio"] = montant_2eme / montant_3eme if montant_3eme != 0 else 0
    features["prize_total_ratio"] = montant_1er / race_data.get("montantPrix", 1)
    
    # Weather features
    meteo = race_data.get("meteo", {})
    features["temperature"] = meteo.get("temperature", 0)
    features["wind_force"] = meteo.get("forceVent", 0)
    
    # Category statistics features
    if CATEGORY_STATS:
        hippodrome = race_data.get("hippodrome", {}).get("libelleCourt", "UNKNOWN")
        discipline = race_data.get("discipline", "UNKNOWN")
        specialite = race_data.get("specialite", "UNKNOWN")
        
        features["hippodrome_win_rate"] = CATEGORY_STATS["hippodrome"].get(hippodrome, 0.5)
        features["discipline_win_rate"] = CATEGORY_STATS["discipline"].get(discipline, 0.5)
        features["specialite_win_rate"] = CATEGORY_STATS["specialite"].get(specialite, 0.5)
        
        # Add hippodrome race count
        if HIPPODROME_RACE_COUNTS:
            features["hippodrome_race_count"] = HIPPODROME_RACE_COUNTS.get(hippodrome, 0)
    
    # Odds-related features
    sorted_odds = get_sorted_odds(race_data)
    if len(sorted_odds) >= 2:
        features["favorite_second_diff"] = sorted_odds[1] - sorted_odds[0]
        features["favorite_second_ratio"] = sorted_odds[1] / sorted_odds[0] if sorted_odds[0] != 0 else 0
        features["favorite_mean_ratio"] = sorted_odds[0] / np.mean(sorted_odds) if np.mean(sorted_odds) != 0 else 0
        features["top_3_std"] = np.std(sorted_odds[:3]) if len(sorted_odds) >= 3 else 0
        features["odds_skewness"] = skew(sorted_odds) if len(sorted_odds) >= 3 else 0
        
        # Convert odds to probabilities and compute probability-based features
        probabilities = [odds_to_probability(odds) for odds in sorted_odds]
        features["prob_skewness"] = skew(probabilities) if len(probabilities) >= 3 else 0
        features["prob_std"] = np.std(probabilities)
        features["prob_range"] = max(probabilities) - min(probabilities)
        features["prob_favorite_second_diff"] = probabilities[0] - probabilities[1]
        features["prob_favorite_second_ratio"] = probabilities[0] / probabilities[1] if probabilities[1] > 0 else 0
        features["prob_favorite_mean_ratio"] = probabilities[0] / np.mean(probabilities) if np.mean(probabilities) > 0 else 0
    else:
        features["favorite_second_diff"] = 0
        features["favorite_second_ratio"] = 0
        features["favorite_mean_ratio"] = 0
        features["top_3_std"] = 0
        features["odds_skewness"] = 0
        features["prob_skewness"] = 0
        features["prob_std"] = 0
        features["prob_range"] = 0
        features["prob_favorite_second_diff"] = 0
        features["prob_favorite_second_ratio"] = 0
        features["prob_favorite_mean_ratio"] = 0
    
    if "rapports" in race_data:
        odds_list = []
        for horse, odds_history in race_data["rapports"].items():
            if odds_history:
                latest_timestamp = max(odds_history.keys())
                odds_list.append(odds_history[latest_timestamp])
        
        if odds_list:
            features["min_odds"] = min(odds_list)
            features["max_odds"] = max(odds_list)
            features["mean_odds"] = np.mean(odds_list)
            features["std_odds"] = np.std(odds_list)
            features["odds_range"] = max(odds_list) - min(odds_list)
        else:
            features["min_odds"] = 0
            features["max_odds"] = 0
            features["mean_odds"] = 0
            features["std_odds"] = 0
            features["odds_range"] = 0
    
    # Betting amounts features
    betting_amounts = get_betting_amounts(race_data)
    if betting_amounts:
        features["total_betting_amount"] = sum(betting_amounts)
        features["max_betting_amount"] = max(betting_amounts)
        features["min_betting_amount"] = min(betting_amounts)
        features["mean_betting_amount"] = np.mean(betting_amounts)
        features["std_betting_amount"] = np.std(betting_amounts)
        features["betting_skewness"] = skew(betting_amounts) if len(betting_amounts) >= 3 else 0
        
        # Add betting concentration features
        sorted_amounts = sorted(betting_amounts, reverse=True)
        features["top_bet_ratio"] = sorted_amounts[0] / sum(betting_amounts) if sum(betting_amounts) != 0 else 0
        if len(sorted_amounts) >= 2:
            features["top_2_bet_ratio"] = sum(sorted_amounts[:2]) / sum(betting_amounts) if sum(betting_amounts) != 0 else 0
        else:
            features["top_2_bet_ratio"] = 0
    else:
        features["total_betting_amount"] = 0
        features["max_betting_amount"] = 0
        features["min_betting_amount"] = 0
        features["mean_betting_amount"] = 0
        features["std_betting_amount"] = 0
        features["betting_skewness"] = 0
        features["top_bet_ratio"] = 0
        features["top_2_bet_ratio"] = 0

    # Ensure all features are present in the correct order
    all_features = {name: features.get(name, 0) for name in get_all_feature_names()}
    return all_features

def prepare_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare X (features) and y (target) for the model."""
    all_features = []
    targets = []
    
    # Load all race data
    for file in sorted(DATA_DIR.glob("*.json")):
        with open(file, "r") as f:
            day_data = json.load(f)
            
            for race_key, race_data in day_data.items():
                # Skip if race is not finished or missing required data
                if ("ordreArrivee" not in race_data or 
                    "rapportsDefinitifs" not in race_data or 
                    "E_SIMPLE_GAGNANT" not in race_data["rapportsDefinitifs"]):
                    continue
                
                favorite_horse = get_highest_prob_horse(race_data)
                if not favorite_horse:
                    continue
                
                # Extract features
                features = extract_features(race_data)
                if not features:
                    continue
                
                # Get target (1 if favorite won, 0 otherwise)
                winner = str(race_data["ordreArrivee"][0][0])
                target = 1 if favorite_horse == winner else 0
                
                all_features.append(features)
                targets.append(target)
    
    # Convert to pandas DataFrame/Series
    X = pd.DataFrame(all_features)
    y = pd.Series(targets)
    
    # Fill missing values for one-hot encoded columns with 0
    X = X.fillna(0)
    
    # Save the processed data
    os.makedirs("epmu/model/data", exist_ok=True)
    X.to_csv("epmu/model/data/features.csv", index=False)
    y.to_csv("epmu/model/data/targets.csv", index=False)
    
    print(f"Dataset prepared:")
    print(f"Number of samples: {len(y)}")
    print(f"Number of features: {len(X.columns)}")
    print(f"\nFeature names:")
    for col in X.columns:
        print(f"- {col}")
    print(f"\nClass distribution:")
    print(y.value_counts(normalize=True))
    
    # Print feature names for debugging
    print("\nFeature names in order:")
    print(X.columns.tolist())
    
    return X, y

if __name__ == "__main__":
    X, y = prepare_dataset()
