import os
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import random
import joblib
import sys
import pandas as pd
from tqdm import tqdm

# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from epmu.model.prepare import extract_features, get_all_feature_names

DATA_DIR = Path("data/2023")
K_PARTICIPANTS = 6
MODEL_DIR = Path("epmu/model/saved_models")
THRESHOLDS = [0.5, 0.6, 0.65, 0.7, 0.75]  # Different thresholds to try


def load_model_and_scaler():
    """Load the trained model and scaler."""
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    scaler = joblib.load(MODEL_DIR / "best_model_scaler.joblib")
    return model, scaler


def load_race_data():
    """Load all JSON files from the data directory."""
    all_data = {}
    for file in sorted(DATA_DIR.glob("*.json")):
        with open(file, "r") as f:
            date_str = file.stem  # Format: dd_mm_yyyy
            all_data[date_str] = json.load(f)
    return all_data


def get_highest_prob_horse(race_data):
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


def get_random_horse(race_data):
    """Get a random horse number from the available horses."""
    if "rapports" not in race_data:
        return None

    available_horses = list(race_data["rapports"].keys())
    if not available_horses:
        return None

    return random.choice(available_horses)


def calculate_bet_multiplier(data):
    """Calculate the bet multiplier for the k-participants strategy."""
    total_valid_races = 0
    races_under_k = 0

    for day_data in data.values():
        for race_key, race_data in day_data.items():
            # Skip if race is not finished or missing required data
            if (
                "ordreArrivee" not in race_data
                or "rapportsDefinitifs" not in race_data
                or "E_SIMPLE_GAGNANT" not in race_data["rapportsDefinitifs"]
            ):
                continue

            total_valid_races += 1
            if len(race_data.get("rapports", {})) <= K_PARTICIPANTS:
                races_under_k += 1

    if races_under_k == 0:
        return 0  # No races to bet on

    # Return multiplier to maintain same total bet amount
    return total_valid_races / races_under_k


def simulate_betting():
    """Simulate betting using all strategies for each race."""
    print("Loading race data...")
    data = load_race_data()
    bet_multiplier = calculate_bet_multiplier(data)

    print("Loading model and scaler...")
    model, scaler = load_model_and_scaler()

    # Initialize profits and bet amounts
    favorite_profit = 0
    random_profit = 0
    k_strat_profit = 0
    threshold_results = {
        threshold: {"profit": 0, "total_bet": 0, "bets_made": 0, "results": []}
        for threshold in THRESHOLDS
    }

    favorite_total_bet = 0
    random_total_bet = 0
    k_strat_total_bet = 0

    favorite_results = []
    random_results = []
    k_strat_results = []

    race_count = 0
    all_predictions = []  # Store all predictions to calculate bet amounts later

    print("\nCollecting predictions for bet amount calculation...")
    # First pass: collect all predictions
    for date_str in tqdm(data.keys(), desc="Collecting predictions", unit="day"):
        day_data = data[date_str]
        for race_key in day_data.keys():
            race_data = day_data[race_key]

            if (
                "ordreArrivee" not in race_data
                or "rapportsDefinitifs" not in race_data
                or "E_SIMPLE_GAGNANT" not in race_data["rapportsDefinitifs"]
            ):
                continue

            favorite_horse = get_highest_prob_horse(race_data)
            if not favorite_horse:
                continue

            features = extract_features(race_data)
            if not features:
                continue

            feature_names = get_all_feature_names()
            feature_values = [features[name] for name in feature_names]
            features_df = pd.DataFrame([feature_values], columns=feature_names)

            features_scaled = scaler.transform(features_df)
            features_scaled_df = pd.DataFrame(features_scaled, columns=feature_names)
            win_prob = model.predict_proba(features_scaled_df)[0][1]

            winner = str(race_data["ordreArrivee"][0][0])
            win_odds = race_data["rapportsDefinitifs"]["E_SIMPLE_GAGNANT"].get(winner)

            all_predictions.append({
                "probability": win_prob,
                "is_winner": favorite_horse == winner,
                "odds": win_odds if favorite_horse == winner else None,
            })

    # Calculate number of bets that would be made for each threshold
    bets_per_threshold = {
        threshold: sum(1 for p in all_predictions if p["probability"] >= threshold)
        for threshold in THRESHOLDS
    }

    print("\nSimulating betting strategies...")
    # Second pass: actual simulation
    race_idx = 0
    for date_str in tqdm(data.keys(), desc="Processing days", unit="day"):
        day_data = data[date_str]
        for race_key in day_data.keys():
            race_data = day_data[race_key]

            if (
                "ordreArrivee" not in race_data
                or "rapportsDefinitifs" not in race_data
                or "E_SIMPLE_GAGNANT" not in race_data["rapportsDefinitifs"]
            ):
                continue

            favorite_horse = get_highest_prob_horse(race_data)
            random_horse = get_random_horse(race_data)

            if not favorite_horse or not random_horse:
                continue

            race_count += 1
            winner = str(race_data["ordreArrivee"][0][0])

            # Process favorite horse bet (1€)
            favorite_total_bet += 1
            if favorite_horse == winner:
                win_odds = race_data["rapportsDefinitifs"]["E_SIMPLE_GAGNANT"].get(winner)
                if win_odds:
                    favorite_profit += win_odds - 1
                else:
                    favorite_profit -= 1
            else:
                favorite_profit -= 1
            favorite_results.append(favorite_profit)

            # Process random horse bet (1€)
            random_total_bet += 1
            if random_horse == winner:
                win_odds = race_data["rapportsDefinitifs"]["E_SIMPLE_GAGNANT"].get(winner)
                if win_odds:
                    random_profit += win_odds - 1
                else:
                    random_profit -= 1
            else:
                random_profit -= 1
            random_results.append(random_profit)

            # Process k-participants strategy
            num_participants = len(race_data.get("rapports", {}))
            if num_participants <= K_PARTICIPANTS:
                bet_amount = bet_multiplier
                k_strat_total_bet += bet_amount
                if favorite_horse == winner:
                    win_odds = race_data["rapportsDefinitifs"]["E_SIMPLE_GAGNANT"].get(winner)
                    if win_odds:
                        k_strat_profit += bet_amount * (win_odds - 1)
                    else:
                        k_strat_profit -= bet_amount
                else:
                    k_strat_profit -= bet_amount
                k_strat_results.append(k_strat_profit)
            else:
                k_strat_results.append(k_strat_profit)

            # Process threshold-based strategies
            prediction = all_predictions[race_idx]
            win_prob = prediction["probability"]

            for threshold in THRESHOLDS:
                if win_prob >= threshold:
                    # Calculate bet amount to maintain same total betting amount
                    bet_amount = race_count / bets_per_threshold[threshold] if bets_per_threshold[threshold] > 0 else 0
                    threshold_results[threshold]["bets_made"] += 1
                    threshold_results[threshold]["total_bet"] += bet_amount

                    if favorite_horse == winner:
                        win_odds = race_data["rapportsDefinitifs"]["E_SIMPLE_GAGNANT"].get(winner)
                        if win_odds:
                            profit = bet_amount * (win_odds - 1)
                            threshold_results[threshold]["profit"] += profit
                        else:
                            threshold_results[threshold]["profit"] -= bet_amount
                    else:
                        threshold_results[threshold]["profit"] -= bet_amount

                threshold_results[threshold]["results"].append(threshold_results[threshold]["profit"])

            race_idx += 1

    # Plot results
    plt.figure(figsize=(15, 10))

    # Plot baseline strategies
    plt.plot(favorite_results, label=f"Favorite (ROI: {100*favorite_profit/favorite_total_bet:.1f}%)")
    plt.plot(random_results, label=f"Random (ROI: {100*random_profit/random_total_bet:.1f}%)")
    if k_strat_total_bet > 0:
        plt.plot(k_strat_results, label=f"K={K_PARTICIPANTS} (ROI: {100*k_strat_profit/k_strat_total_bet:.1f}%)")

    # Plot threshold strategies
    for threshold in THRESHOLDS:
        results = threshold_results[threshold]
        total_bet = results["total_bet"]
        profit = results["profit"]
        roi = 100 * profit / total_bet if total_bet > 0 else 0
        bets_made = results["bets_made"]

        plt.plot(
            results["results"],
            label=f"Threshold {threshold} (ROI: {roi:.1f}%, Bets: {bets_made})"
        )

    plt.xlabel("Race Number")
    plt.ylabel("Cumulative Profit (€)")
    plt.title("Betting Strategy Comparison")
    plt.legend()
    plt.grid(True)

    # Save plot with timestamp
    timestamp = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
    plot_dir = Path("images/simulations")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f"betting_simulation_results_{timestamp}.png")
    plt.close()

    print("\nSimulation Results:")
    print(f"Total races: {race_count}")
    print(f"\nBaseline Strategies:")
    print(f"Favorite: ROI = {100*favorite_profit/favorite_total_bet:.1f}%")
    print(f"Random: ROI = {100*random_profit/random_total_bet:.1f}%")
    if k_strat_total_bet > 0:
        print(f"K={K_PARTICIPANTS}: ROI = {100*k_strat_profit/k_strat_total_bet:.1f}%")

    print(f"\nThreshold Strategies:")
    for threshold in THRESHOLDS:
        results = threshold_results[threshold]
        total_bet = results["total_bet"]
        profit = results["profit"]
        roi = 100 * profit / total_bet if total_bet > 0 else 0
        bets_made = results["bets_made"]

        print(f"Threshold {threshold}:")
        print(f"  ROI = {roi:.1f}%")
        print(f"  Total bet = {total_bet:.2f}€")
        print(f"  Total profit = {profit:.2f}€")
        print(f"  Bets made = {bets_made}")

    print(f"\nResults saved to: images/simulations/betting_simulation_results_{timestamp}.png")


if __name__ == "__main__":
    simulate_betting()
