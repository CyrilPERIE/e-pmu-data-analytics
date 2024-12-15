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

from model.prepare_data import extract_features, get_all_feature_names

DATA_DIR = Path("data/2023")
K_PARTICIPANTS = 6
MODEL_DIR = Path("model/saved_models")

def load_model_and_scaler():
    """Load the trained model and scaler."""
    model = joblib.load(MODEL_DIR / "random_forest.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
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
            if ("ordreArrivee" not in race_data or 
                "rapportsDefinitifs" not in race_data or 
                "E_SIMPLE_GAGNANT" not in race_data["rapportsDefinitifs"]):
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
    model_prob_profit = 0
    
    # Dictionary to store results for different thresholds
    threshold_results = {
        0.5: {"profit": 0, "total_bet": 0, "bets_made": 0, "results": []},
        0.55: {"profit": 0, "total_bet": 0, "bets_made": 0, "results": []},
        0.6: {"profit": 0, "total_bet": 0, "bets_made": 0, "results": []},
        0.65: {"profit": 0, "total_bet": 0, "bets_made": 0, "results": []},
        0.7: {"profit": 0, "total_bet": 0, "bets_made": 0, "results": []}
    }
    
    favorite_total_bet = 0
    random_total_bet = 0
    k_strat_total_bet = 0
    model_prob_total_bet = 0
    
    favorite_results = []
    random_results = []
    k_strat_results = []
    model_prob_results = []
    
    race_count = 0
    all_predictions = []  # Store all predictions to calculate bet amounts later
    
    print("\nSimulating betting strategies...")
    # First pass: collect all predictions
    for date_str in tqdm(data.keys(), desc="Collecting predictions", unit="day"):
        day_data = data[date_str]
        for race_key in day_data.keys():
            race_data = day_data[race_key]
            
            if ("ordreArrivee" not in race_data or 
                "rapportsDefinitifs" not in race_data or 
                "E_SIMPLE_GAGNANT" not in race_data["rapportsDefinitifs"]):
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
                "odds": win_odds if favorite_horse == winner else None
            })
    
    # Calculate number of bets that would be made for each threshold
    bets_per_threshold = {
        threshold: sum(1 for p in all_predictions if p["probability"] >= threshold)
        for threshold in threshold_results.keys()
    }
    
    print("\nSimulating betting strategies with adjusted bet amounts...")
    # Second pass: actual simulation
    race_count = 0
    for date_str in tqdm(data.keys(), desc="Processing days", unit="day"):
        day_data = data[date_str]
        for race_key in day_data.keys():
            race_data = day_data[race_key]
            
            if ("ordreArrivee" not in race_data or 
                "rapportsDefinitifs" not in race_data or 
                "E_SIMPLE_GAGNANT" not in race_data["rapportsDefinitifs"]):
                continue
            
            favorite_horse = get_highest_prob_horse(race_data)
            random_horse = get_random_horse(race_data)
            
            if not favorite_horse or not random_horse:
                continue
            
            race_count += 1
            winner = str(race_data["ordreArrivee"][0][0])
            
            features = extract_features(race_data)
            if not features:
                continue
            
            feature_names = get_all_feature_names()
            feature_values = [features[name] for name in feature_names]
            features_df = pd.DataFrame([feature_values], columns=feature_names)
            
            features_scaled = scaler.transform(features_df)
            features_scaled_df = pd.DataFrame(features_scaled, columns=feature_names)
            win_prob = model.predict_proba(features_scaled_df)[0][1]
            
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
            
            # Process k-participants strategy
            num_participants = len(race_data.get("rapports", {}))
            if num_participants <= K_PARTICIPANTS:
                bet_amount = bet_multiplier
                k_strat_total_bet += bet_amount
                if favorite_horse == winner:
                    win_odds = race_data["rapportsDefinitifs"]["E_SIMPLE_GAGNANT"].get(winner)
                    if win_odds:
                        k_strat_profit += (win_odds * bet_amount) - bet_amount
                    else:
                        k_strat_profit -= bet_amount
                else:
                    k_strat_profit -= bet_amount
            
            # Process model probability strategy (bet 3€ * probability)
            bet_amount = 3 * win_prob
            model_prob_total_bet += bet_amount
            if favorite_horse == winner:
                win_odds = race_data["rapportsDefinitifs"]["E_SIMPLE_GAGNANT"].get(winner)
                if win_odds:
                    model_prob_profit += (win_odds * bet_amount) - bet_amount
                else:
                    model_prob_profit -= bet_amount
            else:
                model_prob_profit -= bet_amount
            
            # Process threshold-based strategies
            base_bet_amount = (favorite_total_bet + random_total_bet + k_strat_total_bet) / 3  # Average of other strategies
            for threshold, result in threshold_results.items():
                if win_prob >= threshold:
                    # Adjust bet amount to match total betting amount of other strategies
                    adjusted_bet = base_bet_amount / bets_per_threshold[threshold] if bets_per_threshold[threshold] > 0 else 0
                    result["total_bet"] += adjusted_bet
                    result["bets_made"] += 1
                    
                    if favorite_horse == winner:
                        win_odds = race_data["rapportsDefinitifs"]["E_SIMPLE_GAGNANT"].get(winner)
                        if win_odds:
                            result["profit"] += (win_odds * adjusted_bet) - adjusted_bet
                        else:
                            result["profit"] -= adjusted_bet
                    else:
                        result["profit"] -= adjusted_bet
                
                result["results"].append((race_count, result["profit"]))
            
            favorite_results.append((race_count, favorite_profit))
            random_results.append((race_count, random_profit))
            k_strat_results.append((race_count, k_strat_profit))
            model_prob_results.append((race_count, model_prob_profit))
    
    return (favorite_results, random_results, k_strat_results, 
            model_prob_results, threshold_results, bet_multiplier,
            favorite_total_bet, random_total_bet, k_strat_total_bet,
            model_prob_total_bet)

def plot_results(favorite_results, random_results, k_strat_results, 
                model_prob_results, threshold_results):
    """Plot the evolution of cumulative profit for all strategies."""
    plt.figure(figsize=(12, 6))
    
    # Plot basic strategies
    plt.plot(*zip(*favorite_results), label='Favorite Horse (1€)', color='blue')
    plt.plot(*zip(*random_results), label='Random Horse (1€)', color='red')
    plt.plot(*zip(*k_strat_results), label=f'K≤{K_PARTICIPANTS} Participants', color='green')
    plt.plot(*zip(*model_prob_results), label='Model Probability (3€ × prob)', color='orange')
    
    # Plot threshold strategies with different shades of purple
    purple_shades = ['#FFB6C1', '#DA70D6', '#9370DB', '#8A2BE2', '#4B0082']
    for (threshold, result), color in zip(threshold_results.items(), purple_shades):
        plt.plot(*zip(*result["results"]), 
                label=f'Threshold {threshold:.2f}', 
                color=color)
    
    plt.title("Comparison of Betting Strategies")
    plt.xlabel("Number of Races")
    plt.ylabel("Cumulative Profit (€)")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("simulations/betting_simulation_results.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    results = simulate_betting()
    if results:
        (favorite_results, random_results, k_strat_results,
         model_prob_results, threshold_results, bet_multiplier,
         favorite_total_bet, random_total_bet, k_strat_total_bet,
         model_prob_total_bet) = results
        
        plot_results(favorite_results, random_results, k_strat_results,
                    model_prob_results, threshold_results)
        
        total_races = favorite_results[-1][0]
        
        print(f"Simulation completed:")
        print(f"Total races analyzed: {total_races}")
        print(f"Bet multiplier for K≤{K_PARTICIPANTS} strategy: {bet_multiplier:.2f}")
        
        # Print results for basic strategies
        basic_strategies = [
            ("Favorite Horse (1€)", favorite_results[-1][1], favorite_total_bet),
            ("Random Horse (1€)", random_results[-1][1], random_total_bet),
            (f"K≤{K_PARTICIPANTS} Participants", k_strat_results[-1][1], k_strat_total_bet),
            ("Model Probability (3€ × prob)", model_prob_results[-1][1], model_prob_total_bet)
        ]
        
        print("\nBasic Strategies:")
        for name, profit, total_bet in basic_strategies:
            print(f"\n{name}:")
            print(f"Total amount bet: {total_bet:.2f}€")
            print(f"Final profit: {profit:.2f}€")
            print(f"ROI: {(profit/total_bet*100 if total_bet > 0 else 0):.2f}%")
            print(f"Average profit per race: {(profit/total_races):.2f}€")
        
        # Print results for threshold strategies
        print("\nThreshold Strategies:")
        for threshold, result in threshold_results.items():
            profit = result["profit"]
            total_bet = result["total_bet"]
            bets_made = result["bets_made"]
            
            print(f"\nThreshold {threshold:.2f}:")
            print(f"Bets made: {bets_made}")
            print(f"Total amount bet: {total_bet:.2f}€")
            print(f"Final profit: {profit:.2f}€")
            print(f"ROI: {(profit/total_bet*100 if total_bet > 0 else 0):.2f}%")
            print(f"Average profit per race: {(profit/total_races):.2f}€")
            print(f"Average bet amount: {(total_bet/bets_made if bets_made > 0 else 0):.2f}€")
