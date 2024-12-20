import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib


MIN_RACES_COUNT = 10


def get_highest_prob_horse(race_data: Dict[str, Any]) -> str:
    """Get the horse number with lowest odds (highest probability)."""
    if "rapports" not in race_data:
        return None

    latest_odds = {}
    for horse, odds_history in race_data["rapports"].items():
        if odds_history:
            latest_timestamp = max(odds_history.keys())
            latest_odds[horse] = odds_history[latest_timestamp]

    if not latest_odds:
        return None

    return min(latest_odds.items(), key=lambda x: x[1])[0]


def compute_category_stats(data_dir: Path = Path("data/2024")) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    """
    Compute mean target value for each category in hippodrome, discipline, and specialite.
    Also compute the total number of races per hippodrome.
    Returns:
        - Dictionary of dictionaries with the mean target values
        - Dictionary with total races per hippodrome
    """
    print(f"\nComputing statistics from {data_dir}...")

    # Initialize dictionaries to store counts and sums
    stats = {
        "hippodrome": {"counts": {}, "wins": {}},
        "discipline": {"counts": {}, "wins": {}},
        "specialite": {"counts": {}, "wins": {}},
    }

    # Initialize dictionary for total races per hippodrome (including invalid races)
    hippodrome_total_races = {}

    total_races = 0
    valid_races = 0

    # Process all files
    for file in sorted(data_dir.glob("*.json")):
        with open(file, "r") as f:
            day_data = json.load(f)

            for race_data in day_data.values():
                total_races += 1

                # Count total races per hippodrome (including invalid ones)
                hippodrome = race_data.get("hippodrome", {}).get("libelleCourt", "UNKNOWN")
                hippodrome_total_races[hippodrome] = hippodrome_total_races.get(hippodrome, 0) + 1

                # Skip if race is not finished
                if (
                    "ordreArrivee" not in race_data
                    or "rapportsDefinitifs" not in race_data
                    or "E_SIMPLE_GAGNANT" not in race_data["rapportsDefinitifs"]
                ):
                    continue

                favorite_horse = get_highest_prob_horse(race_data)
                if not favorite_horse:
                    continue

                valid_races += 1

                # Get target (1 if favorite won, 0 otherwise)
                winner = str(race_data["ordreArrivee"][0][0])
                target = 1 if favorite_horse == winner else 0

                # Update counts and sums for each category
                discipline = race_data.get("discipline", "UNKNOWN")
                specialite = race_data.get("specialite", "UNKNOWN")

                for category, value in [
                    ("hippodrome", hippodrome),
                    ("discipline", discipline),
                    ("specialite", specialite),
                ]:
                    stats[category]["counts"][value] = stats[category]["counts"].get(value, 0) + 1
                    stats[category]["wins"][value] = stats[category]["wins"].get(value, 0) + target

    print(f"Processed {total_races} total races, {valid_races} were valid for analysis")

    # Compute mean values and create final dictionaries
    means = {}
    for category in stats:
        means[category] = {}
        for value in stats[category]["counts"]:
            count = stats[category]["counts"][value]
            wins = stats[category]["wins"][value]
            means[category][value] = (
                wins / count if count > MIN_RACES_COUNT else 0.5
            )  # Use 0.5 as default for unseen categories

    # Print statistics
    print("\nCategory Statistics:")
    for category, values in means.items():
        print(f"\n{category.title()} win rates:")
        sorted_items = sorted(values.items(), key=lambda x: x[1], reverse=True)
        for value, mean in sorted_items:
            count = stats[category]["counts"][value]
            print(f"{value}: {mean:.3f} (based on {count} races)")

    print("\nHippodrome race counts:")
    sorted_hippodromes = sorted(hippodrome_total_races.items(), key=lambda x: x[1], reverse=True)
    for hippodrome, count in sorted_hippodromes:
        print(f"{hippodrome}: {count} races")

    # Save the results
    output_dir = Path("epmu/model/data")
    output_dir.mkdir(exist_ok=True)
    joblib.dump(means, output_dir / "category_stats.joblib")
    joblib.dump(hippodrome_total_races, output_dir / "hippodrome_race_counts.joblib")

    return means, hippodrome_total_races


if __name__ == "__main__":
    # Only compute stats for 2024 (training) data
    stats, race_counts = compute_category_stats(Path("data/2024"))
