import json
from pathlib import Path
from collections import defaultdict


def extract_categories_from_dir(data_dir: Path) -> dict:
    """Extract all unique categories from a data directory."""
    categories = defaultdict(set)

    # Features to extract categories from
    features = ["discipline", "specialite", "conditionSexe", "nature"]

    for file in data_dir.glob("*.json"):
        with open(file, "r") as f:
            day_data = json.load(f)

            for race_data in day_data.values():
                # Extract main features
                for feature in features:
                    if feature in race_data:
                        categories[feature].add(race_data[feature])

                # Extract weather feature
                if "meteo" in race_data and "nebulositeCode" in race_data["meteo"]:
                    categories["nebulosite"].add(race_data["meteo"]["nebulositeCode"])

    return categories


def main():
    # Extract categories from both directories
    categories_2023 = extract_categories_from_dir(Path("data/2023"))
    categories_2024 = extract_categories_from_dir(Path("data/2024"))

    # Merge categories
    all_categories = defaultdict(set)
    for year_categories in [categories_2023, categories_2024]:
        for feature, values in year_categories.items():
            all_categories[feature].update(values)

    # Convert sets to sorted lists and add UNKNOWN
    final_categories = {k: sorted(list(v)) + ["UNKNOWN"] for k, v in all_categories.items()}

    # Print in a format ready to copy-paste into prepare_data.py
    print("CATEGORIES = {")
    for feature, values in final_categories.items():
        print(f'    "{feature}": [')
        for value in values:
            print(f'        "{value}",')
        print("    ],")
    print("}")

    # Print some statistics
    print("\nStatistics:")
    print("Number of categories per feature:")
    for feature, values in final_categories.items():
        print(f"{feature}: {len(values)} categories")

    print("\nCategories found in 2023 but not in 2024:")
    for feature in all_categories:
        diff = categories_2023[feature] - categories_2024[feature]
        if diff:
            print(f"{feature}: {sorted(list(diff))}")

    print("\nCategories found in 2024 but not in 2023:")
    for feature in all_categories:
        diff = categories_2024[feature] - categories_2023[feature]
        if diff:
            print(f"{feature}: {sorted(list(diff))}")


if __name__ == "__main__":
    main()
