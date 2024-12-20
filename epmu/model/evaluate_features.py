import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_data():
    """Load the prepared dataset."""
    data_dir = Path("epmu/model/data")
    X = pd.read_csv(data_dir / "features.csv")
    y = pd.read_csv(data_dir / "targets.csv").iloc[:, 0]  # Get first column
    return X, y


def compute_correlations(X: pd.DataFrame, y: pd.Series):
    """Compute correlations between features and target."""
    # Add target to features for correlation computation
    data = X.copy()
    data["target"] = y

    # Compute correlations with target
    correlations = data.corr()["target"].sort_values(ascending=False)

    return correlations


def plot_correlations(correlations: pd.Series, top_n: int = 20):
    """Plot top N correlations."""
    plt.figure(figsize=(12, 8))

    # Get top N positive and negative correlations (excluding target itself)
    top_positive = correlations[1 : top_n + 1]  # Skip first as it's target itself
    top_negative = correlations[-(top_n):]

    # Combine them for plotting
    top_correlations = pd.concat([top_positive, top_negative])

    # Create bar plot
    bars = plt.barh(range(len(top_correlations)), top_correlations.values)

    # Color positive and negative correlations differently
    for i, bar in enumerate(bars):
        if bar.get_width() < 0:
            bar.set_color("red")
        else:
            bar.set_color("blue")

    # Add feature names as y-axis labels
    plt.yticks(range(len(top_correlations)), top_correlations.index, fontsize=8)

    plt.title(f"Top {top_n} Positive and Negative Correlations with Target")
    plt.xlabel("Correlation Coefficient")

    # Add vertical line at x=0
    plt.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("images/correlation_analysis.png")
    plt.close()


def analyze_correlations(top_n: int = 20):
    """Main function to analyze correlations."""
    print("Loading data...")
    X, y = load_data()

    print("Computing correlations...")
    correlations = compute_correlations(X, y)

    print("\nTop positive correlations with target:")
    print(correlations[1 : top_n + 1])  # Skip first as it's target itself

    print("\nTop negative correlations with target:")
    print(correlations[-(top_n):])

    print("\nPlotting correlations...")
    plot_correlations(correlations, top_n)

    print("\nAnalysis complete! Check 'images/correlation_analysis.png' for visualization.")


if __name__ == "__main__":
    analyze_correlations()
