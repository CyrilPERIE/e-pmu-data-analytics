import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def load_data():
    """Load the prepared dataset."""
    data_dir = Path("epmu/model/data")
    X = pd.read_csv(data_dir / "features.csv")
    y = pd.read_csv(data_dir / "targets.csv").iloc[:, 0]
    return X, y

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot top N feature importances."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    importances[:top_n].plot(kind='bar')
    plt.title(f'Top {top_n} Most Important Features')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("images/feature_importance.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png")
    plt.close()

def train_model():
    """Train and evaluate the model."""
    print("Loading data...")
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features while maintaining DataFrame structure
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    # Perform cross-validation
    print("\nPerforming cross-validation...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Train model
    print("\nTraining final model...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot feature importance
    print("\nPlotting feature importance...")
    plot_feature_importance(model, X.columns)
    
    # Plot confusion matrix
    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    os.makedirs("epmu/model/saved_models", exist_ok=True)
    joblib.dump(model, "epmu/model/saved_models/random_forest.joblib")
    joblib.dump(scaler, "epmu/model/saved_models/scaler.joblib")
    
    return model, scaler

def predict_race(model, scaler, race_features):
    """Predict the probability of favorite horse winning for a new race."""
    # Scale features
    features_scaled = scaler.transform([race_features])
  
    # Get probability of favorite horse winning
    prob_win = model.predict_proba(features_scaled)[0][1]
    
    return prob_win

if __name__ == "__main__":
    model, scaler = train_model() 