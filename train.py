"""
Train an XGBoost classifier on the Seaborn penguins dataset,
evaluate its performance, and save both the model and metadata

"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier

def main() -> None:
    """
    Load penguins data, preprocess features/labels, train an XGBoost model,
    evaluate performance, and save both the model and metadata.
    """
    # 1. Load & clean data
    df: pd.DataFrame = sns.load_dataset("penguins")
    df = df.dropna()
    y_raw: pd.Series = df["species"]
    X_raw: pd.DataFrame = df.drop(columns=["species"])

    # 2. Encode features & labels
    X: pd.DataFrame = pd.get_dummies(X_raw, columns=["sex", "island"], prefix=["sex", "island"])
    label_encoder: LabelEncoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # 3. Split into train/test sets (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Train XGBoost classifier (with overfitting prevention)
    model: XGBClassifier = XGBClassifier(
        max_depth=3,
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 5. Evaluate & report
    print("=== Training set performance ===")
    print(classification_report(y_train, model.predict(X_train)))
    print("Train F1 (weighted):", f1_score(y_train, model.predict(X_train), average='weighted'))

    print("=== Test set performance ===")
    print(classification_report(y_test,  model.predict(X_test)))
    print("Test F1 (weighted):", f1_score(y_test,  model.predict(X_test), average='weighted'))

    # 6. Save model & metadata
    output_dir = Path("app/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.json"
    model.save_model(str(model_path))
    print(f"Model saved to {model_path}")

    metadata: Dict[str, List[str]] = {
        "feature_columns": X.columns.tolist(),
        "label_classes":  label_encoder.classes_.tolist()
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    main()
