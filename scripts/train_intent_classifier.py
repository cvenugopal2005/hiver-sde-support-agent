"""Step 10: first baseline intent classifier.

Trains TF-IDF + Logistic Regression, plus a majority-class baseline, on
data/processed/intent_training_sample_silver.csv (Step 9's LLM-assisted
SILVER labels).

IMPORTANT SCOPE NOTE: these are silver labels, not human ground truth, and
NOT the 150-250-example hand-labeled golden evaluation set the assignment
separately requires. All metrics below measure fit to these silver labels
only -- they are not a claim about real-world classifier accuracy.

UNCLEAR rows are excluded from training/evaluation: UNCLEAR is a labeling
escape hatch ("insufficient evidence to classify"), not one of the 8 real
taxonomy intents a production classifier should route messages into.

No hyperparameter tuning is performed (per instructions: a simple first
baseline, not an over-tuned model). The validation split exists so a later
step can tune against it without touching the test set; it is not used for
tuning here.

Outputs:
    data/processed/intent_classifier.joblib        -- trained sklearn Pipeline
    data/processed/intent_classifier_results.json  -- full metrics

Usage:
    python scripts/train_intent_classifier.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
IN_CSV = DATA_DIR / "intent_training_sample_silver.csv"
MODEL_PATH = DATA_DIR / "intent_classifier.joblib"
RESULTS_PATH = DATA_DIR / "intent_classifier_results.json"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load_data() -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(IN_CSV)
    before = len(df)
    df = df[df["intent"] != "UNCLEAR"].reset_index(drop=True)
    return df, before - len(df)


def split_data(df: pd.DataFrame):
    X, y = df["conversation_context"], df["intent"]
    # 70/15/15: split off 30% first, then split that 30% in half (stratified
    # both times so every class -- including the smallest, Messaging &
    # Communication -- appears in all three splits).
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def counts_dict(y: pd.Series, labels: list[str]) -> dict[str, int]:
    vc = y.value_counts()
    return {label: int(vc.get(label, 0)) for label in labels}


def evaluate(y_true, y_pred, labels: list[str]) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def main() -> None:
    section("LOADING data/processed/intent_training_sample_silver.csv (Step 9 SILVER labels)")
    df, n_excluded = load_data()
    print(f"rows: {len(df)} (excluded {n_excluded} UNCLEAR rows -- not a real taxonomy intent)")

    labels = sorted(df["intent"].unique())
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(df)

    section("SPLIT COUNTS (stratified 70/15/15)")
    for name, y in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        print(f"{name}: {len(y)}")

    section("CLASS DISTRIBUTION PER SPLIT")
    dist = {name: counts_dict(y, labels) for name, y in [("train", y_train), ("validation", y_val), ("test", y_test)]}
    dist_df = pd.DataFrame(dist).reindex(labels)
    print(dist_df.to_string())

    section("TRAINING: MAJORITY-CLASS BASELINE (DummyClassifier)")
    majority = DummyClassifier(strategy="most_frequent", random_state=SEED)
    majority.fit(X_train, y_train)
    maj_pred = majority.predict(X_test)
    maj_metrics = evaluate(y_test, maj_pred, labels)
    print(f"predicts: {maj_pred[0]!r} for every input")
    print(f"accuracy={maj_metrics['accuracy']:.3f}  macro_f1={maj_metrics['macro_f1']:.3f}  "
          f"weighted_f1={maj_metrics['weighted_f1']:.3f}")

    section("TRAINING: TF-IDF + LOGISTIC REGRESSION")
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
        ]
    )
    pipeline.fit(X_train, y_train)
    clf_pred = pipeline.predict(X_test)
    clf_metrics = evaluate(y_test, clf_pred, labels)
    print(f"accuracy={clf_metrics['accuracy']:.3f}  macro_f1={clf_metrics['macro_f1']:.3f}  "
          f"weighted_f1={clf_metrics['weighted_f1']:.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nmodel saved to: {MODEL_PATH}")

    section("PER-CLASS REPORT (TF-IDF + LogReg, test set)")
    print(classification_report(y_test, clf_pred, labels=labels, zero_division=0))

    section("CONFUSION MATRIX (TF-IDF + LogReg, test set) -- rows=true, cols=predicted")
    print("labels order:", labels)
    print(confusion_matrix(y_test, clf_pred, labels=labels))

    results = {
        "note": (
            "Trained on Step 9 LLM-assisted SILVER labels, NOT the Hiver golden evaluation "
            "set. UNCLEAR rows excluded (not a real taxonomy intent)."
        ),
        "labels": labels,
        "excluded_unclear_rows": n_excluded,
        "split_sizes": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "class_distribution": dist,
        "majority_baseline": maj_metrics,
        "tfidf_logreg": clf_metrics,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults saved to: {RESULTS_PATH}")

    section("SUMMARY: CLASSIFIER vs MAJORITY BASELINE (test set)")
    print(f"{'metric':15s} {'majority':>10s} {'tfidf+logreg':>14s}")
    for k in ("accuracy", "macro_f1", "weighted_f1"):
        print(f"{k:15s} {maj_metrics[k]:>10.3f} {clf_metrics[k]:>14.3f}")


if __name__ == "__main__":
    main()
