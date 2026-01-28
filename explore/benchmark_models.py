"""
Comprehensive Benchmark Script for Guardrail Models

This script evaluates guardrail models on multiple datasets to measure
true generalization performance. It compares:
1. Individual trained models (SVC, XGBoost, etc.)
2. The new ensemble model
3. Cross-dataset performance

Usage:
    python benchmark_models.py [--model MODEL_PATH] [--all]
"""

import pathlib
import joblib
import argparse
import warnings
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
)
from app.rag.vectorizer import Vectorizer
from app.rag.guardrail import FeatureExtractor
warnings.filterwarnings("ignore")


def load_benchmark_datasets():
    """Load all benchmark datasets."""
    datasets = {}
    base_path = pathlib.Path(__file__).parent

    # 1. Qualifire English
    print("Loading Qualifire English...")
    df = pd.read_csv(base_path / "prompt-injections-benchmark.csv")
    df["label"] = df["label"].map({"jailbreak": 1, "benign": 0})
    datasets["qualifire_en"] = {
        "texts": df["text"].tolist(),
        "labels": df["label"].tolist(),
    }
    print(f"  Loaded {len(df)} samples")

    # 2. Qualifire French
    print("Loading Qualifire French...")
    df = pd.read_csv(base_path / "prompt-injections-benchmark-fr.csv")
    df["label"] = df["label"].map({"jailbreak": 1, "benign": 0})
    datasets["qualifire_fr"] = {
        "texts": df["text"].tolist(),
        "labels": df["label"].tolist(),
    }
    print(f"  Loaded {len(df)} samples")

    # 3. Deepset
    print("Loading Deepset...")
    deepset = load_dataset("deepset/prompt-injections", split="train")
    datasets["deepset"] = {"texts": deepset["text"], "labels": deepset["label"]}
    print(f"  Loaded {len(deepset)} samples")

    # 4. Custom dataset (jailbreak + helpful instructions)
    print("Loading Custom dataset...")
    jailbreak = pd.read_csv(base_path / "jailbreak_prompts.csv")
    benign = load_dataset("HuggingFaceH4/helpful-instructions", split="train")

    # Sample for faster evaluation
    jb_sample = jailbreak.sample(n=min(2000, len(jailbreak)), random_state=42)
    benign_sample = list(benign["instruction"])[:2000]

    custom_texts = list(jb_sample["Prompt"]) + benign_sample
    custom_labels = [1] * len(jb_sample) + [0] * len(benign_sample)
    datasets["custom"] = {"texts": custom_texts, "labels": custom_labels}
    print(f"  Loaded {len(custom_texts)} samples")

    # 5. Medical consultations (benign + medical injection attacks)
    medical_path = base_path / "medical_consultations_benign.csv"
    if medical_path.exists():
        print("Loading Medical consultations...")
        df = pd.read_csv(medical_path)
        datasets['medical'] = {'texts': df['text'].tolist(), 'labels': df['label'].tolist()}
        benign_count = (df['label'] == 0).sum()
        injection_count = (df['label'] == 1).sum()
        print(f"  Loaded {len(df)} samples (benign: {benign_count}, injection: {injection_count})")
    else:
        print(f"  Skipping medical consultations (file not found: {medical_path})")
        print("  Run 'python create_medical_consultation_dataset.py' to generate it.")

    return datasets


def evaluate_model(model, X, y_true, model_name="Model"):
    """Evaluate a model on given features."""
    y_pred = model.predict(X)

    try:
        y_prob = model.predict_proba(X)[:, 1]
        has_proba = True
    except AttributeError:
        y_prob = None
        has_proba = False

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    if has_proba:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except AttributeError:
            metrics["roc_auc"] = 0.0

    return metrics


def benchmark_old_models(datasets, vectorizer_en, vectorizer_multi):
    """Benchmark the old models from previous notebooks."""
    base_path = pathlib.Path(__file__).parent
    results = []

    old_models = [
        ("svc_model_en.joblib", "SVC English", vectorizer_en),
        ("svc_model_multi.joblib", "SVC Multi-lang", vectorizer_multi),
        ("svc_custom.joblib", "SVC Custom", vectorizer_en),
        ("xgboost_custom.joblib", "XGBoost Custom", vectorizer_en),
    ]

    for model_file, model_name, vectorizer in old_models:
        model_path = base_path / model_file
        if not model_path.exists():
            print(f"  Skipping {model_name} (file not found)")
            continue

        print(f"\nBenchmarking {model_name}...")
        model = joblib.load(model_path)

        for dataset_name, data in datasets.items():
            print(f"  Testing on {dataset_name}...", end=" ")

            # Generate embeddings
            texts = [str(t) for t in data["texts"]]
            X = np.array(vectorizer.generate_embeddings(texts))
            y = data["labels"]

            # Evaluate
            metrics = evaluate_model(model, X, y, model_name)
            metrics["model"] = model_name
            metrics["dataset"] = dataset_name
            results.append(metrics)

            print(f"F1: {metrics['f1']:.4f}")

    return results


def benchmark_new_ensemble(datasets, vectorizer, feature_extractor, scaler, model):
    """Benchmark the new ensemble model."""
    results = []

    print("\nBenchmarking Ensemble Model...")

    for dataset_name, data in datasets.items():
        print(f"  Testing on {dataset_name}...", end=" ")

        texts = [str(t) for t in data["texts"]]

        # Generate embeddings
        embeddings = np.array(vectorizer.generate_embeddings(texts))

        # Extract handcrafted features
        handcrafted = feature_extractor.extract_batch(texts)
        handcrafted_scaled = scaler.transform(handcrafted)

        # Combine features
        X = np.hstack([embeddings, handcrafted_scaled])
        y = data["labels"]

        # Evaluate
        metrics = evaluate_model(model, X, y, "Ensemble")
        metrics["model"] = "Ensemble v2"
        metrics["dataset"] = dataset_name
        results.append(metrics)

        print(f"F1: {metrics['f1']:.4f}")

    return results


def print_comparison_table(results):
    """Print a comparison table of all results."""
    df = pd.DataFrame(results)

    # Pivot table for easy comparison
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    # Group by dataset
    for dataset in df["dataset"].unique():
        print(f"\n--- {dataset} ---")
        subset = df[df["dataset"] == dataset][
            ["model", "accuracy", "precision", "recall", "f1", "mcc"]
        ]
        subset = subset.sort_values("f1", ascending=False)
        print(subset.to_string(index=False))

    # Overall averages
    print("\n" + "=" * 80)
    print("AVERAGE PERFORMANCE ACROSS ALL DATASETS")
    print("=" * 80)
    avg = df.groupby("model")[["accuracy", "precision", "recall", "f1", "mcc"]].mean()
    avg = avg.sort_values("f1", ascending=False)
    print(avg.to_string())

    # Best model per dataset
    print("\n" + "=" * 80)
    print("BEST MODEL PER DATASET (by F1)")
    print("=" * 80)
    for dataset in df["dataset"].unique():
        subset = df[df["dataset"] == dataset]
        best = subset.loc[subset["f1"].idxmax()]
        print(f"{dataset:20s}: {best['model']:20s} (F1: {best['f1']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Benchmark guardrail models")
    parser.add_argument(
        "--all", action="store_true", help="Benchmark all available models"
    )
    parser.add_argument(
        "--old-only", action="store_true", help="Only benchmark old models"
    )
    parser.add_argument(
        "--new-only", action="store_true", help="Only benchmark new ensemble"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("GUARDRAIL MODEL BENCHMARK")
    print("=" * 70)

    # Load datasets
    print("\n[1/3] Loading benchmark datasets...")
    datasets = load_benchmark_datasets()

    # Initialize vectorizers
    print("\n[2/3] Initializing vectorizers...")
    vectorizer_en = Vectorizer(model_name="all-MiniLM-L6-v2")
    vectorizer_multi = Vectorizer(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    print("  Loaded English vectorizer (all-MiniLM-L6-v2)")
    print("  Loaded Multi-lang vectorizer (paraphrase-multilingual-MiniLM-L12-v2)")

    all_results = []

    # Benchmark old models
    if not args.new_only:
        print("\n[3/3] Benchmarking models...")
        old_results = benchmark_old_models(datasets, vectorizer_en, vectorizer_multi)
        all_results.extend(old_results)

    # Benchmark new ensemble (if available)
    models_path = pathlib.Path(__file__).parent.parent / "data" / "ml_models"
    ensemble_path = models_path / "guardrail_ensemble_v2.joblib"

    if ensemble_path.exists() and not args.old_only:
        print("\nLoading new ensemble model...")
        ensemble_model = joblib.load(ensemble_path)
        scaler = joblib.load(models_path / "guardrail_scaler_v2.joblib")
        feature_extractor = joblib.load(
            models_path / "guardrail_feature_extractor_v2.joblib"
        )

        new_results = benchmark_new_ensemble(
            datasets, vectorizer_multi, feature_extractor, scaler, ensemble_model
        )
        all_results.extend(new_results)
    elif not args.old_only:
        print(
            "\n[!] New ensemble model not found. Run improved_guardrail_training.py first."
        )

    # Print comparison
    if all_results:
        print_comparison_table(all_results)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
