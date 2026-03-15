from models import get_models
from data_loader import load_dataset
from evaluate import evaluate_model

import pandas as pd
from pathlib import Path


def run_benchmark():

    X, y = load_dataset()

    models = get_models()

    results = []

    print("Benchmark Results")
    print("-----------------")

    for name, model in models.items():

        score = evaluate_model(model, X, y)

        print(f"{name}: {score:.4f}")

        results.append({
            "model": name,
            "score": score
        })

    results_df = pd.DataFrame(results)

    Path("results").mkdir(exist_ok=True)

    results_df.to_csv("results/benchmark_results.csv", index=False)

    print("\nResults saved to results/benchmark_results.csv")


if __name__ == "__main__":
    run_benchmark()
