import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_benchmark_results(input_path="results/benchmark_results.csv",
                           output_path="results/benchmark_plot.png"):
    """
    Load benchmark results and save a simple comparison plot.
    """
    df = pd.read_csv(input_path)

    df = df.sort_values("score", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["score"])
    plt.title("Model Benchmark Comparison")
    plt.xlabel("Model")
    plt.ylabel("Cross-Validation Score")
    plt.xticks(rotation=20)
    plt.tight_layout()

    Path("results").mkdir(exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    plot_benchmark_results()
