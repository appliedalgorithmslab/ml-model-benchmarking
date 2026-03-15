from models import get_models
from data_loader import load_dataset
from evaluate import evaluate_model


def run_benchmark():

    X, y = load_dataset()

    models = get_models()

    print("Benchmark Results")
    print("-----------------")

    for name, model in models.items():

        score = evaluate_model(model, X, y)

        print(f"{name}: {score:.4f}")


if __name__ == "__main__":
    run_benchmark()
