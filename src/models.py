from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def get_models(random_state=42):
    """
    Returns dictionary of models used in benchmarking.
    """

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state
        ),
        "svm": SVC(probability=True, random_state=random_state),
    }

    return models
