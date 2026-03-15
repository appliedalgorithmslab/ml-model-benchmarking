from sklearn.model_selection import cross_val_score


def evaluate_model(model, X, y, cv=5):
    """
    Evaluate a model using cross-validation.
    """

    scores = cross_val_score(model, X, y, cv=cv)

    return scores.mean()
