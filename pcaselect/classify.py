"""Test accuracy of simple classifiers on the first k PCA scores."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

CLASSIFIERS = {"1nn": "1-nearest neighbour", "logreg": "logistic regression"}


def make_classifier(name: str):
    if name == "1nn":
        return KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    if name == "logreg":
        return LogisticRegression(max_iter=5000)
    raise ValueError(f"unknown classifier {name!r}; choose from {list(CLASSIFIERS)}")


def accuracy_curve(z_train, y_train, z_test, y_test, ks, name: str) -> np.ndarray:
    """Accuracy on the test set of classifier ``name`` trained on the first k scores, for each k in ``ks``."""
    return np.array([
        make_classifier(name).fit(z_train[:, :k], y_train).score(z_test[:, :k], y_test) for k in ks
    ])
