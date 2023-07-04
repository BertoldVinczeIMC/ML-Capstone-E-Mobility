from __future__ import annotations

import polars as pl
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt


def check_data(data: pl.DataFrame) -> None:
    # check if there are any null values in Dataframe
    print(data.shape)
    data = data.drop_nulls()
    print(data.shape)


def decision_tree_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )

    # leave the test data untouched until the very end

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=2)

    # convert the data to numpy arrays
    X_train = X_train
    X_test = X_test
    rf.fit(X_train, y_train)

    rf.predict(X_test)

    # compare the predictions with the actual values of score

    accuracy_score_test = accuracy_score(y_test, rf.predict(X_test))
    accuracy_score_train = accuracy_score(y_train, rf.predict(X_train))

    print("Accuracy score on test data: ", accuracy_score_test)
    print("Accuracy score on train data: ", accuracy_score_train)

    # find the most important features
    feature_importances = pd.DataFrame(
        rf.feature_importances_, index=X.columns, columns=["importance"]
    ).sort_values("importance", ascending=False)

    # delete the features with importance less than 0.01
    feature_importances = feature_importances[feature_importances.importance > 0.01]

    # These ones were the features with importance less than 0.01
    """
    LEM.Ladebox3.P
    LEM.Ladebox2.P
    LEM.Ladebox1.P

    """
    # plot the feature importance
    plt.figure(figsize=(10, 7))
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.barh(feature_importances.index, feature_importances.importance)
    plt.show()

    # plot the confusion matrix native to sklearn
    plt.figure(figsize=(10, 7))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.matshow(confusion_matrix(y_test, rf.predict(X_test)), cmap="Blues")
    plt.show()
