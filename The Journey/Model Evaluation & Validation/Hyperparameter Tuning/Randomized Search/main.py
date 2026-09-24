import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

iris = load_iris()

df = pd.concat(
    [
        pd.DataFrame(iris.data, columns = iris.feature_names),
        pd.DataFrame(iris.target, columns = ['iris_type'])
    ],
    axis = 1
)

scaler = MinMaxScaler(feature_range = (0, 1))

models = {
    "Logistic Regression": {
        "model": LogisticRegression(random_state = 42),
        "parameters": {
            "max_iter": [50, 100, 200]
        }
    },
    "SVM": {
        "model": SVC(random_state = 42),
        "parameters": {
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
            "C": [0.1, 1, 10, 100],
            "gamma": ['scale', 'auto'],
            "max_iter": [50, 100, 200]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state = 42),
        'parameters': {
            "criterion": ['gini', 'entropy', 'log_loss'],
            "n_estimators": [10, 14, 30, 50, 100],
            "max_depth": [None, 1, 2, 4, 7, 10, 20],
            "min_samples_split": [2, 3, 4],
            "min_samples_leaf": [2, 3, 4],
            "max_features": ['sqrt', 'log2']
        }
    },
    "Naive Bayes": {
        "model": MultinomialNB(),
        "parameters": {

        }
    }
}

scores = []
for model, config in models.items():
    pipe = Pipeline([
        ("scaler", scaler),
        ("model", config["model"])
    ])

    parameters = {
        f"model__{param}": values for param, values in config["parameters"].items()
    }

    clf = RandomizedSearchCV(
        pipe,
        parameters,
        cv = 5,
        n_iter = 15,
    )

    clf.fit(df.drop("iris_type", axis = 1), df["iris_type"])

    scores.append(
        pd.DataFrame({
            'Model': model,
            'Parameters': clf.cv_results_['params'],
            'Mean CV Score': clf.cv_results_['mean_test_score'],
            'Std CV Score': clf.cv_results_['std_test_score']
        })
    )

scores_df = pd.concat(scores, ignore_index = True)

print(scores_df.to_string())