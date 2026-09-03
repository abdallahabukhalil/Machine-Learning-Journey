import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris = load_iris()

df = pd.concat(
    [
        pd.DataFrame(iris.data, columns = iris.feature_names),
        pd.DataFrame(iris.target, columns = ['target'])
    ],
    axis = 1
)

kf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)


models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(random_state = 42)
}

params = {
    'Logistic Regression': {
        'max_iter': [200, 300, 400]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    'Random Forest': {
        'criterion': ['gini', 'entropy'],
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
    }
}

X = df.drop('target', axis = 1)
y = df['target']

results = []

for model_name, model in models.items():
    best_score = 0
    best_params = {}

    combinations = [{}]

    for key, values in params[model_name].items():
        new_combinations = []

        for combo in combinations:
            for value in values:
                new_combo = combo.copy()
                new_combo[key] = value

                new_combinations.append(new_combo)

        combinations = new_combinations

    for combo in combinations:
        model.set_params(**combo)

        scores = cross_val_score(model, X, y, cv = kf)
        mean_score = scores.mean()

        if mean_score > best_score:
            best_score = mean_score
            best_params = combo

    results.append({
        'model': model_name,
        'Best Parameters': best_params,
        'Best Score': best_score
    })

results_df = pd.DataFrame(results)
print(results_df.to_string())