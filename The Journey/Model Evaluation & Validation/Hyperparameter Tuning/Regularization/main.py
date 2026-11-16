import os
import numpy as np
import pandas as pd

from bynat.cleaning import BaseData

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))


# Load and prepare data
df = pd.read_csv(os.path.join(current_dir, "Melbourne_housing_FULL.csv"))

df = df.dropna(subset=["Price"])

df.drop(
    ["Propertycount", "Regionname", "Lattitude", "Longtitude"],
    axis=1,
    inplace=True
)

df[["Bathroom", "Bedroom2"]] = (
    df[["Bathroom", "Bedroom2"]].replace(0, np.nan)
)

df.loc[
    (df["Landsize"] == 0)
    & df["Type"].isin(["h", "t"]),
    "Landsize"
] = np.nan


# Split data
train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

train = BaseData(train)
test = BaseData(test)


# Hierarchical imputation configuration
imputation_config = [
    (
        "CouncilArea",
        "mode",
        ["Suburb"]
    ),
    (
        "YearBuilt",
        "median",
        ["CouncilArea", "Suburb", "Distance", "Address"]
    ),
    (
        ["Bathroom", "Bedroom2", "Car"],
        "median",
        ["CouncilArea", "Suburb", "Rooms", "Type"]
    ),
    (
        ["Landsize", "BuildingArea"],
        "median",
        ["CouncilArea", "Suburb", "YearBuilt", "Rooms", "Type"]
    )
]



MIN_SAMPLES = 5

for targets, strategy, hierarchy_cols in imputation_config:

    targets = [targets] if isinstance(targets, str) else targets

    train = train.fill_missing(
        *targets,
        strategy=strategy,
        hierarchy_cols=hierarchy_cols,
        min_samples=MIN_SAMPLES,
        fallback_global=True
    )

    test = test.fill_missing_from(
        train,
        *targets,
        strategy=strategy,
        hierarchy_cols=hierarchy_cols,
        min_samples=MIN_SAMPLES,
        fallback_global=True
    )


# Remove columns no longer needed
train.drop(columns=["Address"], inplace=True)
test.drop(columns=["Address"], inplace=True)


# Feature and target separation
X_train = train.drop(columns=["Price"])
y_train = train["Price"]

X_test = test.drop(columns=["Price"])
y_test = test["Price"]


# Identify numerical and categorical columns
numeric_cols = X_train.select_dtypes(
    include=np.number
).columns

categorical_cols = X_train.select_dtypes(
    include=["object", "str", "category"]
).columns


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler(feature_range=(0, 1)))
            ]),
            numeric_cols
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ))
            ]),
            categorical_cols
        )
    ]
)


# Models and hyperparameter grids
models = {
    "Linear Regression": {
        "model": LinearRegression(),
        "parameters": {}
    },
    "Lasso": {
        "model": Lasso(max_iter=10000),
        "scale_target": True,
        "parameters": {
            "model__regressor__alpha": [0.00001, 0.0001, 0.001, 0.01],
        }
    },
    "Ridge": {
        "model": Ridge(),
        "parameters": {
            "model__alpha": [0.1, 1.0, 10, 100],
        }
    }
}


scores = []
best_models = {}

for model_name, config in models.items():

    reg = config["model"]
    if config.get("scale_target"):
        reg = TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", reg)
    ])

    clf = GridSearchCV(
        estimator=pipe,
        param_grid=config["parameters"],
        cv=3,
        scoring="r2",
        n_jobs=4,
        pre_dispatch=1,
        verbose=2
    )

    clf.fit(X_train, y_train)

    best_models[model_name] = clf.best_estimator_

    scores.append(
        pd.DataFrame({
            "Model": model_name,
            "Parameters": clf.cv_results_["params"],
            "Mean CV Score": clf.cv_results_["mean_test_score"],
            "Std CV Score": clf.cv_results_["std_test_score"]
        })
    )

    print(f"\n{model_name}")
    print("Best Parameters:", clf.best_params_)
    print("Best CV Score:", clf.best_score_)


scores_df = pd.concat(
    scores,
    ignore_index=True
).sort_values(
    "Mean CV Score",
    ascending=False
)

print("\nGrid Search Results:")
print(scores_df)


test_scores = []

for model_name, model in best_models.items():

    predictions = model.predict(X_test)

    test_scores.append({
        "Model": model_name,
        "Test R2": model.score(X_test, y_test),
        "Test MAE": np.mean(np.abs(y_test - predictions)),
        "Test RMSE": np.sqrt(
            np.mean((y_test - predictions) ** 2)
        )
    })


test_scores_df = pd.DataFrame(test_scores)

print("\nTest Results:")
print(test_scores_df)