import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

wine = load_wine()

df = pd.concat(
    [
        pd.DataFrame(wine.data, columns = wine.feature_names),
        pd.DataFrame(wine.target, columns = ["wine_type"])
    ],
    axis = 1
)

train, test = train_test_split(df, test_size = 0.2, stratify = df['wine_type'], random_state = 42)

scaler = MinMaxScaler(feature_range = (0, 1))
scaled_train = pd.concat(
    [
        pd.DataFrame(
            scaler.fit_transform(
                train.drop('wine_type', axis = 1),
            ),
            columns = train.drop('wine_type', axis = 1).columns,
            index = train.index
        ),
        train['wine_type']
    ],
    axis = 1
)
    


model = MultinomialNB()
model.fit(scaled_train.drop('wine_type', axis = 1), scaled_train['wine_type'])

test_scaled = pd.concat(
    [
        pd.DataFrame(
            scaler.transform(
                test.drop('wine_type', axis = 1)
            ),
            columns =  test.drop('wine_type', axis = 1).columns,
            index =  test.drop('wine_type', axis = 1).index
        ),
        test['wine_type']
    ],
    axis = 1
)

print(
    test_scaled.head().to_string(),
    f"Score: {model.score(test_scaled.drop('wine_type', axis = 1), test_scaled['wine_type'])}", # Score: ~94.44%
    f"Predictions: {model.predict(test_scaled.drop('wine_type', axis = 1).head())}",
    df['wine_type'].value_counts(normalize = True),
    sep = "\n"
)
