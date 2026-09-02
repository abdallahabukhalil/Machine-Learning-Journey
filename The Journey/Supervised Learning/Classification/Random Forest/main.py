import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

digits_object = load_digits()

df = pd.concat(
    [
        pd.DataFrame(digits_object.data, columns = digits_object.feature_names),
        pd.DataFrame(digits_object.target, columns = ['digit'])
    ],
    axis = 1
)

train, test = train_test_split(df, test_size = 0.2, random_state = 42, stratify = df['digit'])

model = RandomForestClassifier(criterion = 'gini', min_samples_split = 2, n_estimators = 15, max_depth = 12, random_state = 42)

model.fit(train.drop('digit', axis = 'columns'), train['digit'])

print(test.head())
print(f"prediction: {model.predict(test.drop('digit', axis = 'columns').head())}")

print(f"Score: {model.score(test.drop('digit', axis = 'columns'), test['digit'])}")
print(df['digit'].value_counts(normalize = True))

# score: ~96.39%