import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Test which method of encoding salary is better for our case here.
# representing it as an 'Ordinal' var or as a 'One Hot' Encoding var ?


df = pd.read_csv('HR_comma_sep.csv')

testing_salary = df[['salary', 'left']].copy()

preddf = testing_salary.copy()

# preddf = pd.get_dummies(preddf, columns=['salary'], drop_first=True)
preddf['salary'] = preddf['salary'].map({'low': 0, 'medium': 1, 'high': 2})

train, test = train_test_split(preddf, test_size=0.2, stratify=preddf['left'])

model = LogisticRegression(max_iter=1000)
model.fit(train.drop('left', axis=1), train['left'])

print(test[['left']].head())
print(model.predict(test.drop('left', axis=1).head()))
print(model.score(test.drop('left', axis=1), test['left']))

# The Result I Found That There is No Difference (In This Case)