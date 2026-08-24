import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('carprices.csv')

ready = pd.merge(
    df[['Mileage', 'Sell Price($)', 'Age(yrs)']],
    pd.get_dummies(
        df['Car Model'],
        drop_first = True
    ),
    left_index = True,
    right_index = True
)
# "BMW X5", "Mercedez Benz C class" with hidden "Audi A5"

train, test = train_test_split(ready, test_size = 0.5)

model = LinearRegression()
model.fit(train.drop('Sell Price($)', axis = 1), train['Sell Price($)'])

print(test)
print(model.predict(test.drop('Sell Price($)', axis = 1)))
print(model.score(test.drop('Sell Price($)', axis = 1), test['Sell Price($)']))