from os.path import dirname, abspath, join
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

from Cleaning import CsvFile

current_dir = dirname(abspath(__file__))

df = CsvFile(join(current_dir, "titanic.csv")) # My Library


# ------- Cleaning

data = df.copy()


# --- extract Prefix and Surname from Name

data.extract_keywords('Name', inplace = True, use_regex = True, Prefix = r' ([A-Za-z]+)\.')
# or use data['Prefix'] = df['Name'].str.extract(r' ([A-Za-z]+)\.') by Pandas

data.extract_keywords('Name', inplace = True, use_regex = True, Surname = r'^([^,]+)')
# or use data['Surname'] = df['Name'].str.extract(r'^([^,]+)') by Pandas


# --- fill mising Embarked using the relatives (family name) or prefix

data.fill_missing('Embarked', strategy = 'mode', hierarchy_cols = ['Prefix', 'Surname'], inplace = True)
# or use py Pandas: (may has less performance)
# data['Embarked'] = data['Embarked'].fillna(
#     data.groupby(['Prefix', 'Surname'])['Embarked'].transform(lambda x: x.mode()[0] if not x.mode().empty else pd.NA)
# ) 
# data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)


# --- fill missing ages with the mean of every prefix

data.fill_missing('Age', strategy = 'mean', hierarchy_cols = ['Prefix'], inplace = True)
# or use data['Age'] = data['Age'].fillna(data.groupby('Prefix')['Age'].transform('mean')) by Pandas


# --- make temporarily column 'cabin_list'

data['cabin_list'] = data['Cabin'].str.split(" ")
exploded_cabins = data['cabin_list'].explode()


# --- finding the lowest and highest level of cabin every passenger booked

decks = exploded_cabins.str[0]

data['Highest_Deck'] = decks.groupby(level = 0).min().fillna('Unknown')
data['Lowest_Deck'] = decks.groupby(level = 0).max().fillna('Unknown')


# --- finding which side the passengers lied in their trip

exploded_numbers = exploded_cabins.str.extract(r'(\d+)')[0].astype(float)

sides = exploded_numbers.mod(2).map({0: 'Port', 1: 'Starboard'})
passenger_sides = sides.dropna().groupby(level = 0).unique()

data['Ship_Side'] = passenger_sides.str.join(' & ')
data['Ship_Side'] = data['Ship_Side'].fillna('Unknown')


# --- finding the Group Size

data['Group_Size'] = data.groupby('Ticket')['Ticket'].transform('count')


# --- drop unneccessairly columns

data.drop(['cabin_list', 'Cabin'], axis = 'columns', inplace = True)




# ------- Preprocessing

ml_data = data.drop(['PassengerId', 'Surname', 'Prefix', 'Name', 'Ticket', 'Fare'], axis = 'columns').copy()


# --- mapping the gender into 0 and 1

ml_data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})


# --- Encoding 'Embarked' column

ml_data = pd.get_dummies(ml_data, columns = ['Embarked'], drop_first = True)


# --- Encoding 'Ship_Side' column

encoded_sides = (
    ml_data['Ship_Side']
    .str.get_dummies(sep = ' & ')
    .drop('Unknown', axis = 'columns')
    .rename({'Port': 'Side_Port', 'Starboard': 'Side_Starboard'}, axis = 'columns')
)

ml_data = pd.concat([ml_data.drop('Ship_Side', axis = 'columns'), encoded_sides], axis = 'columns')


# --- Encoding The Deck Column with dropping "the lowest"

ml_data = pd.get_dummies(ml_data, columns = ['Highest_Deck']).drop('Highest_Deck_Unknown', axis = 'columns')

ml_data.drop('Lowest_Deck', axis = 'columns', inplace = True)


# --- Get the People who isn't in 'SibSp' or 'Parch'

ml_data['Friends'] = (ml_data['Group_Size'] - ml_data['Parch'] - ml_data['SibSp'] - 1).clip(lower = 0)

ml_data.drop('Group_Size', axis = 'columns', inplace = True)




# --- The Main Dishes XD

train_data, test_data = train_test_split(ml_data, test_size = 0.2, random_state = 42)


model = tree.DecisionTreeClassifier(max_depth = 12, random_state = 42)
model.fit(train_data.drop('Survived', axis = 'columns'), train_data['Survived'])


print(test_data.head())
print(model.predict(test_data.drop('Survived', axis = 'columns'))[:5])

print(f"Score: {model.score(test_data.drop('Survived', axis = 'columns'), test_data['Survived'])}")
print(ml_data['Survived'].value_counts(normalize = True))

# accuracy: ~78.77%