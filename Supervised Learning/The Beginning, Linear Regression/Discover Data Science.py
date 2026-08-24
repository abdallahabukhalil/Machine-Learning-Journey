import os
import pandas as pd
import sklearn.linear_model as sklm

from Cleaning import CsvFile # My Own Library


current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "API_EGY_DS2_en_csv_v2_12325.csv")

data = CsvFile(csv_file_path = data_path, read_kwargs = {"skiprows": 4})

data.drop(columns = ["Country Code", "Country Name", "Indicator Code", "Unnamed: 70"], inplace = True)

data = data.melt_transform_pivot(
    id_vars = "Indicator Name",
    var_name = "Year",
    value_name = "Population",
    transform_func = lambda x: x * 1000,
    pivot_columns = "Indicator Name",
)

df = data[['Year', 'Population, total', 'Population growth (annual %)']].dropna()
df['Population growth (annual %)'] = df['Population growth (annual %)'] / 100000
df['Population, total'] = df['Population, total'] / 1000

model = sklm.LinearRegression()
model.fit(df[['Year','Population growth (annual %)']], df['Population, total'])

corr_matrix = df.corr(method = 'pearson')
corr_value = corr_matrix.loc['Year', 'Population growth (annual %)']

print(f"Pearson Correlation Coefficient: {corr_value:.2f}")