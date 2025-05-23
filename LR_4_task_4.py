import datetime
import json
import numpy as np
import pandas as pd
from sklearn import covariance, cluster

input_file = 'company_symbol_mapping.json'

with open(input_file, 'r') as f:
    company_symbols_map = json.load(f)

symbols, names = np.array(list(company_symbols_map.items())).T

df = pd.read_csv("stocks.csv")
df['date'] = pd.to_datetime(df['date'])

start_date = datetime.datetime(2003, 7, 3)
end_date = datetime.datetime(2007, 5, 4)
df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

price_table = df.pivot(index='date', columns='symbol', values='price')

price_table = price_table.dropna(axis=0)

quotes_diff = price_table.diff().dropna()

X = quotes_diff.values
X /= X.std(axis=0)

edge_model = covariance.GraphicalLassoCV()
edge_model.fit(X)

_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()

for i in range(num_labels + 1):
    print(f"Cluster {i + 1} =>", ', '.join(names[labels == i]))
