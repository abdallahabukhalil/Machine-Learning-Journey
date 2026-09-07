import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.datasets import load_iris
from kneed import KneeLocator
from scipy.stats import mode

iris_data = load_iris()

df = pd.DataFrame(data = iris_data.data, columns = iris_data.feature_names)

scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = pd.DataFrame(scaler.fit_transform(df), columns = df.columns, index = df.index)

inertia = []
rangee = range(1, 11)

for k in rangee:
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

knee = KneeLocator(rangee, inertia, curve = "convex", direction = "decreasing")

optimal_k = knee.elbow

kmeans = KMeans(n_clusters = optimal_k, random_state = 42, n_init = "auto")

scaled_data['Cluster'] = kmeans.fit_predict(scaled_data)

sil_score = silhouette_score(
    scaled_data.drop(columns = ['Cluster']),
    scaled_data['Cluster']
)

mapped_labels = {}

for num in range(optimal_k):
    cluster_data = scaled_data[scaled_data['Cluster'] == num]
    mapped_labels[num] = mode(iris_data.target[cluster_data.index], keepdims = True).mode[0]

df['Predicted_Cluster'] = scaled_data['Cluster'].map(mapped_labels)
df['Target'] = iris_data.target

acc_score = accuracy_score(df['Target'], df['Predicted_Cluster'])

print(df.head().to_string())
print(f"Optimal K: {optimal_k}")
print(f"Silhouette Score: {sil_score}") # ~48.29%
print(f"Accuracy Score: {acc_score}") # ~88%