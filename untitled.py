# Importing necessary libraries
import random
import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs
%matplotlib inline


# Load your data
file_df = pd.read_excel("online_retail_II.xlsx")

# Cleaning the data
file_df = file_df.drop(["InvoiceDate", "Description", "StockCode"], axis=1)
file_df.head()
#Delete Nan rows
file_df.dropna(inplace=True)
file_df['Invoice'] = pd.to_numeric(file_df['Invoice'], errors= 'coerce')
file_df["Country"] = pd.factorize(file_df["Country"])[0]
print(file_df.dtypes)


# K-Means
### Normalize it in 2 ways
#First Way:
X = file_df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)

#Second Way:
# Normalize each numeric column in the DataFrame
for col in file_df.select_dtypes(include=['float64', 'int64']).columns:
    min = file_df[col].min()
    max = file_df[col].max()
    file_df[col] = (file_df[col] - min) / (max - min)

# Extract features and handle NaN values
X = file_df.values[:, 1:]  # Exclude the first column (e.g., "Invoice")
X = np.nan_to_num(X)  # Replace NaN with zero or another specified value
### With Previous code we knew the best way is Second way


## Eblow Method
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)

for k in K:
	# Building and fitting the model
	kmeanModel = KMeans(n_clusters=k).fit(X)
	kmeanModel.fit(X)

	distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
										'euclidean'), axis=1)) / X.shape[0])
	inertias.append(kmeanModel.inertia_)

	mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
								'euclidean'), axis=1)) / X.shape[0]
	mapping2[k] = kmeanModel.inertia_

for key, val in mapping1.items():
  print(f'{key} : {val}')

plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

for key, val in mapping2.items():
	print(f'{key} : {val}')

plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()




# Create a range of values for k
k_range = range(1, 5)

# Initialize an empty list to
# store the inertia values for each k
inertia_values = []

# Fit and plot the data for each k value
for k in k_range:
	kmeans = KMeans(n_clusters=k, \
					init='k-means++', random_state=42)
	y_kmeans = kmeans.fit_predict(X)
	inertia_values.append(kmeans.inertia_)
	plt.scatter(X[:, 0], X[:, 1], c=y_kmeans)
	plt.scatter(kmeans.cluster_centers_[:, 0],\
				kmeans.cluster_centers_[:, 1], \
				s=100, c='red')
	plt.title('K-means clustering (k={})'.format(k))
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.show()

# Plot the inertia values for each k
plt.plot(k_range, inertia_values, 'bo-')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()
### With Previous code we knew the best K is 4

# Modeling
clusterNum = 4
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)


#Add one Column in data (label)
file_df["label"] = labels
print(file_df.head(5))
print(file_df.groupby('label').mean())



fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Country')
ax.set_ylabel('Price')
ax.set_zlabel('Quantity')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c=labels.astype(np.float64))

plt.show()