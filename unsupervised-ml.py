import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# prepare the data
customer_data = [
    [1000, 2],  # Customer 1
    [1500, 3],  # Customer 2
    [2000, 2],  # Customer 3
    [5000, 6],  # Customer 4
    [6000, 7],  # Customer 5
    [5500, 5],  # Customer 6
    [12000, 12],  # Customer 7
    [15000, 14],  # Customer 8
    [13000, 11],  # Customer 9
    [3000, 4],  # Customer 10
]

# create the model
model = KMeans(n_clusters=3)

#predicting the clusters
clusters = model.fit_predict(customer_data)+1

print("Cluster labels for each customer:", clusters)
for i in range(len(customer_data)):
    print(f"Customer {i + 1}: Data = {customer_data[i]}, Cluster = {clusters[i]}")
# Scatter Plot 

data = np.array(customer_data)

colors = ['blue', 'green', 'orange']

for cluster_num in range(1, 4):
    plt.scatter(
        data[clusters == cluster_num, 0],
        data[clusters == cluster_num, 1],
        color=colors[cluster_num - 1],
        s=100,
        label=f'Cluster {cluster_num}'
    )

# Plot centroids
centroids = model.cluster_centers_

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    color='red',
    marker='X',
    s=250,
    label='Centroids'
)

# Label each point with customer number
for i, point in enumerate(data):
    plt.annotate(
        f'C{i+1}',
        (point[0], point[1]),
        xytext=(5, 5),
        textcoords='offset points'
    )

plt.title("Customer Segmentation with K-Means")
plt.xlabel("Annual Spending ($)")
plt.ylabel("Visits Per Month")
plt.legend()
plt.grid(True)

plt.show()