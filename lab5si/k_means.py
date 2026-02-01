import numpy as np


def initialize_centroids_forgy(data, k):
    indices = np.random.choice(data.shape[0], size=k, replace=False)
    return data[indices]


def initialize_centroids_kmeans_pp(data, k):
    centroids = np.zeros((k, data.shape[1]))
    centroids[0] = data[np.random.choice(data.shape[0], size=1, replace=True)]
    for i in range(1, k):
        distances = np.zeros(data.shape[0])
        for j in range(data.shape[0]):
            for centroid in centroids[:i]:
                distances[j] += np.sqrt(np.sum(data[j]-centroid)**2)
        centroids[i] = data[distances.argmax()]
    return np.array(centroids)


def assign_to_cluster(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    assignments = np.argmin(distances, axis=1)
    return assignments


def update_centroids(data, assignments):
    k = np.max(assignments) + 1
    new_centroids = np.array([
        data[assignments == i].mean(axis=0) for i in range(k)
    ])
    return new_centroids


def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :]) ** 2))


def k_means(data, num_centroids, kmeansplusplus=False):

    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else:
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)

    for i in range(100):
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments):
            break
        assignments = new_assignments

    return assignments, centroids, mean_intra_distance(data, assignments, centroids)
