import torch

def k_means_clustering(points, k, initial_centroids, max_iterations) -> list[tuple[float, ...]]:
    # Convert to tensors
    points_t = torch.as_tensor(points, dtype=torch.float32)
    centroids = torch.as_tensor(initial_centroids, dtype=torch.float32)

    for i in range(max_iterations):
        # 1. Compute distances from every point to every centroid
        # points_t: (N, Features), centroids: (K, Features)
        # We use broadcasting to get a (N, K) distance matrix
        # Distance = sqrt(sum((p - c)^2))
        distances = torch.sqrt(((points_t.unsqueeze(1) - centroids) ** 2).sum(dim=2))

        # 2. Assign each point to the closest centroid
        # labels: (N,) containing indices 0 to K-1
        labels = torch.argmin(distances, dim=1)

        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            # 3. Update centroids: Get all points assigned to cluster j
            cluster_points = points_t[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = cluster_points.mean(dim=0)
            else:
                # If a cluster is empty, keep the centroid where it was
                new_centroids[j] = centroids[j]

        # Check for convergence (if centroids stop moving, we can stop early)
        if torch.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    # 4. Format output: Round to 4 decimals and convert to list of tuples
    final_centroids = [tuple(round(float(f), 4) for f in c) for c in centroids]
    return final_centroids