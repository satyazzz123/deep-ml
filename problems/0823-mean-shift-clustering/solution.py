
import numpy as np

def mean_shift(X, bandwidth, max_iter=300, tol=1e-3):
    X = np.asarray(X, dtype=float)
    num_samples = X.shape[0]

    shifted_pos = np.zeros_like(X)

    # Stage 1: Move each sample toward a density mode
    for i in range(num_samples):
        y = X[i].copy()

        for _ in range(max_iter):
            # Distance from y to every original sample
            distances = np.linalg.norm(X - y, axis=1)

            # Select neighbors within the bandwidth
            positions = X[distances <= bandwidth]

            # Calculate the neighborhood mean
            new_y = np.mean(positions, axis=0)

            # Check how far the position moved
            shift = np.linalg.norm(new_y - y)

            y = new_y

            if shift <= tol:
                break

        shifted_pos[i] = y

    # Stage 2: Assign labels to converged positions
    labels = -np.ones(num_samples, dtype=int)
    cluster_id = 0

    for i in range(num_samples):
        if labels[i] != -1:
            continue

        labels[i] = cluster_id

        for j in range(i + 1, num_samples):
            if labels[j] != -1:
                continue

            distance = np.linalg.norm(
                shifted_pos[i] - shifted_pos[j]
            )

            if distance <= bandwidth:
                labels[j] = cluster_id

        cluster_id += 1

    return labels.tolist()
