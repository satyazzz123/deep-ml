import numpy as np

def rotation_layer(X, angle):
    theta =(angle)
    c, s = np.cos(theta), np.sin(theta)

    rotation_matrix = np.array([
        [c, -s],
        [s,  c]
    ])

    X = np.asarray(X)

    return X @ rotation_matrix.T