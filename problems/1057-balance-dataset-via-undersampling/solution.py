
from collections import Counter
import numpy as np

def balance_undersample(data: list) -> list:
    """
    Undersample majority classes to the minority class count.
    Preserve the original relative order of retained samples.
    """
    if not data:
        return []

    labels = [sample[1] for sample in data]
    counter = Counter(labels)
    minimum = min(counter.values())

    numpy_labels = np.asarray(labels)
    idx = []

    for label in counter:
        print(label)
        class_indices = np.where(numpy_labels == label)[0]
        idx.extend(class_indices[:minimum])

    mini_idx = np.sort(np.asarray(idx))

    return [data[i] for i in mini_idx]
