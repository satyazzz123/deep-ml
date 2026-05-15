import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Return train/test index splits for k-fold cross-validation.

    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds
        shuffle: Whether to shuffle indices before splitting

    Returns:
        List of (train_indices, test_indices) tuples, where each is a list of ints
    """
    indice_list = []
    for i in range(n_samples):
        indice_list.append(i)

    if shuffle:
        shuffled_list = np.random.permutation(indice_list)
    else:
        shuffled_list = indice_list
    
    folds = n_samples // k
    remaining_nums = n_samples % k
    
    folds_size = [folds + remaining_nums]
    for j in range(k - 1):
        folds_size.append(folds)
    
    final_list = []
    start = 0
    for i in range(len(folds_size)):
        end = start + folds_size[i]
        folds_list = shuffled_list[start:end]
        final_list.append(folds_list)
        start = end
    
    final_set = []
    for i in range(len(final_list)):
        test_fold = final_list[i]
        train_fold_segments = final_list[:i] + final_list[i+1:]

        flattened_train = []
        for segment in train_fold_segments:
            for item in segment:
                flattened_train.append(item)

        final_set.append((flattened_train, list(test_fold)))

    return final_set