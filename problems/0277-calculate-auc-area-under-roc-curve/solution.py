import torch
def calculate_auc(y_true, y_scores):
    if len(y_true) != len(y_scores) or len(y_true) == 0:
        raise ValueError("Inputs must have equal, nonzero lengths")

    if any(label not in (0, 1) for label in y_true):
        raise ValueError("Labels must be 0 or 1")

    positives = sum(y_true)
    negatives = len(y_true) - positives

    # Required edge case
    if positives == 0 or negatives == 0:
        return 0.0

    # Pair labels with scores and sort by descending score
    pairs = sorted(
        zip(y_scores, y_true),
        reverse=True
    )

    tp = 0
    fp = 0
    prev_tpr = 0.0
    prev_fpr = 0.0
    auc = 0.0

    i = 0
    while i < len(pairs):
        score = pairs[i][0]

        # Process all examples sharing this threshold together
        while i < len(pairs) and pairs[i][0] == score:
            label = pairs[i][1]

            if label == 1:
                tp += 1
            else:
                fp += 1

            i += 1

        # Current ROC coordinates
        tpr = tp / positives
        fpr = fp / negatives

        # Trapezoidal integration
        width = fpr - prev_fpr
        avg_height = (tpr + prev_tpr) / 2
        auc += width * avg_height

        prev_tpr = tpr
        prev_fpr = fpr

    return auc.item()


