
import torch
from collections import Counter
import math

def bleu_score(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
    candidate = [word.lower() for word in candidate]
    references = [[word.lower() for word in ref] for ref in references]

    if not candidate or not references or max_n < 1:
        return 0.0

    precisions = []

    for n in range(1, max_n + 1):
        # Generate candidate n-grams
        candidate_ngrams = [
            tuple(candidate[j:j+n])
            for j in range(len(candidate) - n + 1)
        ]
        candidate_counter = Counter(candidate_ngrams)

        # Maximum reference count for each n-gram across references
        max_reference_counts = Counter()

        for reference in references:
            reference_ngrams = [
                tuple(reference[j:j+n])
                for j in range(len(reference) - n + 1)
            ]
            reference_counter = Counter(reference_ngrams)

            for gram, count in reference_counter.items():
                max_reference_counts[gram] = max(
                    max_reference_counts[gram], count
                )

        # Clipped matches
        matches = sum(
            min(count, max_reference_counts[gram])
            for gram, count in candidate_counter.items()
        )

        total = len(candidate_ngrams)

        # No available n-grams or no matches => zero (unsmoothed BLEU)
        if total == 0 or matches == 0:
            return 0.0

        precisions.append(matches / total)

    # Brevity penalty: choose reference length closest to candidate length
    candidate_length = len(candidate)
    reference_length = min(
        (len(ref) for ref in references),
        key=lambda length: (abs(length - candidate_length), length)
    )

    bp = 1.0 if candidate_length > reference_length else math.exp(
        1 - reference_length / candidate_length
    )

    # Geometric mean using PyTorch
    precision_tensor = torch.tensor(precisions, dtype=torch.float64)
    geometric_mean = torch.exp(torch.log(precision_tensor).mean())

    return (bp * geometric_mean).item()
