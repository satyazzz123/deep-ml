import torch

def splitting_and_tokenization(sentence: str) -> list[str]:
    return sentence.lower().split()


def chunk_calculator(reference_index: list[int], candidate_index: list[int]) -> int:
    """
    Count contiguous chunks.

    Two consecutive matched words belong to the same chunk iff they are
    consecutive in BOTH the reference and candidate.
    """
    if len(reference_index) == 0:
        return 0

    chunks = 1

    for i in range(1, len(reference_index)):
        if (
            reference_index[i] != reference_index[i - 1] + 1
            or candidate_index[i] != candidate_index[i - 1] + 1
        ):
            chunks += 1

    return chunks


def meteor_score(
    reference: str,
    candidate: str,
    alpha: float = 0.9,
    beta: float = 3,
    gamma: float = 0.5,
) -> torch.Tensor:
    """
    Simplified METEOR using exact word matching.
    """

    reference_list = splitting_and_tokenization(reference)
    candidate_list = splitting_and_tokenization(candidate)

    # ------------------------------------------------------
    # Build alignment
    # ------------------------------------------------------

    used_reference = set()

    reference_index = []
    candidate_index = []

    matches = 0

    # Walk through candidate in order
    for c_idx, c_word in enumerate(candidate_list):

        # Find first unused matching word in reference
        for r_idx, r_word in enumerate(reference_list):

            if r_idx in used_reference:
                continue

            if c_word == r_word:
                matches += 1
                used_reference.add(r_idx)

                reference_index.append(r_idx)
                candidate_index.append(c_idx)

                break

    # No matches
    if matches == 0:
        return torch.tensor(0.0)

    # ------------------------------------------------------
    # Precision / Recall
    # ------------------------------------------------------

    precision = matches / len(candidate_list)
    recall = matches / len(reference_list)

    f_mean = (precision * recall) / (
        alpha * precision + (1 - alpha) * recall
    )

    # ------------------------------------------------------
    # Chunk penalty
    # ------------------------------------------------------

    chunks = chunk_calculator(reference_index, candidate_index)

    penalty = gamma * (chunks / matches) ** beta

    meteor = (1 - penalty) * f_mean

    return torch.tensor(meteor, dtype=torch.float32)