
import numpy as np

def guidance_attention_mask(
    chunk_sizes: list[int],
    current_chunk: int
) -> np.ndarray:

    total_tokens = sum(chunk_sizes)

    current_start = sum(chunk_sizes[:current_chunk])
    current_end = current_start + chunk_sizes[current_chunk]

    mask = np.zeros(
        (total_tokens, total_tokens),
        dtype=bool
    )

    # Historical queries can attend to all historical keys
    mask[:current_start, :current_start] = True

    # Current queries can attend to all historical keys
    mask[current_start:current_end, :current_start] = True

    # Current chunk can attend to itself and earlier tokens
    mask[current_start:current_end, current_start:current_end] = np.tril(
        np.ones(
            (current_end - current_start, current_end - current_start),
            dtype=bool
        )
    )

    return mask
