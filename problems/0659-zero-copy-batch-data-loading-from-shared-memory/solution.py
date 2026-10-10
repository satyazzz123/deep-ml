
import numpy as np

class ZeroCopyBatchLoader:
    def __init__(self, data: np.ndarray, batch_size: int):
        if data.ndim != 2:
            raise ValueError("data must be a 2D array")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.num_samples = data.shape[0]
        self.num_features = data.shape[1]
        self.batch_size = batch_size

        # Flat, contiguous storage
        self.buffer = np.ascontiguousarray(data).reshape(-1)

    def num_batches(self) -> int:
        return (
            self.num_samples + self.batch_size - 1
        ) // self.batch_size

    def get_batch(self, batch_idx: int) -> np.ndarray:
        if batch_idx < 0 or batch_idx >= self.num_batches():
            raise IndexError("Invalid batch index")

        start_row = batch_idx * self.batch_size
        end_row = min(
            start_row + self.batch_size,
            self.num_samples
        )

        start = start_row * self.num_features
        end = end_row * self.num_features

        return self.buffer[start:end].reshape(
            end_row - start_row,
            self.num_features
        )

    def is_zero_copy(self, batch_idx: int) -> bool:
        return np.shares_memory(
            self.get_batch(batch_idx),
            self.buffer
        )

    def get_batch_means(self) -> list:
        return [
            round(float(self.get_batch(i).mean()), 4)
            for i in range(self.num_batches())
        ]

    def write_to_buffer(
        self, row: int, col: int, value: float
    ) -> None:
        if not (0 <= row < self.num_samples):
            raise IndexError("Invalid row")
        if not (0 <= col < self.num_features):
            raise IndexError("Invalid column")

        self.buffer[row * self.num_features + col] = value
