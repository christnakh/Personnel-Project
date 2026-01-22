import numpy as np
from typing import Tuple, Optional

def encode_vector(vector: np.ndarray, dtype: str = 'float32') -> bytes:
    return vector.astype(dtype).tobytes()

    return vector.astype(dtype).tobytes()
def decode_vector(data: bytes, dtype: str = 'float32', shape: tuple = None) -> np.ndarray:
    vector = np.frombuffer(data, dtype=dtype)
    if shape:
        vector = vector.reshape(shape)
    return vector
def generate_random_vector(dimension: int, dtype: str = 'float32') -> np.ndarray:
    return np.random.randn(dimension).astype(dtype)