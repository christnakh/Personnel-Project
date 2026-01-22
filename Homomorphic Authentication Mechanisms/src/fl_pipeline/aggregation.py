import numpy as np
from typing import List, Any

class Aggregator:
    def __init__(self, method: str = "average"):
        self.method = method
    
    def aggregate(self, updates: List[Any]) -> np.ndarray:
        if not updates:
            return np.array([])
        if isinstance(updates[0], np.ndarray):
            updates_array = np.array(updates)
            if self.method == "average":
                return np.mean(updates_array, axis=0)
            elif self.method == "sum":
                return np.sum(updates_array, axis=0)
            elif self.method == "fedavg":
                return np.mean(updates_array, axis=0)
            else:
                return np.mean(updates_array, axis=0)
        return updates[0] if updates else None

