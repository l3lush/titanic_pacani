from dataclasses import dataclass
from typing import Union


@dataclass()
class TrainParams:
    random_state: int
    estimator_type: str
    model_path: str
    penalty: str
    C: float
    solver: str
    n_jobs: int
    max_iter: int
    max_depth: int
    min_samples_split: Union[float, int]
    min_samples_leaf: Union[float, int]
