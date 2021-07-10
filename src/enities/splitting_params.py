from dataclasses import dataclass


@dataclass()
class SplittingParams:
    random_state: int
    test_size: float
    stratify: bool
    X_train_path: str
    X_test_path: str
    y_train_path: str
    y_test_path: str
