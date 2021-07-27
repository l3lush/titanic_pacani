from dataclasses import dataclass


@dataclass()
class FeatureTransformingParams:
    X_train_transformed_path: str
    X_test_transformed_path: str
    transformer_path: str
