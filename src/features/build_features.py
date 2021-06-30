import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from src.enities.feature_params import FeatureParams


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, params: FeatureParams):
        self.params = params
        self.categorical_pipeline = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(missing_values=np.nan, strategy="most_frequent"),
                ),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        self.numerical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

    def fit(self, data: pd.DataFrame):
        self.numerical_pipeline.fit(data[self.params.numerical_features])
        self.categorical_pipeline.fit(data[self.params.categorical_features])

    def transform(self, data: pd.DataFrame):
        _data = data.copy()
        new_categorical_cols = self.categorical_pipeline.get_params()[
            "ohe"
        ].get_feature_names(self.params.categorical_features)
        _data[self.params.numerical_features] = self.numerical_pipeline.transform(
            _data[self.params.numerical_features]
        )
        _data[new_categorical_cols] = self.categorical_pipeline.transform(
            _data[self.params.categorical_features]
        ).toarray()
        _data.drop(self.params.categorical_features, axis=1, inplace=True)
        if self.params.target in _data.columns:
            _data.drop(self.params.target, axis=1, inplace=True)
        return _data


def extract_target(data: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = data[params.target]
    return target
