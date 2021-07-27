from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml

from src.enities.feature_params import FeatureParams
from src.enities.splitting_params import SplittingParams
from src.enities.feature_transforming import FeatureTransformingParams
from src.enities.train_params import TrainParams


@dataclass()
class TrainingPipelineParams:
    input_url_data: str
    output_raw_data: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    feature_transforming: FeatureTransformingParams
    train_params: TrainParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
