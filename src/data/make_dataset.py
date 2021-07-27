import click

import pandas as pd

from src.enities.train_pipeline_params import read_training_pipeline_params
from src.features.build_features import CustomTransformer
from src.utils import make_logger, serialize_model


@click.command()
@click.argument("config_path")
def main(config_path):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    config = read_training_pipeline_params(config_path)
    logger.info(f"got config from path {config_path}")

    X_train = pd.read_csv(config.splitting_params.X_train_path)
    X_test = pd.read_csv(config.splitting_params.X_test_path)
    logger.info(f"read all data")

    transformer = CustomTransformer(config.feature_params)
    transformer.fit(X_train)
    logger.info("transformer fitted")

    X_train_transformed = transformer.transform(X_train)
    X_test_transformed = transformer.transform(X_test)
    logger.info("succesfully transformed X_train and X_test")

    X_train_transformed.to_csv(config.feature_transforming.X_train_transformed_path)
    X_test_transformed.to_csv(config.feature_transforming.X_test_transformed_path)

    serialize_model(config.feature_transforming.transformer_path, transformer)


if __name__ == "__main__":
    logger = make_logger(__file__)
    main()
