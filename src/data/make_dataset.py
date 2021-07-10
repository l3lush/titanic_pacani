import click
import logging

import pandas as pd

from src.enities.train_pipeline_params import read_training_pipeline_params
from src.features.build_features import CustomTransformer


@click.command()
@click.argument("config_path")
def main(config_path):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger("make_dataset")
    logger.info("making final data set from raw data")

    config = read_training_pipeline_params(config_path)
    logger.info(f"got config from path {config_path}")

    data: pd.DataFrame = pd.read_csv(config.output_raw_data)
    logger.info(f"read data from {config.output_raw_data} with shape {data.shape}")

    transformer = CustomTransformer(config.feature_params)
    transformer.fit(data)
    train_features = transformer.transform()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
