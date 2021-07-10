import click
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.enities.train_pipeline_params import read_training_pipeline_params


@click.command()
@click.argument("config_path")
def main(config_path):
    """Split data to train and test"""
    logger = logging.getLogger("split data")
    config = read_training_pipeline_params(config_path)
    data = pd.read_csv(config.output_raw_data)

    X = data.drop(
        [config.feature_params.target] + config.feature_params.features_to_drop, axis=1
    )
    y = data[config.feature_params.target]
    logger.info(f"got X with shape {X.shape}, y shape {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.splitting_params.test_size,
        random_state=config.splitting_params.random_state,
        stratify=y if config.splitting_params.stratify else None,
    )
    logger.info("splitted X and y via train_test_split")

    X_train.to_csv(config.splitting_params.X_train_path, index=False)
    X_test.to_csv(config.splitting_params.X_test_path, index=False)
    y_train.to_csv(config.splitting_params.y_train_path, index=False)
    y_test.to_csv(config.splitting_params.y_test_path, index=False)
    logger.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    logger.info(f"y_train: {y_train.shape}, y_test: {y_test.shape}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
