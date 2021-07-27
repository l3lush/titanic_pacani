# -*- coding: utf-8 -*-
import click

import pandas as pd

from src.enities.train_pipeline_params import read_training_pipeline_params
from src.utils import make_logger


@click.command()
@click.argument("config_path")
def main(config_path):
    """
    Get config file and download data
    """
    config = read_training_pipeline_params(config_path)
    logger.info(f"got config from path {config_path}")

    data = pd.read_csv(config.input_url_data)
    logger.info(f"successully download data from {config.input_url_data}")

    data.to_csv(config.output_raw_data, index=False)
    logger.info(f"saved raw downloaded data to {config.output_raw_data}")


if __name__ == "__main__":
    logger = make_logger(__file__)
    main()
