# -*- coding: utf-8 -*-
import click
import logging

import pandas as pd

from src.enities.train_pipeline_params import read_training_pipeline_params


@click.command()
@click.argument('config_path')
def main(config_path):
    """
    Get config file and download data
    """
    logger = logging.getLogger('download data')
    logger.info('download data')

    config = read_training_pipeline_params(config_path)
    logger.info(f'got config from path {config_path}')

    data = pd.read_csv(config.input_url_data)
    logger.info(f'successully download data from {config.input_url_data}')

    data.to_csv(config.output_raw_data, index=False)
    logger.info(f'saved raw downloaded data to {config.output_raw_data}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
