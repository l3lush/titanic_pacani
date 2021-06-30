# -*- coding: utf-8 -*-
import click
import logging

import pandas as pd

from src.enities.train_pipeline_params import read_training_pipeline_params


@click.command()
@click.argument('config_path')
def main(config_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger('make_dataset')
    logger.info('making final data set from raw data')

    config = read_training_pipeline_params(config_path)
    logger.info(f'got config from path {config_path}')

    data = pd.read_csv(config.output_raw_data)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
