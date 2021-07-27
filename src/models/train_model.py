import click

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.enities.train_pipeline_params import read_training_pipeline_params, TrainingPipelineParams
from src.utils import make_logger, serialize_model


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, config: TrainingPipelineParams):
    if config.train_params.estimator_type == 'LogisticRegression':
        estimator = LogisticRegression(random_state=config.train_params.random_state,
                                       C=config.train_params.C,
                                       solver=config.train_params.solver,
                                       n_jobs=config.train_params.n_jobs,
                                       penalty=config.train_params.penalty,
                                       max_iter=config.train_params.max_iter)
    elif config.train_params.estimator_type == 'DecisionTreeClassifier':
        estimator = DecisionTreeClassifier(max_depth=config.train_params.max_depth,
                                           min_samples_split=config.train_params.min_samples_split,
                                           min_samples_leaf=config.train_params.min_samples_leaf,
                                           random_state=config.train_params.random_state,
                                           )
    else:
        raise NotImplementedError

    estimator.fit(X_train, y_train.values.ravel())
    return estimator


@click.command()
@click.argument("config_path")
def main(config_path):
    """
    Train model on train dataset.
    """
    config = read_training_pipeline_params(config_path)

    X_train = pd.read_csv(config.feature_transforming.X_train_transformed_path)
    y_train = pd.read_csv(config.splitting_params.y_train_path)

    estimator = train_model(X_train, y_train, config)
    logger.info(f'trained model {estimator}')

    serialize_model(config.train_params.model_path, estimator)
    logger.info(f'model saved in {config.train_params.model_path}')


if __name__ == '__main__':
    logger = make_logger(__file__)
    main()
