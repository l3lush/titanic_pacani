import logging
import pickle
from typing import NoReturn

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def make_logger(name: str) -> logging.getLoggerClass():
    logger = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)
    return logger


def serialize_model(path: str, model) -> NoReturn:
    with open(path, "wb") as fin:
        pickle.dump(model, fin)


def download_model(path: str):
    with open(path, "rb") as fin:
        model = pickle.load(fin)
    return model
