import functools
import logging
import os
import time
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

# === LOGGER SETTINGS ===
LOG_FORMAT = "[%(levelname)s] [%(name)s] [%(asctime)s] %(message)s"

logging.basicConfig(format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def get_logger(file_name: str) -> logging.Logger:
    """
    Return logger with name and level given in input. Call should be:
    my_logger = get_logger(__name__)
    Parameters
    ----------
        - file_name: name of the logger. It should be the name of the
        current file.
    """

    logger = logging.getLogger(file_name)
    logger.setLevel(logging.DEBUG)  # the level is fixed to DEBUG
    return logger


def logtimer(func):
    """
    Log the runtime of the decorated function
    """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        logger = get_logger(__name__)
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        logger.debug("Finished %s in %.4f secs", func.__name__, run_time)
        return value, run_time

    return wrapper_timer


def load_generated_data(path: str):
    """
    load data expecting generator format
    """
    dataframe = pd.read_csv(path)
    input_columns = [col for col in dataframe if col.startswith("X")]
    X = dataframe[input_columns]
    output_columns = [col for col in dataframe if col.startswith("y_")]
    y = dataframe[output_columns]
    return X, y


def split_scored_data(dataframe):
    """
    load input prediction gt and error
    """
    input_columns = [col for col in dataframe if col.startswith("X")]
    X = dataframe[input_columns]
    output_columns = [col for col in dataframe if col.startswith("y_")]
    y = dataframe[output_columns]
    error_columns = [col for col in dataframe if col.startswith("error")]
    error = dataframe[error_columns]
    return X, y, error


def compute_default_columns_names(n_inputs, n_outputs):
    """
    return an array with default columns names
    """
    columns_names = []
    for i in range(n_inputs):
        columns_names.append("X_" + str(i))
    for i in range(n_outputs):
        columns_names.append("y_" + str(i))
    return columns_names


def save_df_to_csv(dataframe, path):
    """
    Dataframe.to_csv() extended with make dir
    """
    slashes_pos = [pos for pos, char in enumerate(path) if char == "/"]
    if not os.path.exists(path[: slashes_pos[-1]]):
        os.makedirs(path[: slashes_pos[-1]])
    dataframe.to_csv(path)


def split_dataframe(dataframe, chunk_size=100):
    """
    This function split a dataframe into sub dataframes given
    a fixed amount of chunk size
    """
    chunks = []
    num_chunks = len(dataframe) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(dataframe[(i * chunk_size) : (i + 1) * chunk_size])  # noqa: E203
    return chunks


def get_root_level_dir(dir_name):
    """
    get path of a directory in project root level
    """
    dir_path = os.path.dirname(__file__)
    # go up one level to reach project root folder
    slashes_pos = [pos for pos, char in enumerate(dir_path) if char == "/"]
    root_level_path = dir_path[: slashes_pos[-1]]

    datasets_dir = root_level_path + "/" + dir_name
    return datasets_dir


def generate_triangular_matrix(values, size):
    """
    this method generate a triangular matrix
    """
    upper = np.zeros((size, size))
    upper[np.triu_indices(size)] = values
    return upper


def nearest_pd(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1],
    which
    credits [2].
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if is_pd(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])  # noqa: E741
    k = 1
    while not is_pd(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def is_pd(B):
    """
    Returns true when input is positive-definite
    """
    try:
        multivariate_normal(mean=np.zeros(B.shape[0]), cov=B)
        return True
    except np.linalg.LinAlgError:
        return False
    except ValueError:
        return False


def split_dataframe_xy(dataframe):
    input_columns = [col for col in dataframe if col.startswith("X")]
    X = dataframe[input_columns]
    output_columns = [col for col in dataframe if col.startswith("y_")]
    y = dataframe[output_columns]
    return X.values, y.values.flatten()


def split_dataframe_xy_v3(dataframe):
    input_columns = [col for col in dataframe if col.startswith("X")]
    X = dataframe[input_columns]
    output_columns = [col for col in dataframe if col.startswith("y_")]
    y = dataframe[output_columns]
    return X, y


def squared_error(y, y_hat):
    return (y - y_hat) ** 2


def classification_error(y, y_hat):
    return int(y != y_hat)
