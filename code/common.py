import pickle
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Callable, List, Tuple
from tensorflow import keras


# Setting global variables
INPUT_DATA = "../data/input_data/"  # Path for input data
REPORT_DATA = "../data/report_data/"  # Path for data ment for the report
REPORT_FIGURES = "../figures/"  # Path for figures ment for the report
test_samples = [36]


def save_model(model: Callable, path_filename: str) -> None:
    """saving the medel as .pkl filetype

    Args:
        model (Callable): Model to be saved
        path_filename (str): /path/to/model
    """

    with open(path_filename, 'wb') as outp:  # Overwrites existing .pkl file.
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)


def load_model(path_filename: str) -> Callable:
    """Loading a .pkl filetype

    Args:
        path_filename (str): /path/to/model

    Returns:
        Callable: Loaded model
    """

    with open(path_filename, 'rb') as inp:
        model = pickle.load(inp)
    return model


def last_time_step_mse(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Method from Gerome(Hands on machine-learning)

    Args:
        Y_true (np.ndarray): Target data
        Y_pred (np.ndarray): Predicted data

    Returns:
        float: MSE
    """
    return keras.metrics.mean_absolute_error(Y_true[:, -1], Y_pred[:, -1])


if __name__ == '__main__':
    print("Import this file as a package please!")
