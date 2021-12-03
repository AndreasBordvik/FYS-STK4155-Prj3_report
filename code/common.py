import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Callable, List, Tuple


# Setting global variables
INPUT_DATA = "../data/input_data/"  # Path for input data
REPORT_DATA = "../data/report_data/"  # Path for data ment for the report
REPORT_FIGURES = "../figures/"  # Path for figures ment for the report


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



if __name__ == '__main__':
    print("Import this file as a package please!")
