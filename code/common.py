import pickle
import time
import autograd.numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Callable, List, Tuple



# Setting global variables
INPUT_DATA = "../data/input_data/"  # Path for input data
REPORT_DATA = "../data/report_data/"  # Path for data ment for the report
REPORT_FIGURES = "../figures/"  # Path for figures ment for the report
EX_A = "EX_A_"
EX_B = "EX_B_"
EX_C = "EX_C_"
EX_D = "EX_D_"
EX_E = "EX_E_"
EX_F = "EX_F_"

# Common methods


def learning_rate_upper_limit(X_train : np.ndarray) -> float:
    """Computes the upper limit for the learning rate from the Hessian

    Args:
        X_train (np.ndarray): Traning data

    Returns:
        float: Computed learning rate
    """

    XT_X = X_train.T @ X_train
    H = (2./X_train.shape[0]) * XT_X  # The Hessian is the second derivate
    # Picking the largest eigenvalue of the Hessian matrix to use as a guide for determain upper limit for learning rate
    lr_upper_limit = 2./np.max(np.linalg.eig(H)[0])
    print(f"Upper limit learing rate: {lr_upper_limit}")
    return lr_upper_limit


# Methods below are reused from from project1
def manual_scaling(data: np.ndarray) -> np.ndarray:
    """    Avoids the use of sklearn StandardScaler(), which also
    divides the scaled value by the standard deviation.
    This scaling is essentially just a zero centering

    Args:
        data (np.ndarray): Input data

    Returns:
        np.ndarray: Scaled data
    """

    return data - np.mean(data, axis=0)


def standard_scaling(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Scales data using the StandarScaler from sklearn.preprocessing

    Args:
        train (np.ndarray): Training data
        test (np.ndarray): test data

    Returns:
        Tuple[np.ndarray, np.ndarray]: Scaled data
    """

    scaler = StandardScaler()
    scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled


def standard_scaling_single(data: np.ndarray) -> Tuple[np.ndarray, Callable]:
    """Scales data using the StandarScaler from sklearn.preprocessing. For scaling a single dataset.    

    Args:
        data ([type]): Input data set       

    Returns:
        [type]: Scaled data
    """

    scaler = StandardScaler()
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled, scaler


def min_max_scaling(data: np.ndarray) -> Tuple[np.ndarray, Callable]:
    """Scales data using the MinMaxScaler from sklearn.preprocessing

    Args:
        data ([type]): input data

    Returns:
        [type]: Scaled data
    """

    scaler = MinMaxScaler()
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled, scaler


def create_X(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Function based on code from course website. Creates design matrix. 

    Args:
        x (np.ndarray): input data
        y (np.ndarray): input data
        n (int): Number of degrees

    Returns:
        np.ndarray: Design Matrix
    """

    if (len(x.shape)) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)  # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:, q+k] = (x**(i-k))*(y**k)

    return X
    # return X[:, 1:]


def remove_intercept(X: np.ndarray) -> np.ndarray:
    """Removes the intercept from design matrix

    Args:
        X ([type]): Design matrix

    Returns:
        [type]: Design matrix with intercept removed. 
    """

    return X[:, 1:]


def timer(func: Callable) -> float:
    """
    Simple timer that can be used as a decorator to time functions
    """

    def timer_inner(*args, **kwargs):
        t0: float = time.time()
        result = func(*args, **kwargs)
        t1: float = time.time()
        print(
            f"Elapsed time {1000*(t1 - t0):6.4f}ms in function {func.__name__}"
        )
        return result
    return timer_inner


def create_img_patches(img: np.ndarray, ySteps: int, xSteps: int) -> List[np.ndarray]:
    """Divides an image into set of patches

    Args:
        img (np.ndarray): Original image
        ySteps (int): size of patch horizontally
        xSteps (int): size of patch vertically

    Returns:
        List[np.ndarray]: List containing patches
    """

    patches = []
    for y in range(0, img.shape[0], ySteps):
        for x in range(0, img.shape[1], xSteps):
            y_from = y
            y_to = y+ySteps
            x_from = x
            x_to = x+xSteps
            img_patch = img[y_from:y_to, x_from:x_to]
            patches.append(img_patch)

    return patches


def patches_to_img(patches: List[np.ndarray], ySteps: int, xSteps: int, nYpatches: int, nXpatches: int, plotImage: bool=False) -> np.ndarray:
    """Reconstructing the original image from a set of patches

    Args:
        patches (List[np.ndarray]): List of patches
        ySteps (int): size of patch horizontally
        xSteps (int): size of patch vertically
        nYpatches (int): number of patches in the horizontal
        nXpatches (int): number of patches in the vertical
        plotImage (bool, optional): True if image is to be plotted and shown. Defaults to False.

    Returns:
        np.ndarray: The reconstructed image
    """

    img = np.zeros((ySteps*nYpatches, xSteps*nXpatches))
    i = 0
    for y in range(0, img.shape[0], ySteps):
        for x in range(0, img.shape[1], xSteps):
            y_from = y
            y_to = y+ySteps
            x_from = x
            x_to = x+xSteps
            img[y_from:y_to, x_from:x_to] = patches[i]
            i += 1

    if plotImage:
        plt.imshow(img, cmap='gray')
        plt.title("Reconstructed img")
        plt.show()
    return img


def plotTerrainPatches(patches: List[np.ndarray], nYpatches: int, nXpatches: int, plotTitle: str="Terrain patches") -> Callable:
    """Plots a set of terrain patches

    Args:
        patches (List[np.ndarray]): List of patches
        nYpatches (int): Number of patches in the horizontal
        nXpatches (int): Number of patches in the vertical
        plotTitle (str, optional): Title of the plot. Defaults to "Terrain patches".

    Returns:
        Callable: Matplotlib Figure object
    """

    # Plotting terrain patches)
    fig, ax = plt.subplots(nYpatches, nXpatches, figsize=(4, 10))
    i = 0
    for y in range(nYpatches):
        for x in range(nXpatches):
            ax[y, x].title.set_text(f"Patch{i}")
            ax[y, x].set_xlabel("X")
            ax[y, x].set_ylabel("Y")
            ax[y, x].imshow(patches[i], cmap='gray')
            i += 1

    fig.suptitle(f"{plotTitle}")  # or plt.suptitle('Main title')
    plt.tight_layout()

    return fig

def createTerrainData(terrain: np.ndarray, includeMeshgrid: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sets up (x,y) gridpoints in relation to the terrain values as z

    Args:
        terrain (np.ndarray): terrain data
        includeMeshgrid (bool, optional): combines the (x,y) points as a mesh. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: returns the x,y and z values. 
    """
    
    z = np.array(terrain)
    x = np.arange(0, z.shape[1])
    y = np.arange(0, z.shape[0])
    if includeMeshgrid:
        x, y = np.meshgrid(x, y)
    return x, y, z


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


# Methods and classes from project1
class Regression():
    """ Super class containing methods for fitting, predicting and producing stats for regression models.   
    """

    def __init__(self):
        self.betas = None
        self.X_train = None
        self.t_train = None
        self.t_hat_train = None
        self.param = None
        self.param_name = None
        self.SVDfit = None
        self.SE_betas = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Polymorph 

        """
        pass

    @property
    def get_all_betas(self) -> np.ndarray:
        """Returns predictor values

        Returns:
            [np.ndarray]: betas
        """
        return self.betas

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Performs a prediction:       

        Args:
            X (np.ndarray): input data

        Returns:
            np.ndarray: Predicted values
        """

        prediction = X @ self.betas
        return prediction

    @property
    def SE(self):
        """Returns the standard error

        Returns:
            [type]: [description]
        """
        var_hat = (1./self.X_train.shape[0]) * \
            np.sum((self.t_train - self.t_hat_train)**2)

        if self.SVDfit:
            invXTX_diag = np.diag(SVDinv(self.X_train.T @ self.X_train))
        else:
            invXTX_diag = np.diag(np.linalg.pinv(
                self.X_train.T @ self.X_train))
        return np.sqrt(var_hat * invXTX_diag)

    def summary(self) -> pd.DataFrame:
        """Produces a summary with coeffs,  STD, confidence intervals

        Returns:
            [pd.DataFrame]: dataframe with values. 
        """
        # Estimated standard error for the beta coefficients
        N, P = self.X_train.shape
        SE_betas = self.SE

        # Calculating 95% confidence intervall
        CI_lower_all_betas = self.betas - (1.96 * SE_betas)
        CI_upper_all_betas = self.betas + (1.96 * SE_betas)

        # Summary dataframe
        params = np.zeros(self.betas.shape[0])
        params.fill(self.param)

        coeffs_df = pd.DataFrame.from_dict({f"{self.param_name}": params,
                                            "coeff name": [rf"$\beta${i}" for i in range(0, self.betas.shape[0])],
                                            "coeff value": np.round(self.betas, decimals=4),
                                            "std error": np.round(SE_betas, decimals=4),
                                            "CI lower": np.round(CI_lower_all_betas, decimals=4),
                                            "CI upper": np.round(CI_upper_all_betas, decimals=4)},
                                           orient='index').T

        return coeffs_df


class OLS(Regression):
    """Class for ordinary least squares regression. 

    Args:
        Regression ([Class]): Class to inherit. 
    """

    def __init__(self, degree: int=1, param_name: str="degree"):
        """init.

        Args:
            degree (int, optional): [description]. Defaults to 1.
            param_name (str, optional): [description]. Defaults to "degree".
        """
        super().__init__()
        self.param = degree
        self.param_name = param_name

    def fit(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Function to fit model

        Args:
            X (np.ndarray): Input data
            t (np.ndarray): target data

        Returns:
            np.ndarray: Predicted values. 
        """
        #self.SVDfit = SVDfit
        #self.keep_intercept = keep_intercept
        # if keep_intercept == False:
        #    X = X[:, 1:]

        self.X_train = X
        self.t_train = t

        # if SVDfit:
        #    self.betas = SVDinv(X.T @ X) @ X.T @ t
        # else:
        self.betas = np.linalg.pinv(X.T @ X) @ X.T @ t
        self.t_hat_train = X @ self.betas
        # print("betas.shape in train before squeeze:",self.betas.shape)
        self.betas = np.squeeze(self.betas)
        # print("betas.shape in train after squeeze:",self.betas.shape)
        return self.t_hat_train


def prepare_data(X: np.ndarray, t: np.ndarray, random_state: int, test_size: float=0.2, shuffle: bool=True, scale_X: bool=False, scale_t: bool=False, skip_intercept: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Function to prepare data. Has the ability to set test size, shuffle, scale both X and t, and skip intercept. 

    Args:
        X (np.ndarray): Input data
        t (np.ndarray): Target Data
        random_state ([type]): Seed value
        test_size (float, optional): Size of test data. Defaults to 0.2.
        shuffle (bool, optional): Shuffles the data before split. Defaults to True.
        scale_X (bool, optional): Scales x. Defaults to False.
        scale_t (bool, optional): Scales target data. Defaults to False.
        skip_intercept (bool, optional): Skips intercept. Defaults to True.

    Returns:
        Tuple: Arrays containing X_train, X_test, t_train, t_test
    """
    X_train, X_test, t_train, t_test = train_test_split(
        X, t, test_size=test_size, shuffle=shuffle, random_state=random_state)

    # Scale data
    if(scale_X):
        X_train, X_test = standard_scaling(X_train, X_test)

    if(scale_t):
        t_train, t_test = standard_scaling(t_train, t_test)

    if (skip_intercept):
        X_train = X_train[:, 1:]
        X_test = X_test[:, 1:]

    return X_train, X_test, t_train, t_test


if __name__ == '__main__':
    print("Import this file as a package please!")
