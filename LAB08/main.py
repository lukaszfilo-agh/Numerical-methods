import numpy as np
import scipy as sp
from scipy import linalg
from datetime import datetime
import pickle

from typing import Union, List, Tuple

'''
Do celów testowych dla elementów losowych uzywaj seed = 24122022
'''


def random_matrix_by_egval(egval_vec: np.ndarray):
    """Funkcja z pierwszego zadania domowego
    Parameters:
    egval_vec : wetkor wartości własnych
    Results:
    np.ndarray: losowa macierza o zadanych wartościach własnych 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    try:
        if not isinstance(egval_vec, (np.ndarray, List)):
            raise TypeError

        np.random.seed(24122022)

        len_v = len(egval_vec)
        jord = np.diag(egval_vec)
        p = np.random.rand(len_v, len_v)
        p_inv = np.linalg.inv(p)
        a = p @ jord @ p_inv
        return a
    except TypeError:
        return None


def frob_a(coef_vec: np.ndarray):
    """Funkcja z drugiego zadania domowego
    Parameters:
    coef_vec : wetkor wartości wspołczynników
    Results:
    np.ndarray: macierza Frobeniusa o zadanych wartościach współczynników wielomianu 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    try:
        if not isinstance(coef_vec, np.ndarray):
            raise TypeError
        len_v = len(coef_vec)
        frob = np.eye(len_v, k=1)
        frob[len_v - 1] = -coef_vec[::-1]
        return frob
    except TypeError:
        return None


def polly_from_egval(egval_vec: np.ndarray):
    """Funkcja z laboratorium 8
    Parameters:
    egval_vec: wetkor wartości własnych
    Results:
    np.ndarray: wektor współczynników wielomianu charakterystycznego
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    return None
