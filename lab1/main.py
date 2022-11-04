import math

import numpy as np


def cylinder_area(r: float, h: float):
    """Obliczenie pola powierzchni walca. 
    Szczegółowy opis w zadaniu 1.
    
    Parameters:
    r (float): promień podstawy walca 
    h (float): wysokosć walca
    
    Returns:
    float: pole powierzchni walca 
    """
    if h <= 0 or r <= 0:
        return math.nan
    else:
        return 2 * math.pi * r ** 2 + 2 * math.pi * r * h


def fib(n: int):
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego. 
    Szczegółowy opis w zadaniu 3.
    
    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia 
    
    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """

    if n <= 0 or not isinstance(n, int):
        return None
    elif n == 1:
        return np.array([1])
    elif n == 2:
        return np.array([1, 1])
    else:
        result = np.array([1, 1])
        while len(result) < n:
            result = np.append(result, [result[-1] + result[-2]])
    return np.reshape(result, (1, n))


def matrix_calculations(a: float):
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej 
    na podstawie parametru a.  
    Szczegółowy opis w zadaniu 4.
    
    Parameters:
    a (float): wartość liczbowa 
    
    Returns:
    touple: krotka zawierająca wyniki obliczeń 
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """
    M = np.array([[a, 1, -a], [0, 1, 1], [-a, a, 1]])
    Mt = M.transpose()
    Mdet = np.linalg.det(M)
    if Mdet == 0:
        Minv = np.NaN
    else:
        Minv = np.linalg.inv(M)

    return Minv, Mt, Mdet


def custom_matrix(m: int, n: int):
    """Funkcja zwraca macierz o wymiarze mxn zgodnie 
    z opisem zadania 7.  
    
    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy  
    
    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """
    if m < 0 or n < 0 or not isinstance(m, int) or not isinstance(n, int):
        return None
    else:
        result = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                if i > j:
                    result[i][j] = i
                else:
                    result[i][j] = j
    return result
