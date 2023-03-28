##
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt

from typing import Union, List, Tuple


def chebyshev_nodes(n: int = 10) -> np.ndarray:
    """Funkcja tworząca wektor zawierający węzły czybyszewa w postaci wektora (n+1,)

    Parameters:
    n(int): numer ostaniego węzła Czebyszewa. Wartość musi być większa od 0.

    Results:
    np.ndarray: wektor węzłów Czybyszewa o rozmiarze (n+1,).
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    try:
        if not isinstance(n, int):
            raise ValueError
        k = np.arange(0, n + 1, 1)
        cos = np.cos((k * np.pi) / n)
        return cos
    except(ValueError, TypeError):
        return None


def bar_czeb_weights(n: int = 10) -> np.ndarray:
    """Funkcja tworząca wektor wag dla węzłów czybyszewa w postaci (n+1,)
    
    Parameters:
    n(int): numer ostaniej wagi dla węzłów Czebyszewa. Wartość musi być większa od 0.
     
    Results:
    np.ndarray: wektor wag dla węzłów Czybyszewa o rozmiarze (n+1,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    try:
        wj = np.ndarray((n + 1,))
        for j in range(n + 1):
            if j == 0 or j == n:
                dj = 0.5
                wj[j] = (-1) ** j * dj
            elif 0 < j < n:
                dj = 1
                wj[j] = (-1) ** j * dj
        return wj
    except TypeError:
        return None


def barycentric_inte(xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Funkcja przprowadza interpolację metodą barycentryczną dla zadanych węzłów xi
        i wartości funkcji interpolowanej yi używając wag wi. Zwraca wyliczone wartości
        funkcji interpolującej dla argumentów x w postaci wektora (n,) gdzie n to dłógość
        wektora n. 
    
    Parameters:
    xi(np.ndarray): węzły interpolacji w postaci wektora (m,), gdzie m > 0
    yi(np.ndarray): wartości funkcji interpolowanej w węzłach w postaci wektora (m,), gdzie m>0
    wi(np.ndarray): wagi interpolacji w postaci wektora (m,), gdzie m>0
    x(np.ndarray): argumenty dla funkcji interpolującej (n,), gdzie n>0 
     
    Results:
    np.ndarray: wektor wartości funkcji interpolujący o rozmiarze (n,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    try:
        if not all(isinstance(i, np.ndarray) for i in [xi, yi, wi, x]):
            raise TypeError
        if xi.shape != yi.shape or yi.shape != wi.shape:
            raise ValueError
        y = []
        for x in np.nditer(x):
            le = wi / (x - xi)
            y.append(yi @ le / sum(le))

        return np.array(y)
    except (ValueError, TypeError):
        return None


def L_inf(xr: Union[int, float, List, np.ndarray], x: Union[int, float, List, np.ndarray]) -> float:
    """Obliczenie normy  L nieskończonośćg.
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach biblioteki numpy.

    Parameters:
    xr (Union[int, float, List, np.ndarray]): wartość dokładna w postaci wektora (n,)
    x (Union[int, float, List, np.ndarray]): wartość przybliżona w postaci wektora (n,1)

    Returns:
    float: wartość normy L nieskończoność,
                                    NaN w przypadku błędnych danych wejściowych
    """
    try:
        if np.array(x).shape != np.array(xr).shape:
            raise ValueError
        elif np.size(np.array(x)) == 1:
            return abs(x - xr)
        else:
            return max(abs(np.array(xr) - np.array(x)))
    except (ValueError, TypeError):
        return np.NaN
