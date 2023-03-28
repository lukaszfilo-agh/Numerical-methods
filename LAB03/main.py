import numpy as np
import scipy
import pickle
import math
from typing import Union, List, Tuple


def absolut_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[
    int, float, np.ndarray]:
    """Obliczenie błędu bezwzględnego. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu bezwzględnego,
                                       NaN w przypadku błędnych danych wejściowych
    """

    try:
        return abs(np.array(v_aprox) - np.array(v))
    except(ValueError, TypeError):
        return np.NaN


def relative_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[
    int, float, np.ndarray]:
    """Obliczenie błędu względnego.
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu względnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    try:
        if np.array(v).all() == 0:
            raise ValueError
        return abs(np.array(v_aprox) - np.array(v)) / np.array(v)
    except (ValueError, TypeError):
        return np.NaN


def p_diff(n: int, c: float) -> float:
    """Funkcja wylicza wartości wyrażeń P1 i P2 w zależności od n i c.
    Następnie zwraca wartość bezwzględną z ich różnicy.
    Szczegóły w Zadaniu 2.
    
    Parameters:
    n Union[int]: 
    c Union[int, float]: 
    
    Returns:
    diff float: różnica P1-P2
                NaN w przypadku błędnych danych wejściowych
    """
    if not isinstance(n, int) or (not isinstance(c, float) and not isinstance(c, int)):
        return np.NaN
    p1 = 2 ** n - 2 ** n + c
    p2 = 2 ** n + c - 2 ** n

    return abs(p1 - p2)


def exponential(x: Union[int, float], n: int) -> float:
    """Funkcja znajdująca przybliżenie funkcji exp(x).
    Do obliczania silni można użyć funkcji scipy.math.factorial(x)
    Szczegóły w Zadaniu 3.
    
    Parameters:
    x Union[int, float]: wykładnik funkcji ekspotencjalnej 
    n Union[int]: liczba wyrazów w ciągu
    
    Returns:
    exp_aprox float: aproksymowana wartość funkcji,
                     NaN w przypadku błędnych danych wejściowych
    """

    ex = 0
    if not isinstance(n, int) or (not isinstance(x, float) and not isinstance(x, int)) or n < 0:
        return np.NaN
    for i in range(n):
        ex += (1 / scipy.special.factorial(i)) * x ** i
    return ex


def coskx1(k: int, x: Union[int, float]) -> float:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 1.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx float: aproksymowana wartość funkcji,
                 NaN w przypadku błędnych danych wejściowych
    """
    if not isinstance(k, int) or (not isinstance(x, int) and not isinstance(x, float)):
        return np.NaN
    if k < 0:
        return np.NaN
    elif k == 0 or x == 0:
        return 1
    elif k == 1:
        return np.cos(x)
    elif k > 1:
        coskx = 2 * coskx1(1, x) * coskx1(k - 1, x) - coskx1(k - 2, x)
        return coskx


def coskx2(k: int, x: Union[int, float]) -> Tuple[float, float]:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 2.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx, sinkx float: aproksymowana wartość funkcji,
                        NaN w przypadku błędnych danych wejściowych
    """
    if not isinstance(k, int) or (not isinstance(x, int) and not isinstance(x, float)):
        return np.NaN
    if k < 0:
        return np.NaN
    elif k == 0 or x == 0:
        return 1, 0
    elif k == 1:
        return np.cos(x), np.sin(x)
    elif k > 1:
        coskx = coskx2(1, x)[0] * coskx2(k - 1, x)[0] - coskx2(1, x)[1] * coskx2(k - 1, x)[1]
        sinkx = coskx2(1, x)[1] * coskx2(k - 1, x)[0] + coskx2(1, x)[0] * coskx2(k - 1, x)[1]
        return coskx, sinkx
