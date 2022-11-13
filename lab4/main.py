from typing import Union, List, Tuple
import pickle
import numpy as np
import numpy.linalg as linalg
import numpy.random as random
from numpy.linalg import LinAlgError


class ParameterError(Exception):
    pass


def random_matrix_Ab(m: int):
    """Funkcja tworząca zestaw składający się z macierzy A (m,m) i wektora b (m,)  zawierających losowe wartości
    Parameters:
    m(int): rozmiar macierzy
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (m,m) i wektorem (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    try:
        a1 = np.random.rand(m, m)
        a2 = np.random.rand(m)
    except(ValueError, TypeError):
        return None
    return a1, a2


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray):
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b

      Parameters:
      A: macierz A (m,m) zawierająca współczynniki równania
      x: wektor x (m.) zawierający rozwiązania równania
      b: wektor b (m,) zawierający współczynniki po prawej stronie równania

      Results:
      (float)- wartość normy residuom dla podanych parametrów"""
    try:
        rn = linalg.norm(b - A @ x)
        return rn
    except ValueError:
        return None


def log_sing_value(n: int, min_order: Union[int, float], max_order: Union[int, float]):
    """Funkcja generująca wektor wartości singularnych rozłożonych w skali logarytmiczne
    
        Parameters:
         n(np.ndarray): rozmiar wektora wartości singularnych (n,), gdzie n>0
         min_order(int,float): rząd najmniejszej wartości w wektorze wartości singularnych
         max_order(int,float): rząd największej wartości w wektorze wartości singularnych
        Results:
         np.ndarray - wektor nierosnących wartości logarytmicznych o wymiarze (n,) zawierający wartości logarytmiczne
         na zadanym przedziale
         """
    try:
        if min_order >= max_order or n <= 0 or not isinstance(n, int):
            raise ValueError
        ls = np.logspace(min_order, max_order, num=n)
        return ls[::-1]
    except(ValueError, TypeError):
        return None


def order_sing_value(n: int, order: Union[int, float] = 2, site: str = 'gre'):
    """Funkcja generująca wektor losowych wartości singularnych (n,) będących wartościami zmiennoprzecinkowymi
        losowanymi przy użyciu funkcji np.random.rand(n)*10.
        A następnie ustawiająca wartość minimalną (site = 'low') albo maksymalną (site = 'gre') na wartość
        o  10**order razy mniejszą/większą.
    
        Parameters:
        n(np.ndarray): rozmiar wektora wartości singularnych (n,), gdzie n>0
        order(int,float): rząd przeskalowania wartości skrajnej
        site(str): zmienna wskazująca stronnę zmiany:
            - site = 'low' -> sing_value[-1] * 10**order
            - site = 'gre' -> sing_value[0] * 10**order
        
        Results:
        np.ndarray - wektor wartości singularnych o wymiarze (n,) zawierający wartości logarytmiczne
        na zadanym przedziale
        """
    try:
        if n <= 0 or not isinstance(n, int) or not isinstance(order, (int, float)):
            raise ValueError
        if site not in ['low', 'gre']:
            raise ParameterError
        sing_value = np.random.rand(n) * 10
        sing_value = np.sort(sing_value)
        if site == 'low':
            sing_value[-1] = sing_value[-1] * 10 ** order
        elif site == 'gre':
            sing_value[0] = sing_value[0] * 10 ** order
        sing_value = np.sort(sing_value)[::-1]
        return sing_value
    except(ValueError, ParameterError):
        return None


def create_matrix_from_A(A: np.ndarray, sing_value: np.ndarray):
    """Funkcja generująca rozkład SVD dla macierzy A i zwracająca otworzenie macierzy A z wykorzystaniem zdefiniowanego
        wektora warości singularnych

            Parameters:
            A(np.ndarray): rozmiarz macierzy A (m,m)
            sing_value(np.ndarray): wektor wartości singularnych (m,)


            Results:
            np.ndarray: macierz (m,m) utworzoną na podstawie rozkładu SVD zadanej macierzy A z podmienionym wektorem
            wartości singularnych na wektor sing_value """
    try:
        u, s, v = linalg.svd(A)
        result = np.dot(u * sing_value, v)
        return result
    except(ValueError, LinAlgError):
        return None
