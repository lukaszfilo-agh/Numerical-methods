import numpy as np
import scipy as sp
import pickle
import numpy.linalg as linalg

from typing import Union, List, Tuple, Optional


def diag_dominant_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Macierz A ma być diagonalnie zdominowana, tzn. wyrazy na przekątnej sa wieksze od pozostałych w danej kolumnie i wierszu
    Parameters:
    m int: wymiary macierzy i wektora

    Returns:
    Tuple[np.ndarray, np.ndarray]: macierz diagonalnie zdominowana o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    try:
        if m <= 0:
            raise ValueError
        a = np.random.randint(1, 100, (m, m))
        b = np.random.randint(0, 9, (m,))
        max = np.sum(a, axis=1) - np.diag(a)
        a = a + np.diag(max)
        return a, b
    except (TypeError, ValueError):
        return None


def is_diag_dominant(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest diagonalnie zdominowana
    Parameters:
    A np.ndarray: macierz wejściowa

    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    try:
        if not isinstance(A, np.ndarray):
            raise ValueError
        if len(A.shape) != 2:
            raise ValueError
        if A.shape[0] != A.shape[1]:
            raise ValueError
        if A.shape[0] == 1:
            return True
        d = np.diag(A)
        max = np.sum(A, axis=1) - d
        if np.all(d > max):
            return True
        else:
            return False
    except ValueError:
        return None


def symmetric_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Parameters:
    m int: wymiary macierzy i wektora

    Returns:
    Tuple[np.ndarray, np.ndarray]: symetryczną macierz o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    try:
        if m <= 0:
            raise ValueError
        a = np.random.randint(1, 100, (m, m))
        b = np.random.randint(0, 9, (m,))
        a = np.tril(a)
        a = a + a.T
        return a, b
    except (TypeError, ValueError):
        return None


def is_symmetric(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest symetryczna
    Parameters:
    A np.ndarray: macierz wejściowa

    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    try:
        if not isinstance(A, np.ndarray):
            raise ValueError
        if len(A.shape) != 2:
            raise ValueError
        if A.shape[0] != A.shape[1]:
            raise ValueError
        if A.shape[0] == 1:
            return True
        if np.all(A - A.T == 0):
            return True
        else:
            return False
    except ValueError:
        return None


def solve_jacobi(A: np.ndarray, b: np.ndarray, x_init: np.ndarray,
                 epsilon: Optional[float] = 1e-8, maxiter: Optional[int] = 100) -> Tuple[np.ndarray, int]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych
    Parameters:
    A np.ndarray: macierz współczynników
    b np.ndarray: wektor wartości prawej strony układu
    x_init np.ndarray: rozwiązanie początkowe
    epsilon Optional[float]: zadana dokładność
    maxiter Optional[int]: ograniczenie iteracji

    Returns:
    np.ndarray: przybliżone rozwiązanie (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    int: iteracja
    """
    try:
        if not isinstance(A, np.ndarray):
            raise ValueError
        if A.shape[0] != A.shape[1] or len(b) != A.shape[0] or maxiter < 1:
            raise ValueError
        D = np.diag(np.diag(A))
        LU = A - D
        x = x_init
        D_inv = np.diag(1 / np.diag(D))
        resid = []
        for i in range(maxiter):
            x_new = np.dot(D_inv, b - np.dot(LU, x))
            r_norm = np.linalg.norm(x_new - x)
            resid.append(r_norm)
            if r_norm < epsilon:
                return x_new, resid
            x = x_new
        return x, resid
    except (ValueError, TypeError):
        return None


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
    except (ValueError, TypeError):
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


def solve_req(A: np.ndarray) -> bool:
    if A.shape[0] == A.shape[1]:
        return True
    else:
        return False


def jacobi_req(A: np.ndarray) -> bool:
    if is_diag_dominant(A) == True:
        return True
    else:
        return False


def cg_req(A: np.ndarray) -> bool:
    if is_symmetric(A) == True:
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def gmres_req(A: np.ndarray) -> bool:
    if A.shape[0] == A.shape[1]:
        return True
    else:
        return False
