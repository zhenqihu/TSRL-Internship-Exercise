# -*- coding: utf-8 -*-
# @File    : utils.py
# @Author  : Zhenqi Hu
# @Date    : 26/11/2023 12:33 am
import numpy as np
from scipy.linalg import solve_triangular


def vech(A: np.ndarray):
    """Returns the lower triangular part of a matrix as a vector.
    :param A: a square matrix
    :return: a vector
    """
    m, n = A.shape
    assert m == n, "Matrix must be square"
    v = []
    for j in range(m):
        for i in range(j, m):
            v.append(A[i, j])
    return np.array(v)


def inv_vech(v: np.ndarray, n=None):
    """Returns the inverse of vech(A) as a matrix.
    :param v: a vector
    :param n: the size of the square matrix A
    :return: lower-triangular square matrix A
    """
    if n is None:
        n = int(round(0.5 * (np.sqrt(1 + 8 * len(v)) - 1)))
    assert n * (n + 1) / 2 == len(v), "length(v) != n(n+1)/2"
    A = np.zeros((n, n))
    indices = np.cumsum([0] + list(range(n, 0, -1)))
    for j in range(n):
        for i in range(j, n):
            A[i, j] = v[indices[j] + i - j]
    return A


def all_fields_equal(x1, x2, fields):
    """Returns True if all fields of x1 and x2 are equal.
    :param x1: object 1
    :param x2: object 2
    :param fields: object fields to compare
    :return: Boolean
    """
    return all([getattr(x1, field) == getattr(x2, field) for field in fields])


def maybe_call_function(f, *args):
    """Calls a function if it is not None.
    :param f: a function to call
    :param args: arguments to pass to the function
    :return:
    """
    if f is not None:
        return f(*args)


def fill_zeros(x):
    """Fills an array or a list of arrays with zeros.
    :param x: a numpy array or a list of numpy arrays
    :return: None
    """
    if isinstance(x, np.ndarray):
        x.fill(0)
    elif isinstance(x, list):
        for i in range(len(x)):
            fill_zeros(x[i])
