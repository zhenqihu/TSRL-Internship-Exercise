#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : utils.py
# @Author  : Zhenqi Hu
# @Date    : 28/11/2023 2:12 pm
import numpy as np


def vech(A):
    """
    Vectorization of the lower triangular part of a square matrix
    :param A: a square matrix (np.ndarray)
    :return: a vector (np.ndarray)
    """
    m, n = A.shape
    if m != n:
        raise ValueError("Input must be a square matrix")
    v = []
    for j in range(m):
        for i in range(j, m):
            v.append(A[i, j])
    return np.array(v)


def inv_vech(v, n=None):
    """
    Inverse of vech() operator, i.e. create the lower triangular matrix from a vector
    :param v: results of vech() operator, a vector (np.ndarray)
    :param n: dimension of the square matrix (int)
    :return: a square matrix (np.ndarray)
    """
    if n is None:
        n = int(round(0.5 * (np.sqrt(1 + 8 * len(v)) - 1)))
    if n * (n + 1) / 2 != len(v):
        raise ValueError("length(v) != n(n+1)/2")
    A = np.zeros((n, n))
    indices = np.cumsum(np.arange(n, 0, -1))
    indices = np.insert(indices, 0, 0)
    for j in range(n):
        for i in range(j, n):
            A[i, j] = v[indices[j] + i - j]
    return np.tril(A)


def all_fields_equal(x1, x2, fields):
    """
    Check if two objects have the same values for a list of fields
    :param x1: object 1
    :param x2: object 2
    :param fields: a list of fields
    :return: True if all fields are equal, False otherwise
    """
    return all(getattr(x1, field) == getattr(x2, field) for field in fields)


def maybe_call_function(f, *args):
    """
    Call a function if it is not None
    :param f: a function or None
    :param args: arguments of the function
    :return: None if f is None, otherwise the result of f(*args)
    """
    if f is None:
        return None
    else:
        return f(*args)


def fill_zeros(x):
    """
    Fill an object with zeros recursively, in place
    :param x: np.ndarray or list of np.ndarray
    :return: None
    """
    if x is None:
        return None
    elif isinstance(x, np.ndarray):
        x.fill(0)
    elif isinstance(x, list):
        for i in range(len(x)):
            fill_zeros(x[i])
    return None
