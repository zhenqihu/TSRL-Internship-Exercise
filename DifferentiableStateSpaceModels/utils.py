#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : utils.py
# @Author  : Zhenqi Hu
# @Date    : 28/11/2023 2:12 pm
import numpy as np
from DifferentiableStateSpaceModels.types import FirstOrderSolverBuffers, FirstOrderDerivativeSolverBuffers, SecondOrderSolverBuffers, SecondOrderDerivativeSolverBuffers, SolverCache

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
    elif isinstance(x, FirstOrderSolverBuffers):
        fill_zeros(x.A)
        fill_zeros(x.B)
        fill_zeros(x.Z)
        fill_zeros(x.Z_ll)
        fill_zeros(x.S_bb)
        fill_zeros(x.T_bb)
    elif isinstance(x, FirstOrderDerivativeSolverBuffers):
        fill_zeros(x.R)
        fill_zeros(x.A)
        fill_zeros(x.C)
        fill_zeros(x.D)
        fill_zeros(x.E)
        fill_zeros(x.dH)
        fill_zeros(x.bar)
    elif isinstance(x, SecondOrderSolverBuffers):
        fill_zeros(x.A)
        fill_zeros(x.B)
        fill_zeros(x.C)
        fill_zeros(x.D)
        fill_zeros(x.E)
        fill_zeros(x.R)
        fill_zeros(x.A_σ)
        fill_zeros(x.R_σ)
    elif isinstance(x, SecondOrderDerivativeSolverBuffers):
        fill_zeros(x.A)
        fill_zeros(x.B)
        fill_zeros(x.C)
        fill_zeros(x.D)
        fill_zeros(x.E)
        fill_zeros(x.R)
        fill_zeros(x.dH)
        fill_zeros(x.dΨ)
        fill_zeros(x.gh_stack)
        fill_zeros(x.g_xx_flat)
        fill_zeros(x.Ψ_x_sum)
        fill_zeros(x.Ψ_y_sum)
        fill_zeros(x.bar)
        fill_zeros(x.kron_h_x)
        fill_zeros(x.R_p)
        fill_zeros(x.A_σ)
        fill_zeros(x.R_σ)
    elif isinstance(x, SolverCache):
        fill_zeros(x.H)
        fill_zeros(x.H_yp)
        fill_zeros(x.H_y)
        fill_zeros(x.H_xp)
        fill_zeros(x.H_x)
        fill_zeros(x.H_yp_p)
        fill_zeros(x.H_y_p)
        fill_zeros(x.H_xp_p)
        fill_zeros(x.H_x_p)
        fill_zeros(x.H_p)
        fill_zeros(x.Γ)
        fill_zeros(x.Γ_p)
        fill_zeros(x.Σ)
        fill_zeros(x.Σ_p)
        fill_zeros(x.Ω)
        fill_zeros(x.Ω_p)
        fill_zeros(x.Ψ)
        fill_zeros(x.x)
        fill_zeros(x.y)
        fill_zeros(x.y_p)
        fill_zeros(x.x_p)
        fill_zeros(x.g_x)
        fill_zeros(x.h_x)
        fill_zeros(x.g_x_p)
        fill_zeros(x.h_x_p)
        fill_zeros(x.B)
        fill_zeros(x.B_p)
        fill_zeros(x.A_1_p)
        fill_zeros(x.C_1)
        fill_zeros(x.C_1_p)
        fill_zeros(x.V)
        fill_zeros(x.V_p)
        fill_zeros(x.η_Σ_sq)
        fill_zeros(x.Ψ_p)
        fill_zeros(x.Ψ_yp)
        fill_zeros(x.Ψ_y)
        fill_zeros(x.Ψ_xp)
        fill_zeros(x.Ψ_x)
        fill_zeros(x.g_xx)
        fill_zeros(x.h_xx)
        fill_zeros(x.g_σσ)
        fill_zeros(x.h_σσ)
        fill_zeros(x.g_xx_p)
        fill_zeros(x.h_xx_p)
        fill_zeros(x.g_σσ_p)
        fill_zeros(x.h_σσ_p)
        fill_zeros(x.A_0_p)
        fill_zeros(x.A_2_p)
        fill_zeros(x.C_0)
        fill_zeros(x.C_2)
        fill_zeros(x.C_0_p)
        fill_zeros(x.C_2_p)

        # Buffers for additional calculations
        fill_zeros(x.first_order_solver_buffer)
        fill_zeros(x.first_order_solver_p_buffer)
        fill_zeros(x.second_order_solver_buffer)
        fill_zeros(x.second_order_solver_p_buffer)
    return None
