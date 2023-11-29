#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : test_symbolic_utils.py
# @Author  : Zhenqi Hu
# @Date    : 28/11/2023 4:48 pm
import pytest
from sympy import symbols, Function, sin
import numpy as np
import pandas as pd
from DifferentiableStateSpaceModels import make_substitutions, order_vector_by_symbols, substitute_and_simplify


def test_order_vector_by_symbols():
    """Test make_substitutions() and order_vector_by_symbols() functions"""
    # Define symbolic variables
    α, β, ρ, δ, σ = symbols('α β ρ δ σ')
    t = symbols('t', integer=True)

    # Define symbolic functions
    k, z, c, q = symbols('k z c q', cls=Function)

    # Define lists of symbolic functions and variables
    x = [k, z]
    y = [c, q]
    p = [α, β, ρ, δ, σ]

    # Apply make_substitutions function to each element in x and y
    subs_x = pd.DataFrame([make_substitutions(t, var) for var in x])
    subs_y = pd.DataFrame([make_substitutions(t, var) for var in y])

    # Concatenate subs_x and subs_y
    subs = pd.concat([subs_x, subs_y]).reset_index(drop=True)
    print(subs)

    # Convert p to symbols
    p_symbols = [str(var) for var in p]

    p_val = {'α': 0.1, 'β': 0.5, 'ρ': 0.1, 'δ': 1.9, 'σ': 1.9}
    p_val_2 = {'ρ': 0.1, 'α': 0.1, 'σ': 1.9, 'β': 0.5, 'δ': 1.9}
    p_vec = order_vector_by_symbols(p_val, p_symbols)
    p_vec_2 = order_vector_by_symbols(p_val_2, p_symbols)

    # Test if p_vec and p_vec_2 are approximately equal
    assert np.allclose(p_vec, p_vec_2)


def test_substitute_and_simplify():
    """Test substitute_and_simplify() function"""
    # Define symbolic variables
    x, y = symbols('x y')

    # Test with a single symbolic expression
    f = sin(x)
    subs = {x: y}
    result = substitute_and_simplify(f, subs)
    assert result == sin(y)

    # Test with a list of symbolic expressions
    f = [sin(x), x**2]
    result = substitute_and_simplify(f, subs)
    assert result == [sin(y), y**2]

    # Test with None
    f = None
    result = substitute_and_simplify(f, subs)
    assert result is None

    # Test with a dictionary of symbolic expressions
    f = {'expr1': sin(x), 'expr2': x**2}
    result = substitute_and_simplify(f, subs)
    assert result == {'expr1': sin(y), 'expr2': y**2}

    # Test with simplification
    f = x**2 + 2*x + 1
    subs = {x: y - 1}
    result = substitute_and_simplify(f, subs, do_simplify=True)
    assert result == y**2
