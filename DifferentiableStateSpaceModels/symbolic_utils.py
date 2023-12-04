#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : symbolic_utils.py
# @Author  : Zhenqi Hu
# @Date    : 28/11/2023 3:15 pm
from sympy import symbols, simplify, diff, Function, Matrix, MatrixBase
from collections.abc import Iterable
import inspect


def make_substitutions(t, f_var):
    """
    Substitutes the function f_var with its symbolic representation
    :param t: time symbol
    :param f_var: function to substitute
    :return: dictionary with the following keys: symbol, var, var_p, var_ss, markov_t, markov_tp1, markov_inf,
    tp1_to_var, inf_to_var
    """
    sym_name = f_var.name  # get the `name` attribute of the sympy `Function`, which is a string
    sym_name_p = sym_name + "_p"
    sym_name_ss = sym_name + "_ss"
    names = symbols(f"{sym_name} {sym_name_p} {sym_name_ss}")
    return {'symbol': sym_name, 'var': names[0], 'var_p': names[1], 'var_ss': names[2],
            'markov_t': (f_var(t), names[0]), 'markov_tp1': (f_var(t + 1), names[1]),
            'markov_inf': (f_var(float('inf')), names[2]), 'tp1_to_var': (names[1], names[0]),
            'inf_to_var': (names[2], names[0])}


def order_vector_by_symbols(x, symbol_list):
    """
    Orders a vector according to a list of symbols
    :param x: dictionary of symbols
    :param symbol_list: list of symbols
    :return: a new Matrix ordered according to the list of symbols
    """
    return Matrix([x[sym] for sym in symbol_list])


def substitute_and_simplify(f, subs, do_simplify=False):
    """
    Substitutes the symbols in f with the values in subs and (optionally) simplifies the result
    :param f: multiple types - sympy expression, list, dict or None
    :param subs: substitution pairs (symbol, value), or list of pairs, or dictionary
    :param do_simplify: boolean indicating whether to simplify
    :return: the result of substituting the symbols in f with the values in subs and (optionally) simplifying the result
    """
    if f is None:
        return None
    elif isinstance(f, dict):
        return {key: substitute_and_simplify(value, subs, do_simplify) for key, value in f.items()}
    elif isinstance(f, Iterable):
        return [substitute_and_simplify(value, subs, do_simplify) for value in f]
    else:
        result = f.subs(subs)
        return simplify(result) if do_simplify else result


# TODO: figure out how to make this function work with sympy matrices
def nested_differentiate(f, x, do_simplify=False):
    if f is None or x is None:
        return None

    if isinstance(f, MatrixBase):
        # f is a sympy matrix
        if isinstance(x, MatrixBase):
            return Matrix([simplify(diff(f, x_i).doit()) if do_simplify else diff(f, x_i).doit() for x_i in x])
        else:
            return simplify(diff(f, x).doit()) if do_simplify else diff(f, x).doit()
    elif isinstance(f, list):
        if isinstance(f[0], MatrixBase):
            # f is a list of sympy matrices (e.g. Psi)
            if isinstance(x, MatrixBase):
                return [nested_differentiate(f, x_i, do_simplify) for x_i in x]
            else:
                return [nested_differentiate(f_i, x, do_simplify) for f_i in f]
        else:
            # f is a list of sympy expressions (e.g. H, system of equations)
            if isinstance(x, MatrixBase):
                return Matrix([simplify(diff(f_i, x).doit().T) if do_simplify else diff(f_i, x).doit().T for f_i in f])
            else:
                return Matrix([simplify(diff(f, x).doit()) if do_simplify else diff(f, x).doit() for f in f])


def differentiate_to_dict(f, p):
    """
    Differentiates f with respect to each element in p and stores the result in a dictionary
    :param f: sympy expression
    :param p: parameter vector
    :return: a dictionary with the following keys: parameter value, derivative
    """
    if f is None or p is None:
        return None
    return {str(p_val): nested_differentiate(f, p_val) for p_val in p}
