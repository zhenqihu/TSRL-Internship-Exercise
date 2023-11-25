# -*- coding: utf-8 -*-
# @File    : symbolic_utils.py
# @Author  : Zhenqi Hu
# @Date    : 26/11/2023 1:15 am
from sympy import symbols, simplify, diff, Matrix, Function


def make_substitutions(t, f_var):
    """Returns a dictionary of substitutions for a variable and its derivatives.
    :param t: time variable
    :param f_var: function of t
    :return: dictionary of substitutions
    """
    sym_name = f_var.func.__name__
    sym_name_p = sym_name + "_p"
    sym_name_ss = sym_name + "_ss"
    names = symbols(f"{sym_name} {sym_name_p} {sym_name_ss}")
    return {'symbol': sym_name, 'var': names[0], 'var_p': names[1], 'var_ss': names[2],
            'markov_t': (f_var(t), names[0]), 'markov_tp1': (f_var(t + 1), names[1]),
            'markov_inf': (f_var(float('inf')), names[2]), 'tp1_to_var': (names[1], names[0]),
            'inf_to_var': (names[2], names[0])}


def order_vector_by_symbols(x, symbol):
    """Returns a vector ordered by a list of symbols.
    :param x: a vector
    :param symbol: a list of symbols
    :return: a vector
    """
    return [x[sym] for sym in symbol]


def substitute_and_simplify(f, subs, simplify_expr=False):
    """Substitutes variables in an expression and (optionally) simplifies it.
    :param f: an expression or a dictionary of expressions or a list of expressions
    :param subs: a dictionary of substitutions
    :param simplify_expr: (optional) whether to simplify the expression (default: False
    :return: an expression or a dictionary of expressions or a list of expressions
    """
    if f is None:
        return None
    elif isinstance(f, dict):
        return {key: substitute_and_simplify(value, subs, simplify_expr) for key, value in f.items()}
    elif isinstance(f, list):
        return [substitute_and_simplify(value, subs, simplify_expr) for value in f]
    else:
        f = f.subs(subs)
        return simplify(f) if simplify_expr else f


def nested_differentiate(f, x, simplify_expr=False):
    """Differentiates an expression (or a list of expressions) with respect to a variable (or a list of variables).
    :param f: an expression or a list of expressions
    :param x: a variable or a list of variables
    :param simplify_expr: a boolean indicating whether to simplify the expression (default: False)
    :return: an expression or a list of expressions
    """
    if f is None or x is None:
        return None
    elif isinstance(f, list) and isinstance(f[0], Matrix) and isinstance(x, list):
        return [nested_differentiate(f, x_val, simplify_expr) for x_val in x]
    elif isinstance(f, list) and isinstance(f[0], Matrix):
        return [diff(f_val, x).doit() for f_val in f]
    elif isinstance(f, Matrix) and isinstance(x, list):
        return [[diff(f, var).doit() for var in x]]
    elif isinstance(f, Matrix):
        return diff(f, x).doit()
    elif isinstance(f, list) and isinstance(x, list):
        return [[diff(f_val, var).doit() for f_val in f] for var in x]
    elif isinstance(f, list):
        return [diff(f_val, x).doit() for f_val in f]
    else:
        return diff(f, x).doit()


def differentiate_to_dict(f, p):
    """Differentiates an expression with respect to a list of parameters and returns a dictionary.
    :param f: a function or a list of functions expressed as sympy expressions
    :param p: a list of parameters
    :return: a dictionary of derivatives
    """
    if f is None or p is None:
        return None
    else:
        return {str(p_val): nested_differentiate(f, p_val) for p_val in p}


def name_symbolics_function(expr, name, inplace=False, symbol_dispatch=None, striplines=True):
    if symbol_dispatch is not None:
        dispatch_position = 1 if inplace else 0
        expr.args[dispatch_position] = symbols(f"Val_{symbol_dispatch}")
    expr.func = Function(name)
    return expr
