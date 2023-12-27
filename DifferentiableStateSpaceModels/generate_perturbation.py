#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : generate_perturbation.py
# @Author  : Zhenqi Hu
# @Date    : 12/26/2023 4:31 PM

import numpy as np
from DifferentiableStateSpaceModels.utils import fill_zeros, maybe_call_function
from DifferentiableStateSpaceModels.symbolic_utils import order_vector_by_symbols
from DifferentiableStateSpaceModels.types import *

# Utility to call the cache's organized by symbol
def fill_array_by_symbol_dispatch(f, c, symbols, *args):
    for i, sym in enumerate(symbols):
        f(c[i], sym, *args)


def create_or_zero_cache(m, cache, Order, p_d, zero_cache):
    if cache is None:
        # create it if not supplied
        return SolverCache.from_model(m, Order, p_d)
    else:
        # otherwise conditionally zero it and return the argument
        if zero_cache:
            fill_zeros(cache)  # recursively works through cache and sub-types
        return cache


def verify_steady_state(m, p_d, p_f, atol=1e-8, *args):
    sol = generate_perturbation(m, p_d, p_f, *args)
    p_d_symbols = list(p_d.keys())
    p = order_vector_by_symbols({**p_d, **p_f}, m.mod.m.p_symbols)

    w = np.concatenate((sol.y, sol.x))  # get a vector for the proposed steadystate
    H = np.empty(len(w))  # allocate it, but leave undef to make sure we can see if it goes to 0 or not
    m.mod.m.H_bar(H, w, p)  # evaluate in place
    return np.linalg.norm(H) < atol


# The generate_perturbation function calculates the perturbation itself
# It can do used without any derivatives overhead (except, perhaps, extra memory in the cache)
def generate_perturbation(m, p_d, p_f, order, cache=None, zero_cache=False, settings=PerturbationSolverSettings()):
    c = create_or_zero_cache(m, cache, order, p_d, zero_cache)  # SolverCache object
    c.p_d_symbols = list(p_d.keys())

    p = p_d if p_f is None else order_vector_by_symbols({**p_d, **p_f}, m.mod.m.p_symbols)
    if order == 1:
        # First order solutions
        # solver type provided to all callbacks
        ret = calculate_steady_state(m, c, settings, p)
        maybe_call_function(settings.calculate_steady_state_callback, ret, m, c, settings, p)  # before returning
        if ret != 'Success':
            return FirstOrderPerturbationSolution(ret, m, c, settings)

        ret = evaluate_first_order_functions(m, c, settings, p)
        maybe_call_function(settings.evaluate_functions_callback, ret, m, c, settings, p)
        if ret != 'Success':
            return FirstOrderPerturbationSolution(ret, m, c, settings)

        ret = solve_first_order(m, c, settings)
        maybe_call_function(settings.solve_first_order_callback, ret, m, c, settings)
        if ret != 'Success':
            return FirstOrderPerturbationSolution(ret, m, c, settings)

        return FirstOrderPerturbationSolution('Success', m, c, settings)
    elif order == 2:
        # Second order solutions
        # Calculate the first-order perturbation
        sol_first = generate_perturbation(m, p_d, p_f, 1, cache=c, settings=settings, zero_cache=False)

        if sol_first.retcode != 'Success':
            return SecondOrderPerturbationSolution(sol_first.retcode, m, c, settings)

        # solver type provided to all callbacks
        ret = evaluate_second_order_functions(m, c, settings, p)
        if ret != 'Success':
            return SecondOrderPerturbationSolution(ret, m, c, settings)
        maybe_call_function(settings.evaluate_functions_callback, ret, m, c, settings, p)

        ret = solve_second_order(m, c, settings)
        if ret != 'Success':
            return SecondOrderPerturbationSolution(ret, m, c, settings)

        return SecondOrderPerturbationSolution('Success', m, c, settings)


def calculate_steady_state(m, c, settings, p):
    n_y, n_x = m.n_y, m.n_x
    n = n_y + n_x

    if settings.print_level > 2:
        print("Calculating steady state")

    #try:
    #    if m.mod.m.y_bar is not None and m.mod.m.x_bar is not None:
    #        # use the user-supplied steady state
    #        c.y_bar = m.mod.m.y_bar(p)
    #        c.x_bar = m.mod.m.x_bar(p)
    return None


def evaluate_first_order_functions(m, c, settings, p):
    return None


def evaluate_second_order_functions(m, c, settings, p):
    return None


def solve_first_order(m, c, settings):
    return None


def solve_second_order(m, c, settings):
    return None