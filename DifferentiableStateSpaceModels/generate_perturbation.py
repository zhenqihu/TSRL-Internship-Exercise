#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : generate_perturbation.py
# @Author  : Zhenqi Hu
# @Date    : 12/26/2023 4:31 PM

import numpy as np

# Utility to call the cache's organized by symbol
def fill_array_by_symbol_dispatch(f, c, symbols, *args):
    for i, sym in enumerate(symbols):
        f(c[i], sym, *args)


def create_or_zero_cache(m, cache, Order, p_d, zero_cache):
    if cache is None:
        # create it if not supplied
        return SolverCache(m, Order, p_d)
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
def generate_perturbation(m, p_d, p_f, order, cache=None, zero_cache=False, settings = PerturbationSolverSettings()):
    c = create_or_zero_cache(m, cache, order, p_d, zero_cache)
    c.p_d_symbols = list(p_d.keys())

    p = p_d if p_f is None else order_vector_by_symbols({**p_d, **p_f}, m.mod.m.p_symbols)