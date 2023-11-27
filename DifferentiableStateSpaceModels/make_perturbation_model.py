# -*- coding: utf-8 -*-
# @File    : make_perturbation_model.py
# @Author  : Zhenqi Hu
# @Date    : 25/11/2023 10:41 pm

import os
import numpy as np
from collections import OrderedDict
from sympy import symbols, simplify, Identity
import DifferentiableStateSpaceModels


def default_model_cache_location():
    return os.path.join(DifferentiableStateSpaceModels.__path__[0], ".function_cache")


def make_perturbation_model(H, t, y, x, steady_states=None,
                            steady_states_iv=None, Γ=None, Ω=None, η=None, Q=I,
                            p=None, model_name=None,
                            model_cache_location=default_model_cache_location(),
                            overwrite_model_cache=False, print_level=1,
                            max_order=2, save_ip=True, save_oop=False,  # only does inplace by default
                            skipzeros=True, fillzeros=False, simplify_Ψ=True,
                            simplify=True, simplify_p=True):
    """

    :param H: a system of equations
    :param t: time variable
    :param y: control variables
    :param x: state variables
    :param steady_states:
    :param steady_states_iv:
    :param Γ:
    :param Ω:
    :param η:
    :param Q:
    :param p:
    :param model_name:
    :param model_cache_location:
    :param overwrite_model_cache:
    :param print_level:
    :param max_order:
    :param save_ip:
    :param save_oop:
    :param skipzeros:
    :param fillzeros:
    :param simplify_Ψ:
    :param simplify:
    :param simplify_p:
    :return:
    """

    assert max_order in [1, 2]
    assert save_ip or save_oop

    module_cache_path = os.path.join(model_cache_location, model_name + ".jl")

    # only load cache if the module isn't already loaded in memory
    if (model_name in globals()) and (not overwrite_model_cache):
        if print_level > 0:
            print(f"Using existing module {model_name}\n")
        return module_cache_path

    # if path already exists
    if (os.path.exists(module_cache_path)) and (not overwrite_model_cache):
        # path exists and not overwriting
        if print_level > 0:
            print(f"Model already generated at {module_cache_path}\n")
        return module_cache_path

    n_y = len(y)
    n_x = len(x)
    n = n_y + n_x
    n_p = len(p)
    assert n_p > 0  # code written to have at least one parameter
    n_ϵ = η.shape[1]
    n_z = n if Q == Identity(n) else Q.shape[0]

    # TODO: error check that p, y, x has no overlap
    # Get the markovian variables and create substitutions
    y_subs = StructArray([make_substitutions(t, yi) for yi in y])
    x_subs = StructArray([make_substitutions(t, xi) for xi in x])
    y = y_subs.var
    x = x_subs.var
    y_p = y_subs.var_p
    x_p = x_subs.var_p
    y_ss = y_subs.var_ss
    x_ss = x_subs.var_ss
    subs = np.concatenate((x_subs, y_subs))
    all_to_markov = np.concatenate((subs.markov_t, subs.markov_tp1, subs.markov_inf))
    all_to_var = np.concatenate((subs.tp1_to_var, subs.inf_to_var))
    # Helper to take the [z(∞) ~ expr] and become [z => expr] after substitutions
    def equations_to_dict(equations):
        return OrderedDict((symbols(str(substitute(substitute(eq.lhs, all_to_markov), all_to_var))),
                            symbols(str(substitute(eq.rhs, all_to_markov)))) for eq in equations)
    if (print_level > 0):
        print(f"Building model up to order {max_order}\n")
    if (print_level > 1):
        print(f"simplify = {simplify}, simplify_p = {simplify_p}, simplify Ψ = {simplify_Ψ}\n")
    # create functions in correct order
    ȳ = None if steady_states is None else order_vector_by_symbols(equations_to_dict(steady_states), y_subs.symbol)
    x̄ = None if steady_states is None else order_vector_by_symbols(equations_to_dict(steady_states), x_subs.symbol)
    ȳ_iv = None if steady_states_iv is None else order_vector_by_symbols(equations_to_dict(steady_states_iv), y_subs.symbol)
    x̄_iv = None if steady_states_iv is None else order_vector_by_symbols(equations_to_dict(steady_states_iv), x_subs.symbol)