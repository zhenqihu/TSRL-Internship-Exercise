#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : make_perturbation_model.py
# @Author  : Zhenqi Hu
# @Date    : 29/11/2023 12:18 am
import os
from pathlib import Path
import DifferentiableStateSpaceModels
from DifferentiableStateSpaceModels import *
import numpy as np
import pandas as pd


def default_model_cache_location():
    """
    Returns the default location for the model cache
    :return: path of a folder joining the package directory and ".function_cache"
    """
    return os.path.join(Path(DifferentiableStateSpaceModels.__file__).parent.absolute(), ".function_cache")


def make_perturbation_model(H, t, y, x, steady_states=None,
                            steady_states_iv=None, Γ=None, Ω=None, η=None, Q=None,
                            p=None, model_name=None,
                            model_cache_location=default_model_cache_location(),
                            overwrite_model_cache=False, print_level=1,
                            max_order=2, save_ip=True, save_oop=False,  # only does inplace by default
                            skipzeros=True, fillzeros=False, simplify_Ψ=True,
                            simplify=True, simplify_p=True):

    assert max_order in [1, 2], "max_order must be 1 or 2"
    assert save_ip or save_oop, "Either save_ip or save_oop must be True"

    # path to save the model modules
    module_cache_path = os.path.join(model_cache_location, model_name + ".py")

    # only load cache if the module isn't already loaded in memory
    if (str(model_name) in globals()) and (not overwrite_model_cache):
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
    assert n_p > 0, "Code written to have at least one parameter"
    n_ϵ = η.shape[1]
    n_z = n if Q is None else Q.shape[0]

    # Get the markovian variables and create substitutions
    y_subs = [make_substitutions(t, y_i) for y_i in y]
    x_subs = [make_substitutions(t, x_i) for x_i in x]
    y = Matrix([y_i['var'] for y_i in y_subs])
    x = Matrix([x_i['var'] for x_i in x_subs])
    y_p = Matrix([y_i['var_p'] for y_i in y_subs])
    x_p = Matrix([x_i['var_p'] for x_i in x_subs])
    y_ss = Matrix([y_i['var_ss'] for y_i in y_subs])
    x_ss = Matrix([x_i['var_ss'] for x_i in x_subs])
    subs = x_subs + y_subs
    all_to_markov = [sub['markov_t'] for sub in subs] + [sub['markov_tp1']
                                                         for sub in subs] + [sub['markov_inf'] for sub in subs]
    all_to_var = [sub['tp1_to_var']
                  for sub in subs] + [sub['inf_to_var'] for sub in subs]

    # Helper to take the [z(∞) ~ expr] and become [z => expr] after substitutions
    def equations_to_dict(equations):
        return {str(eq.lhs.subs(all_to_markov).subs(all_to_var)): eq.rhs.subs(all_to_markov) for eq in equations}

    if print_level > 0:
        print("\033[96mBuilding model up to order {}\033[0m".format(max_order))
    if print_level > 1:
        print("\033[96msimplify = {}, simplify_p = {}, simplify Ψ = {}\033[0m".format(
            simplify, simplify_p, simplify_Ψ))

    # create functions in correct order
    y_bar = None if steady_states is None else order_vector_by_symbols(
        equations_to_dict(steady_states), [y_sub['symbol'] for y_sub in y_subs])
    x_bar = None if steady_states is None else order_vector_by_symbols(
        equations_to_dict(steady_states), [x_sub['symbol'] for x_sub in x_subs])
    y_bar_iv = None if steady_states_iv is None else order_vector_by_symbols(
        equations_to_dict(steady_states_iv), [y_sub['symbol'] for y_sub in y_subs])
    x_bar_iv = None if steady_states_iv is None else order_vector_by_symbols(
        equations_to_dict(steady_states_iv), [x_sub['symbol'] for x_sub in x_subs])
