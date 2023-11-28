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
                            steady_states_iv=None, Gamma=None, Omega=None, eta=None, Q=None,
                            p=None, model_name=None,
                            model_cache_location=default_model_cache_location(),
                            overwrite_model_cache=False, print_level=1,
                            max_order=2, save_ip=True, save_oop=False,  # only does inplace by default
                            skipzeros=True, fillzeros=False, simplify_psi=True,
                            simplify=True, simplify_p=True):
    assert max_order in [1, 2]  # only implemented up to second order
    assert save_ip or save_oop  # save in place or out of place

    # path of the module to be saved in the cache
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
    assert n_p > 0  # code written to have at least one parameter
    n_epsilon = eta.shape[1]
    n_z = n if Q is None else Q.shape[0]

    # Get the markovian variables and create substitutions
    y_subs = pd.DataFrame([make_substitutions(t, y_i) for y_i in y])
    x_subs = pd.DataFrame([make_substitutions(t, x_i) for x_i in x])
    y = y_subs['var'].values
    x = x_subs['var'].values
    y_p = y_subs['var_p'].values
    x_p = x_subs['var_p'].values
    y_ss = y_subs['var_ss'].values
    x_ss = x_subs['var_ss'].values
    subs = pd.concat([x_subs, y_subs]).reset_index(drop=True)
    all_to_markov = pd.concat([subs['markov_t'], subs['markov_tp1'], subs['markov_inf']], axis=1).values
    all_to_var = pd.concat([subs['tp1_to_var'], subs['inf_to_var']], axis=1).values

    # Helper to take the [z(∞) ~ expr] and become [z => expr] after substitutions
