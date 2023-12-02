#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : make_perturbation_model.py
# @Author  : Zhenqi Hu
# @Date    : 29/11/2023 12:18 am
import os
from pathlib import Path
import DifferentiableStateSpaceModels
from DifferentiableStateSpaceModels import *
from sympy import Matrix, latex, hessian, simplify, lambdify
from copy import deepcopy


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
                            do_simplify=True, simplify_p=True):
    """
    Creates a perturbation model for a system of equations of the form
    :param H: system of equations (list of sympy expressions)
    :param t: time symbol (sympy symbol)
    :param y: control variables (list of sympy symbols; length = n_y)
    :param x: state variables (list of sympy symbols; length = n_x)
    :param steady_states: analytic solutions for the steady state (list of sympy Equations)
    :param steady_states_iv: steady state equations for initial values (list of sympy Equations)
    :param Γ: matrix for volatility of shocks (sympy Matrix; n_ϵ by n_ϵ)
    :param Ω: diagonal cholesky of covariance matrix for observation noise (so these are standard deviations).
              Non-diagonal observation noise not currently supported (sympy Matrix; n_ϵ by n_ϵ)
    :param η: matrix of the loading of the shocks for state variables (sympy Matrix; n_x by n_ϵ)
    :param Q: observables matrix (sympy Matrix; n_z by n)
    :param p: parameters (list of sympy symbols; length = n_p)
    :param model_name: name of the model (string)
    :param model_cache_location: directory to save the model modules (string)
    :param overwrite_model_cache: whether to overwrite the model cache (boolean)
    :param print_level: level of printing (int)
    :param max_order: maximum order of the model (int)
    :param save_ip: whether to save the model in place (boolean)
    :param save_oop: whether to save the model out of place (boolean)
    :param skipzeros:
    :param fillzeros:
    :param simplify_Ψ:
    :param do_simplify:
    :param simplify_p:
    :return:
    """

    # check inputs
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

    # model structure
    n_y = len(y)  # number of control variables
    n_x = len(x)  # number of state variables
    n = n_y + n_x  # number of all variables
    n_p = len(p)  # number of parameters
    assert n_p > 0, "Code written to have at least one parameter"
    n_ϵ = η.shape[1]  # number of shocks
    n_z = n if Q is None else Q.shape[0]  # number of observables

    # Get the markovian variables and create substitutions
    y_subs = [make_substitutions(t, y_i) for y_i in y]  # list of dictionaries to store substitutions for y
    x_subs = [make_substitutions(t, x_i) for x_i in x]  # list of dictionaries to store substitutions for x
    y = Matrix([y_i['var'] for y_i in y_subs])  # Matrix of y variables
    x = Matrix([x_i['var'] for x_i in x_subs])  # Matrix of x variables
    y_p = Matrix([y_i['var_p'] for y_i in y_subs])  # Matrix of y' variables
    x_p = Matrix([x_i['var_p'] for x_i in x_subs])  # Matrix of x' variables
    y_ss = Matrix([y_i['var_ss'] for y_i in y_subs])  # Matrix of y_ss variables
    x_ss = Matrix([x_i['var_ss'] for x_i in x_subs])  # Matrix of x_ss variables
    subs = x_subs + y_subs  # list of all substitutions
    all_to_markov = ([sub['markov_t'] for sub in subs] +
                     [sub['markov_tp1'] for sub in subs] + [sub['markov_inf'] for sub in subs])
    all_to_var = [sub['tp1_to_var'] for sub in subs] + [sub['inf_to_var'] for sub in subs]

    # Helper function to take the Equation and become i.e. {'expr': expr} after substitutions
    def equations_to_dict(equations):
        return {str(eq.lhs.subs(all_to_markov).subs(all_to_var)): eq.rhs.subs(all_to_markov) for eq in equations}

    # -------------------------------- Beginning building the model ---------------------------------------------------#
    if print_level > 0:
        print("\033[96mBuilding model up to order {}\033[0m".format(max_order))

    if print_level > 1:
        print("\033[96mdo_simplify = {}, simplify_p = {}, simplify Ψ = {}\033[0m".format(
            do_simplify, simplify_p, simplify_Ψ))

    # create functions in correct order (steady states & steady states initial values)
    y_bar = None if steady_states is None else order_vector_by_symbols(
        equations_to_dict(steady_states), [y_sub['symbol'] for y_sub in y_subs])
    x_bar = None if steady_states is None else order_vector_by_symbols(
        equations_to_dict(steady_states), [x_sub['symbol'] for x_sub in x_subs])
    y_bar_iv = None if steady_states_iv is None else order_vector_by_symbols(
        equations_to_dict(steady_states_iv), [y_sub['symbol'] for y_sub in y_subs])
    x_bar_iv = None if steady_states_iv is None else order_vector_by_symbols(
        equations_to_dict(steady_states_iv), [x_sub['symbol'] for x_sub in x_subs])

    # Get any latex generated stuff we wish for pretty display of the model
    H_latex = latex(H)
    steady_states_latex = latex(steady_states)
    steady_states_iv_latex = latex(steady_states_iv) if steady_states_iv is not None else None

    # steady state requires differentiation after substitution, and wrt [y; x]
    H = [expr.subs(all_to_markov) for expr in H]
    H_bar = deepcopy(H)
    H_bar = substitute_and_simplify(H_bar, all_to_var)

    # Differentiate the system of equations with respect to state/control variables
    if print_level > 2:
        print("\033[96mDifferentiating H\033[0m")
    H_bar_w = nested_differentiate(H_bar, y.col_join(x))
    H_yp = nested_differentiate(H, y_p)
    H_y = nested_differentiate(H, y)
    H_xp = nested_differentiate(H, x_p)
    H_x = nested_differentiate(H, x)

    # Calculate the Hessian for each function in H
    if print_level > 1:
        print("\033[96mCalculating hessian\033[0m")
    Psi = [hessian(f, y_p.col_join(y).col_join(x_p).col_join(x)) for f in H]
    if simplify_Ψ:
        Psi = [simplify(psi) for psi in Psi]

    # Differentiate the Hessian with respect to state/control variables
    if print_level > 2 and max_order >= 2:
        print("\033[96mDifferentiating hessian\033[0m")
    Psi_yp = None if max_order < 2 else nested_differentiate(Psi, y_p)
    Psi_y = None if max_order < 2 else nested_differentiate(Psi, y)
    Psi_xp = None if max_order < 2 else nested_differentiate(Psi, x_p)
    Psi_x = None if max_order < 2 else nested_differentiate(Psi, x)

    # Differentiate steady states with respect to parameters
    if print_level > 2:
        print("\033[96mDifferentiating steady state with respect to parameters\n\033[0m")
    p = Matrix(p)  # DenseMatrix
    H_p = differentiate_to_dict(H, p)
    Γ_p = differentiate_to_dict(Γ, p)
    Ω_p = differentiate_to_dict(Ω, p)
    y_bar_p = differentiate_to_dict(y_bar, p)
    x_bar_p = differentiate_to_dict(x_bar, p)

    # Differentiate H derivatives with respect to parameters
    if print_level > 2:
        print("\033[96mDifferentiating H derivatives state with respect to parameters\n\033[0m")
    H_yp_p = differentiate_to_dict(H_yp, p)
    H_xp_p = differentiate_to_dict(H_xp, p)
    H_y_p = differentiate_to_dict(H_y, p)
    H_x_p = differentiate_to_dict(H_x, p)

    # Differentiate Hessian with respect to parameters
    if print_level > 2 and max_order >= 2:
        print("\033[96mDifferentiating hessian with respect to parameters\n\033[0m")
    Psi_p = None if max_order < 2 else differentiate_to_dict(Psi, p)

    # Substitute and simplify
    if print_level > 0:
        print("\033[96mSubstituting and simplifying\n\033[0m")
    H = substitute_and_simplify(H, all_to_markov, do_simplify)
    H_yp = substitute_and_simplify(H_yp, all_to_var, do_simplify)
    H_xp = substitute_and_simplify(H_xp, all_to_var, do_simplify)
    H_x = substitute_and_simplify(H_x, all_to_var, do_simplify)
    H_y = substitute_and_simplify(H_y, all_to_var, do_simplify)
    Psi = substitute_and_simplify(Psi, all_to_var, do_simplify)

    # Substitute and simplify parameters derivatives
    if print_level > 2:
        print("\033[96mSubstituting and simplifying parameter derivatives\n\033[0m")
    H_p = substitute_and_simplify(H_p, all_to_var, simplify_p)
    Γ_p = substitute_and_simplify(Γ_p, all_to_var, simplify_p)
    Ω_p = substitute_and_simplify(Ω_p, all_to_var, simplify_p)
    y_bar_p = substitute_and_simplify(y_bar_p, all_to_var, simplify_p)
    x_bar_p = substitute_and_simplify(x_bar_p, all_to_var, simplify_p)
    H_yp_p = substitute_and_simplify(H_yp_p, all_to_var, simplify_p)
    H_xp_p = substitute_and_simplify(H_xp_p, all_to_var, simplify_p)
    H_y_p = substitute_and_simplify(H_y_p, all_to_var, simplify_p)
    H_x_p = substitute_and_simplify(H_x_p, all_to_var, simplify_p)

    # Substitute and simplify second order of Hessian
    if print_level > 1 and max_order >= 2:
        print("\033[96mSubstituting and simplifying 2nd order\033[0m")
    Psi_yp = substitute_and_simplify(Psi_yp, all_to_var, do_simplify)
    Psi_y = substitute_and_simplify(Psi_y, all_to_var, do_simplify)
    Psi_xp = substitute_and_simplify(Psi_xp, all_to_var, do_simplify)
    Psi_x = substitute_and_simplify(Psi_x, all_to_var, do_simplify)

    # Substitute and simplify second order derivatives wrt parameters of Hessian
    if print_level > 2 and max_order >= 2:
        print("\033[96mSubstituting and simplifying 2nd order parameter derivatives\n\033[0m")
    Psi_p = substitute_and_simplify(Psi_p, all_to_var, simplify_p)

    # ------------------------------ Beginning building the model function --------------------------------------------#
    # Generate all functions, and rename using utility
    def build_named_functions(expr, name, *args, symbol_dispatch=None):
        if expr is None:
            return None
        if isinstance(expr, dict):
            return {key: build_named_functions(value, name, *args, symbol_dispatch=key)
                    for key, value in expr.items()}
        elif isinstance(expr, Iterable):
            return [build_named_functions(value, name, *args, symbol_dispatch=symbol_dispatch)
                    for value in expr]
        else:
            # create the function from the expression
            f = lambdify(args, expr, modules=["numpy", "scipy", "tensorflow"])
            func = name_symbolics_function(f, name, symbol_dispatch=symbol_dispatch)  # rename the function
        return func