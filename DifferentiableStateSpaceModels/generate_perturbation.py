#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : generate_perturbation.py
# @Author  : Zhenqi Hu
# @Date    : 12/26/2023 4:31 PM

import numpy as np
from DifferentiableStateSpaceModels.utils import fill_zeros, maybe_call_function
from DifferentiableStateSpaceModels.symbolic_utils import order_vector_by_symbols
from DifferentiableStateSpaceModels.types import *
from scipy.optimize import fsolve
from scipy.linalg import schur, lu, qz, ordqz
from scipy.linalg import solve_continuous_lyapunov as lyapd


def create_or_zero_cache(m, cache, order, p_d, zero_cache):
    if cache is None:
        # create it if not supplied
        return SolverCache.from_model(m, order, p_d)
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

    p = list(p_d.values()) if p_f is None else order_vector_by_symbols({**p_d, **p_f}, m.mod.m.p_symbols)
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
    try:
        if (m.mod.m.y_bar is not None) and (m.mod.m.x_bar is not None):  # use closed form if possible
            c.y = m.mod.m.y_bar(p)
            c.x = m.mod.m.x_bar(p)
            for i, sym in enumerate(c.p_d_symbols):
                f_y = getattr(m.mod.m, 'y_bar_p' + '_' + sym)
                if f_y is not None:
                    c.y_p[i] = f_y(p)

                f_x = getattr(m.mod.m, 'x_bar_p' + '_' + sym)
                if f_x is not None:
                    c.x_p[i] = f_x(p)
        elif m.mod.m.steady_state is not None:  # use user-provided calculation otherwise
            c.y, c.x = m.mod.m.steady_state(p)
        else:  # fallback is to solve system of equations from user-provided initial condition
            # TODO: solve steady state
            y_0 = np.zeros(n_y)
            x_0 = np.zeros(n_x)
            if m.mod.m.y_bar_iv is not None:
                y_0 = m.mod.m.y_bar_iv(p)
            if m.mod.m.x_bar_iv is not None:
                x_0 = m.mod.m.x_bar_iv(p)
            w_0 = np.concatenate((y_0, x_0))
    except Exception as e:
        if settings.rethrow_exceptions:
            raise e
        elif isinstance(e, np.linalg.LinAlgError):  # equivalent to LAPACKException in Julia
            if settings.print_level > 0:
                print(e)
            return 'LAPACK_Error'
        elif isinstance(e, np.linalg.LinAlgError):  # equivalent to PosDefException in Julia
            if settings.print_level > 0:
                print(e)
            return 'POSDEF_EXCEPTION'
        elif isinstance(e, ValueError):  # equivalent to DomainError in Julia
            if settings.print_level > 0:
                print(e)
            return 'Evaluation_Error'  # function evaluation error
        else:
            if settings.print_level > 0:
                print(e)
            return 'FAILURE'  # generic failure
    return "Success"


def evaluate_first_order_functions(m, c, settings, p):
    if settings.print_level > 2:
        print("Evaluating first-order functions into cache")
    try:
        y, x = c.y, c.x  # Precondition: valid (y, x) steady states
        c.H_yp = m.mod.m.H_yp(y, x, p)
        c.H_y = m.mod.m.H_y(y, x, p)
        c.H_xp = m.mod.m.H_xp(y, x, p)
        c.H_x = m.mod.m.H_x(y, x, p)
        c.Γ = m.mod.m.Γ(p)
        c.Ω = maybe_call_function(m.mod.m.Ω, p)
        if len(c.p_d_symbols) > 0:
            c.Ψ = m.mod.m.Ψ(y, x, p)
    except Exception as e:
        if settings.rethrow_exceptions:
            raise e
        elif isinstance(e, ValueError):  # equivalent to DomainError in Julia
            if settings.print_level > 0:
                print(e)
            return 'Evaluation_Error'  # function evaluation error
        else:
            if settings.print_level > 0:
                print(e)
            return 'FAILURE'  # generic failure
    return "Success"


def evaluate_second_order_functions(m, c, settings, p):
    if settings.print_level > 2:
        print("Evaluating second-order functions into cache")
    try:
        y, x = c.y, c.x  # Precondition: valid (y, x) steady states
        if len(c.p_d_symbols) == 0:
            c.Ψ = m.mod.m.Ψ(y, x, p)  # would have been called otherwise in first_order_functions
        c.Ψ_yp = m.mod.m.Ψ_yp(y, x, p)
        c.Ψ_y = m.mod.m.Ψ_y(y, x, p)
        c.Ψ_xp = m.mod.m.Ψ_xp(y, x, p)
        c.Ψ_x = m.mod.m.Ψ_x(y, x, p)
    except Exception as e:
        if settings.rethrow_exceptions:
            raise e
        elif isinstance(e, ValueError):  # equivalent to DomainError in Julia
            if settings.print_level > 0:
                print(e)
            return 'Evaluation_Error'  # function evaluation error
        else:
            if settings.print_level > 0:
                print(e)
            return 'FAILURE'  # generic failure
    return "Success"  # no failing code-paths yet.


def solve_first_order(m, c, settings):
    ϵ_BK, print_level = settings.ϵ_BK, settings.print_level
    n_x, n_y, n_p, n_ϵ = m.n_x, m.n_y, m.n_p, m.n_ϵ
    n = n_x + n_y

    if settings.print_level > 2:
        print("Solving first order perturbation")
    try:
        buff = c.first_order_solver_buffer
        buff.A = np.concatenate((c.H_yp, c.H_xp), axis=1)
        buff.B = np.concatenate((c.H_y, c.H_x), axis=1)
        if settings.print_level > 3:
            print("Calculating schur")
        S, T, Q, Z = qz(buff.A, buff.B, output='complex')  # Generalized Schur decomposition
        buff.A = S  # overwrite A with S
        buff.B = T  # overwrite B with T
        α = np.diag(S)  # diagonal of S
        β = np.diag(T)  # diagonal of T
        # The generalized eigenvalues λ_i are S_ii / T_ii
        # Following Blanchard-Kahn condition, we reorder the Schur so that
        # S_22 ./ T_22 < 1, ie, the eigenvalues < 1 come last
        # inds = [s.α[i] / s.β[i] >= 1 for i in 1:n]
        inds = np.abs(α) >= (1 - ϵ_BK) * np.abs(β)
        if np.sum(inds) != n_x:
            if print_level > 0:
                print("Blanchard-Kahn condition not satisfied\n")
            if settings.rethrow_exceptions:
                raise Exception("Failure of the Blanchard Khan Condition")
            else:
                return 'Blanchard_Kahn_Failure'
    except Exception as e:
        if settings.rethrow_exceptions:
            raise e
        else:
            if settings.print_level > 0:
                print(e)
            return 'FAILURE'
    return "Success"


def solve_second_order(m, c, settings):
    return "Success"