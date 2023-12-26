#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : types.py
# @Author  : Zhenqi Hu
# @Date    : 12/26/2023 1:30 PM
import numpy as np
from abc import ABC
from collections.abc import Iterable

# A wrapper for Module.
# The only purpose is a specialization of deepcopy since otherwise the "mod" property in the PerturbationModule brakes multithreaded MCMC
class ModuleWrapper:
    def __init__(self, m):
        self.m = m

    def __deepcopy__(self, stackdict):
        if self in stackdict:
            return stackdict[self]
        else:
            clone = type(self)(self.m)
            stackdict[self] = clone
            return clone


# Model Types.  The template args are required for inference for cache/perturbation solutions
class PerturbationModel:
    def __init__(self, mod):
        self.mod = ModuleWrapper(mod)
        self.max_order = mod.max_order
        self.n_y = mod.n_y
        self.n_x = mod.n_x
        self.n_p = mod.n_p
        self.n_ϵ = mod.n_ϵ
        self.n_z = mod.n_z
        self.has_Ω = mod.has_Ω
        self.η = mod.η
        self.Q = mod.Q

    def __str__(self):
        return f"Perturbation Model: n_y = {self.n_y}, n_x = {self.n_x}, n_p = {self.n_p}, n_ϵ = {self.n_ϵ}, n_z = {self.n_z}\n y = {self.mod.m.y_symbols} \n x = {self.mod.m.x_symbols} \n p = {self.mod.m.p_symbols}"

    def model_H_latex(self):
        return self.mod.m.H_latex

    def model_steady_states_latex(self):
        return self.mod.m.steady_states_latex

    def model_steady_states_iv_latex(self):
        return self.mod.m.steady_states_iv_latex


# Buffers for the solvers to reduce allocations
# General rule for cache vs. buffers
# 1. If something should be used in multiple parts of the algorithm, put it in the cache
# 2. Otherwise, use the buffers, which you can "trash" with inplace operations as required
# 3. The cache should never be modified after it has been filled in a given sequence of events, buffers can be

class FirstOrderSolverBuffers:
    def __init__(self, n_y, n_x, n_p_d, n_ϵ, n_z):
        self.A = np.zeros((n_x + n_y, n_x + n_y), dtype=np.complex128)
        self.B = np.zeros((n_x + n_y, n_x + n_y), dtype=np.complex128)
        self.Z = np.zeros((n_x + n_y, n_x + n_y))
        self.Z_ll = np.zeros((n_y, n_y))
        self.S_bb = np.triu(np.zeros((n_x, n_x)))
        self.T_bb = np.triu(np.zeros((n_x, n_x)))


class FirstOrderDerivativeSolverBuffers:
    def __init__(self, n_y, n_x, n_p_d, n_ϵ, n_z):
        self.R = np.zeros((2 * (n_x + n_y), n_x))
        self.A = np.zeros((n_x + n_y, n_x + n_y))
        self.C = np.zeros((n_x + n_y, n_x + n_y))
        self.D = np.zeros((n_x, n_x))
        self.E = np.zeros((n_x + n_y, n_x))
        self.dH = np.zeros((n_x + n_y, 2 * (n_x + n_y)))
        self.bar = np.zeros((2 * (n_x + n_y), 1))


class SecondOrderSolverBuffers:
    def __init__(self, n_y, n_x, n_p_d, n_ϵ, n_z):
        self.A = np.zeros((n_x + n_y, n_x + n_y))
        self.B = np.zeros((n_x**2, n_x**2))
        self.C = np.zeros((n_x + n_y, n_x + n_y))
        self.D = np.zeros((n_x**2, n_x**2))
        self.E = np.zeros((n_x + n_y, n_x**2))
        self.R = np.zeros((2 * (n_x + n_y), n_x))
        self.A_σ = np.zeros((n_x + n_y, n_x + n_y))
        self.R_σ = np.zeros((2 * (n_x + n_y), n_x))


class SecondOrderDerivativeSolverBuffers:
    def __init__(self, n_y, n_x, n_p_d, n_ϵ, n_z):
        self.A = np.zeros((n_x + n_y, n_x + n_y))
        self.B = np.zeros((n_x**2, n_x**2))
        self.C = np.zeros((n_x + n_y, n_x + n_y))
        self.D = np.zeros((n_x**2, n_x**2))
        self.E = np.zeros((n_x + n_y, n_x**2))
        self.R = np.zeros((2 * (n_x + n_y), n_x))
        self.dH = np.zeros((n_x + n_y, 2 * (n_x + n_y)))
        self.dΨ = [np.zeros((2 * (n_x + n_y), 2 * (n_x + n_y))) for _ in range(n_x + n_y)]
        self.gh_stack = np.zeros((n_x + n_y, n_x**2))
        self.g_xx_flat = np.zeros((n_y, n_x**2))
        self.Ψ_x_sum = [[np.zeros((2 * (n_x + n_y), 2 * (n_x + n_y))) for _ in range(n_x + n_y)] for _ in range(n_x)]
        self.Ψ_y_sum = [[np.zeros((2 * (n_x + n_y), 2 * (n_x + n_y))) for _ in range(n_x + n_y)] for _ in range(n_y)]
        self.bar = np.zeros((2 * (n_x + n_y), 1))
        self.kron_h_x = np.zeros((n_x**2, n_x**2))
        self.R_p = np.zeros((2 * (n_x + n_y), n_x))
        self.A_σ = np.zeros((n_x + n_y, n_x + n_y))
        self.R_σ = np.zeros((2 * (n_x + n_y), n_x))


class AbstractSolverCache(ABC):
    pass


class SolverCache(AbstractSolverCache):
    def __init__(self, Order, HasΩ, N_p_d, N_y, N_x, N_ϵ, N_z, Q, η):
        self.order = Order
        self.p_d_symbols = [None] * N_p_d
        self.H = np.zeros(N_x + N_y)
        self.H_yp = np.zeros((N_x + N_y, N_y))
        self.H_y = np.zeros((N_x + N_y, N_y))
        self.H_xp = np.zeros((N_x + N_y, N_x))
        self.H_x = np.zeros((N_x + N_y, N_x))
        self.H_yp_p = [np.zeros((N_x + N_y, N_y)) for _ in range(N_p_d)]
        self.H_y_p = [np.zeros((N_x + N_y, N_y)) for _ in range(N_p_d)]
        self.H_xp_p = [np.zeros((N_x + N_y, N_x)) for _ in range(N_p_d)]
        self.H_x_p = [np.zeros((N_x + N_y, N_x)) for _ in range(N_p_d)]
        self.H_p = [np.zeros(N_x + N_y) for _ in range(N_p_d)]
        self.Γ = np.zeros((N_ϵ, N_ϵ))
        self.Γ_p = [np.zeros((N_ϵ, N_ϵ)) for _ in range(N_p_d)]
        self.Σ = np.zeros((N_ϵ, N_ϵ))
        self.Σ_p = [np.zeros((N_ϵ, N_ϵ)) for _ in range(N_p_d)]
        self.Ω = None if not HasΩ else np.zeros(N_z)
        self.Ω_p = None if not HasΩ else [np.zeros(N_z) for _ in range(N_p_d)]
        self.Ψ = [np.zeros((2*(N_x + N_y), 2*(N_x + N_y))) for _ in range(N_x + N_y)]

        # Used in solution
        self.y = np.zeros(N_y)
        self.x = np.zeros(N_x)
        self.y_p = [np.zeros(N_y) for _ in range(N_p_d)]
        self.x_p = [np.zeros(N_x) for _ in range(N_p_d)]
        self.g_x = np.zeros((N_y, N_x))
        self.h_x = np.zeros((N_x, N_x))
        self.g_x_p = [np.zeros((N_y, N_x)) for _ in range(N_p_d)]
        self.h_x_p = [np.zeros((N_x, N_x)) for _ in range(N_p_d)]
        self.B = np.zeros((N_x, N_ϵ))
        self.B_p = [np.zeros((N_x, N_ϵ)) for _ in range(N_p_d)]
        self.Q = Q
        self.η = η
        self.A_1_p = [np.zeros((N_x, N_x)) for _ in range(N_p_d)]
        self.C_1 = np.zeros((N_z, N_x))
        self.C_1_p = [np.zeros((N_z, N_x)) for _ in range(N_p_d)]
        self.V = np.zeros((N_x, N_x))
        self.V_p = [np.zeros((N_x, N_x)) for _ in range(N_p_d)]
        self.η_Σ_sq = np.zeros((N_x, N_x))

        # Additional for 2nd order
        self.Ψ_p = None if Order < 2 else [[np.zeros((2 * (N_x + N_y), 2 * (N_x + N_y))) for _ in range(N_x + N_y)] for _ in range(N_p_d)]
        self.Ψ_yp = None if Order < 2 else [[np.zeros((2 * (N_x + N_y), 2 * (N_x + N_y))) for _ in range(N_x + N_y)] for _ in range(N_y)]
        self.Ψ_y = None if Order < 2 else [[np.zeros((2 * (N_x + N_y), 2 * (N_x + N_y))) for _ in range(N_x + N_y)] for _ in range(N_y)]
        self.Ψ_xp = None if Order < 2 else [[np.zeros((2 * (N_x + N_y), 2 * (N_x + N_y))) for _ in range(N_x + N_y)] for _ in range(N_x)]
        self.Ψ_x = None if Order < 2 else [[np.zeros((2 * (N_x + N_y), 2 * (N_x + N_y))) for _ in range(N_x + N_y)] for _ in range(N_x)]
        self.g_xx = None if Order < 2 else np.zeros((N_y, N_x, N_x))
        self.h_xx = None if Order < 2 else np.zeros((N_x, N_x, N_x))
        self.g_σσ = None if Order < 2 else np.zeros(N_y)
        self.h_σσ = None if Order < 2 else np.zeros(N_x)
        self.g_xx_p = None if Order < 2 else [np.zeros((N_y, N_x, N_x)) for _ in range(N_p_d)]
        self.h_xx_p = None if Order < 2 else [np.zeros((N_x, N_x, N_x)) for _ in range(N_p_d)]
        self.g_σσ_p = None if Order < 2 else np.zeros(N_y, N_p_d)
        self.h_σσ_p = None if Order < 2 else np.zeros(N_x, N_p_d)

        # Additional for solution type 2nd order
        self.A_0_p = None if Order < 2 else np.zeros(N_x, N_p_d)
        self.A_2_p = None if Order < 2 else [np.zeros((N_x, N_x, N_x)) for _ in range(N_p_d)]
        self.C_0 = None if Order < 2 else np.zeros(N_z)
        self.C_2 = None if Order < 2 else np.zeros((N_z, N_x, N_x))
        self.C_0_p = None if Order < 2 else np.zeros(N_z, N_p_d)
        self.C_2_p = None if Order < 2 else [np.zeros((N_z, N_x, N_x)) for _ in range(N_p_d)]

        # Buffers for additional calculations
        self.first_order_solver_buffer = FirstOrderSolverBuffers(N_y, N_x, N_p_d, N_ϵ, N_z)
        self.first_order_solver_p_buffer = FirstOrderDerivativeSolverBuffers(N_y, N_x, N_p_d, N_ϵ, N_z)
        self.second_order_solver_buffer = None if Order < 2 else SecondOrderSolverBuffers(N_y, N_x, N_p_d, N_ϵ, N_z)
        self.second_order_solver_p_buffer = None if Order < 2 else SecondOrderDerivativeSolverBuffers(N_y, N_x, N_p_d, N_ϵ, N_z)
        self.I_x = np.eye(N_x)
        self.I_x_2 = np.eye(N_x**2)
        self.zeros_x_x = np.zeros((N_x, N_x))
        self.zeros_y_x = np.zeros((N_y, N_x))

    @classmethod
    def from_model(cls, m, order, p_d):
        return cls(order, m.has_Ω, len(p_d), m.n_y, m.n_x, m.n_ϵ, m.n_z, m.Q, m.η)


class PerturbationSolverSettings:
    def __init__(self, rethrow_exceptions=False, print_level=1, ϵ_BK=1e-6, tol_cholesky=1e9, perturb_covariance=0.0,
                 singular_covariance_value=1e-12, calculate_ergodic_distribution=True, nlsolve_method='trust_region',
                 nlsolve_iterations=1000, nlsolve_show_trace=False, nlsolve_ftol=1e-8, use_solution_cache=True,
                 evaluate_functions_callback=None, calculate_steady_state_callback=None,
                 solve_first_order_callback=None, solve_first_order_p_callback=None, solve_second_order_callback=None,
                 solve_second_order_p_callback=None, sylvester_solver='MatrixEquations'):
        self.rethrow_exceptions = rethrow_exceptions
        self.print_level = print_level
        self.ϵ_BK = ϵ_BK
        self.tol_cholesky = tol_cholesky
        self.perturb_covariance = perturb_covariance
        self.singular_covariance_value = singular_covariance_value
        self.calculate_ergodic_distribution = calculate_ergodic_distribution
        self.nlsolve_method = nlsolve_method
        self.nlsolve_iterations = nlsolve_iterations
        self.nlsolve_show_trace = nlsolve_show_trace
        self.nlsolve_ftol = nlsolve_ftol
        self.use_solution_cache = use_solution_cache
        self.evaluate_functions_callback = evaluate_functions_callback
        self.calculate_steady_state_callback = calculate_steady_state_callback
        self.solve_first_order_callback = solve_first_order_callback
        self.solve_first_order_p_callback = solve_first_order_p_callback
        self.solve_second_order_callback = solve_second_order_callback
        self.solve_second_order_p_callback = solve_second_order_p_callback
        self.sylvester_solver = sylvester_solver


def nlsolve_options(s):
    """
    Returns a dictionary with some of the attributes of a PerturbationSolverSettings instance
    :param s: `PerturbationSolverSettings` object
    :return:
    """
    return {'method': s.nlsolve_method, 'iterations': s.nlsolve_iterations,
            'show_trace': s.nlsolve_show_trace, 'ftol': s.nlsolve_ftol}


# State Space types
class AbstractPerturbationSolution(ABC):
    pass


class AbstractFirstOrderPerturbationSolution(AbstractPerturbationSolution):
    pass


class AbstractSecondOrderPerturbationSolution(AbstractPerturbationSolution):
    pass


def make_covariance_matrix(x):
    if isinstance(x, Iterable):
        return [np.square(np.abs(x_i)) for x_i in x]
    else:
        return x


class FirstOrderPerturbationSolution(AbstractFirstOrderPerturbationSolution):
    def __init__(self, retcode, m, c, settings):
        """
        :param retcode:
        :param m: PerturbationModel
        :param c: SolverCache
        :param settings: PerturbationSolverSettings
        """
        self.retcode = retcode
        self.x_symbols = m.mod.m.x_symbols
        self.y_symbols = m.mod.m.y_symbols
        self.p_symbols = m.mod.m.p_symbols
        self.p_d_symbols = c.p_d_symbols
        self.u_symbols = m.mod.m.u_symbols
        self.n_y = m.n_y
        self.n_x = m.n_x
        self.n_p = m.n_p
        self.n_ϵ = m.n_ϵ
        self.n_z = m.n_z
        self.y = c.y
        self.x = c.x
        self.g_x = c.g_x
        self.A = c.h_x
        self.B = c.B
        self.C = c.C_1
        self.D = make_covariance_matrix(c.Ω)
        self.Q = c.Q
        self.η = c.η
        if settings.calculate_ergodic_distribution and retcode == "Success":
            self.x_ergodic_var = c.V
        else:
            self.x_ergodic_var = np.diag(settings.singular_covariance_value * np.ones(m.n_x))
        self.Γ = c.Γ