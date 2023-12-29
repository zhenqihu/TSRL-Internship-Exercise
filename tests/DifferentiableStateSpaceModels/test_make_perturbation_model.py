#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : test_make_perturbation_model.py
# @Author  : Zhenqi Hu
# @Date    : 29/12/2023 3:01 am

import pytest
import os
import sys
from DifferentiableStateSpaceModels import make_perturbation_model, default_model_cache_location, PerturbationModel, SolverCache
from sympy import symbols, Function, exp, Eq, oo, Matrix

sys.path.append(default_model_cache_location())


def test_basic_construction_of_model():
    # Define symbolic variables
    α, β, ρ, δ, σ, Ω_1, Ω_2 = symbols('α β ρ δ σ Ω_1 Ω_2')  # parameters
    t = symbols('t', integer=True)  # time index (integer)
    k, z, c, q = symbols('k z c q', cls=Function)  # undefined functions

    # Define the states, controls, and parameters
    x = [k, z]  # states - list of functions
    y = [c, q]  # controls - list of functions
    p = [α, β, ρ, δ, σ]  # parameters - list of symbols

    # Define the system of model equations - list of expressions
    H = [
        1 / c(t) - (β / c(t + 1)) * (α * exp(z(t + 1)) * k(t + 1) ** (α - 1) + (1 - δ)),
        c(t) + k(t + 1) - (1 - δ) * k(t) - q(t),
        q(t) - exp(z(t)) * k(t) ** α,
        z(t + 1) - ρ * z(t)
    ]

    # Define the steady states - list of equations
    steady_states = [
        Eq(k(oo), (((1 / β) - 1 + δ) / α) ** (1 / (α - 1))),
        Eq(z(oo), 0),
        Eq(c(oo), (((1 / β) - 1 + δ) / α) ** (α / (α - 1)) - δ * (((1 / β) - 1 + δ) / α) ** (1 / (α - 1))),
        Eq(q(oo), (((1 / β) - 1 + δ) / α) ** (α / (α - 1)))
    ]

    steady_states_iv = steady_states

    n_ϵ = 1
    n_z = 2
    n_x = len(x)
    n_y = len(y)
    n_p = len(p)
    Γ = Matrix([σ])
    η = Matrix([0, -1])
    Q = Matrix([[1.0, 0, 0, 0], [0, 0, 1.0, 0]])
    Ω = [Ω_1, Ω_2]

    model_name = "rbc_temp"
    print_level = 1
    max_order = 2

    make_perturbation_model(H, model_name=model_name, t=t, y=y, x=x, p=p,
                                                steady_states=steady_states, steady_states_iv=steady_states_iv,
                                                Γ=Γ, Ω=Ω, η=η, Q=Q, overwrite_model_cache=True,
                                                print_level=print_level, max_order=max_order)

    make_perturbation_model(H, t, y, x, steady_states, steady_states_iv, Γ,
                            Ω, η, Q, p, overwrite_model_cache=True, model_name=model_name)

    module_cache_path = make_perturbation_model(H, t, y, x, steady_states, steady_states_iv, Γ,
                                                Ω, η, Q, p,
                                                overwrite_model_cache=False, print_level=print_level,
                                                max_order=max_order, model_name=model_name)

    # Load the model from the cache
    exec(f"import {model_name}", globals())

    # Test the construction
    m = PerturbationModel(sys.modules[model_name])
    assert m.n_y == n_y
    assert m.max_order == max_order
    assert m.mod.m.n_z == n_z

    c1 = SolverCache.from_model(m, 2, ["α", "β"])
    c2 = SolverCache.from_model(m, 1, ["α", "β"])
    c3 = SolverCache.from_model(m, 2, ["α"])
    assert isinstance(c1, SolverCache)
    assert isinstance(c2, SolverCache)
    assert isinstance(c3, SolverCache)

