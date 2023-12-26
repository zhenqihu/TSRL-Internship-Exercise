#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : types.py
# @Author  : Zhenqi Hu
# @Date    : 12/26/2023 1:30 PM


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

