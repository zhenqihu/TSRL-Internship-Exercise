#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : test_utils.py
# @Author  : Zhenqi Hu
# @Date    : 28/11/2023 2:24 pm
import pytest
import numpy as np
from DifferentiableStateSpaceModels import vech, inv_vech, maybe_call_function, fill_zeros


def test_vech():
    """Test vech() function"""
    A = np.array([[1, 2], [3, 4]])
    assert np.array_equal(vech(A), np.array([1, 3, 4]))
    assert np.array_equal(vech(np.tril(A)), np.array([1, 3, 4]))
    with pytest.raises(ValueError):
        vech(np.array([[1, 2, 4], [3, 4, 5]]))


def test_inv_vech():
    """Test inv_vech() function"""
    A = np.array([[1, 2], [3, 4]])
    assert np.array_equal(inv_vech(np.array([1, 3, 4]), 2), np.tril(A))
    assert np.array_equal(inv_vech(np.array([1, 3, 4])), np.tril(A))
    with pytest.raises(ValueError):
        inv_vech(np.array([1, 3, 4, 5]))
    with pytest.raises(ValueError):
        inv_vech(np.array([1, 3, 4]), 3)


def test_vech_inv_vech():
    """Test vech() and inv_vech() functions"""
    B = np.tril(np.random.rand(10, 10))
    assert np.allclose(inv_vech(vech(B)), B)


def test_maybe_call_function():
    """Test maybe_call_function() function"""
    assert maybe_call_function(sum, [1, 2, 3]) == 6
    assert maybe_call_function(None, [1, 2, 3]) is None
    assert maybe_call_function(lambda x, y: x + y, 1, 2) == 3


def test_fill_zeros():
    """Test fill_zeros() function"""
    # Case 1: one-dimensional np.ndarray
    x = np.array([1, 2, 3])
    fill_zeros(x)
    assert np.array_equal(x, np.array([0, 0, 0]))

    # Case 2: None
    x = None
    assert fill_zeros(x) is None

    # Case 3: list of one-dimensional np.ndarray
    x = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    fill_zeros(x)
    assert np.array_equal(x[0], np.array([0, 0, 0]))
    assert np.array_equal(x[1], np.array([0, 0, 0]))

    # Case 4: two-dimensional np.ndarray
    x = np.array([[1, 2, 3], [4, 5, 6]])
    fill_zeros(x)
    assert np.array_equal(x, np.array([[0, 0, 0], [0, 0, 0]]))
