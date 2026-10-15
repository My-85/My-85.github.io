"""Narrow NumPy execution of the original A4 pure-JAX numerical operations."""
import dataclasses
from types import SimpleNamespace
import numpy as np


class ReplaceMixin:
    def replace(self, **values):
        return dataclasses.replace(self, **values)


def at_set(array, index, value):
    result = np.array(array, copy=True)
    result[index] = value
    return result


class _NP32:
    pi = np.float32(np.pi)
    def __getattr__(self, name):
        return getattr(np, name)
    @staticmethod
    def asarray(value, dtype=None, **kwargs):
        result = np.asarray(value, dtype=dtype, **kwargs)
        if dtype is None and not hasattr(value, 'dtype'):
            if result.dtype.kind == 'f':
                result = result.astype(np.float32)
            elif result.dtype.kind in 'iu':
                result = result.astype(np.int32)
        return result
    array = asarray
    @staticmethod
    def arange(*args, dtype=None, **kwargs):
        result = np.arange(*args, dtype=dtype, **kwargs)
        if dtype is None:
            result = result.astype(np.float32 if result.dtype.kind == 'f' else np.int32)
        return result
    @staticmethod
    def zeros(shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype)
    @staticmethod
    def ones(shape, dtype=np.float32):
        return np.ones(shape, dtype=dtype)
    @staticmethod
    def full(shape, fill_value, dtype=None):
        return np.full(shape, fill_value, dtype=dtype or (np.float32 if np.asarray(fill_value).dtype.kind == 'f' else np.int32))


jnp = _NP32()


def _stack_tree(values, axis=0):
    first = values[0]
    if isinstance(first, tuple):
        return tuple(_stack_tree([v[i] for v in values], axis) for i in range(len(first)))
    if isinstance(first, dict):
        return {k: _stack_tree([v[k] for v in values], axis) for k in first}
    return np.stack(values, axis=axis)


def vmap(function, in_axes=0, out_axes=0):
    def mapped(*args):
        axes = (in_axes,) * len(args) if isinstance(in_axes, (int, type(None))) else in_axes
        size = next(np.shape(a)[axis] for a, axis in zip(args, axes) if axis is not None)
        values = [function(*[a if axis is None else np.take(a, i, axis=axis) for a, axis in zip(args, axes)]) for i in range(size)]
        return _stack_tree(values, out_axes)
    return mapped


def fori_loop(lower, upper, body, value):
    for i in range(int(lower), int(upper)):
        value = body(i, value)
    return value


def cond(predicate, true_function, false_function, *args, **kwargs):
    if 'operand' in kwargs:
        args = (kwargs['operand'],)
    return (true_function if bool(np.asarray(predicate)) else false_function)(*args)


jax = SimpleNamespace(vmap=vmap, lax=SimpleNamespace(fori_loop=fori_loop, cond=cond, select=np.where))
