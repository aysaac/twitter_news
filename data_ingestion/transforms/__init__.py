import functools
from typing import Callable, Any

__all__ = ["build_transform"]


# ´NAME_TRANSFORMATION_MAP´ holds a mapping from transformation name to
# transformation function. Currently is empty, but it is filled at runtime by
# a decorator mechanic
NAME_TRANSFORMATION_MAP = {}


# Definition of the decorator that it's used to register functions to
# ´NAME_TRANSFORMATION_MAP´
def register(name):
    """Register a function to ´NAME_TRANSFORMATION_MAP´"""

    def register_wrappper(func):
        NAME_TRANSFORMATION_MAP[name] = func

        @functools.wraps(func)
        def function_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return function_wrapper

    return register_wrappper


def register_function(name: str, funct: Callable[[Any], Any]) -> None:
    """Register a function in ´NAME_TRANSFORMATION_MAP´. Useful for foreign
    functions"""
    NAME_TRANSFORMATION_MAP[name] = funct#func


from .basic_transformations import *
from .composed_transformations import *
from .transformation_builder import build_transform
