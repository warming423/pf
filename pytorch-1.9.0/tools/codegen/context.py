from tools.codegen.utils import S, T, context
from tools.codegen.model import (NativeFunction, NativeFunctionsGroup, ExternalBackendFunction,
                                 ExternalBackendFunctionsGroup)
import tools.codegen.local as local

import functools
from typing import TypeVar, Union, Iterator, Callable
import contextlib

# Helper functions for defining generators on things in the model

F = TypeVar(
    'F',
    NativeFunction,
    NativeFunctionsGroup,
    ExternalBackendFunction,
    ExternalBackendFunctionsGroup,
    Union[NativeFunction, NativeFunctionsGroup],
    Union[ExternalBackendFunctionsGroup, ExternalBackendFunction],
    Union[NativeFunction, NativeFunctionsGroup, ExternalBackendFunction, ExternalBackendFunctionsGroup]
)

@contextlib.contextmanager
def native_function_manager(g: Union[
        NativeFunctionsGroup, NativeFunction, ExternalBackendFunction, ExternalBackendFunctionsGroup]) -> Iterator[None]:
    if isinstance(g, ExternalBackendFunctionsGroup):
        f = g.primary.native_function
    elif isinstance(g, ExternalBackendFunction):
        f = g.native_function
    elif isinstance(g, NativeFunctionsGroup):
        # By default, we associate all errors with structured native functions
        # with the out variant.  In some cases, it might be better to have
        # a more specific place to hang things; if so, use
        # native_function_manager again on the inside
        f = g.out
    else:
        f = g
    with context(f'in native_functions.yaml line {f.loc}:\n  {f.func}'):
        with local.parametrize(use_const_ref_for_mutable_tensors=f.use_const_ref_for_mutable_tensors):
            yield

# Given a function that operates on NativeFunction, wrap it into a new function
# that sets some appropriate context managers for that native function.
# YOU MUST WRAP FUNCTIONS IN THIS for calls to api modules to be sound
# (you will get an error if we try to access the local variables without having
# set them).
def with_native_function(func: Callable[[F], T]) -> Callable[[F], T]:
    @functools.wraps(func)
    def wrapper(f: F) -> T:
        with native_function_manager(f):
            return func(f)
    return wrapper

def method_with_native_function(func: Callable[[S, F], T]) -> Callable[[S, F], T]:
    @functools.wraps(func)
    def wrapper(slf: S, f: F) -> T:
        with native_function_manager(f):
            return func(slf, f)
    return wrapper
