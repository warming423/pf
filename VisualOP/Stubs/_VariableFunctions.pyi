# ${generated_comment}
# mypy: disable-error-code="type-arg"

import builtins
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import torch
from torch import contiguous_format, Generator, inf, memory_format, strided, SymInt, Tensor
from torch.types import (
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Device,
    Number,
)

from torch._prims_common import DeviceLikeType

${function_hints}

${all_directive}
