"""Method registry: maps ``--method`` to its implementation."""
from methods.base import Method
from methods.chronos_base import ChronosBaseMethod
from methods.raf import RAFMethod
from methods.cross_raf import CrossRAFMethod

_REGISTRY = {
    "none": ChronosBaseMethod,
    "raf": RAFMethod,
    "cross_raf": CrossRAFMethod,
}


def get_method(args) -> Method:
    if args.method not in _REGISTRY:
        raise ValueError(f"Unknown method: {args.method}. Choices: {list(_REGISTRY)}")
    return _REGISTRY[args.method](args)
