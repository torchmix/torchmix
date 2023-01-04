import contextlib
import unittest.mock


@contextlib.contextmanager
def no_parameters():
    """
    Context manager that temporarily patches `MixModul.__new__` method to
    skip calling the original `__init__` method of `MixModul` instances.

    This can be useful if you want to create a `MixModul` instance
    solely for the purpose of generating its configuration, without actually
    instantiating the module.

    Example:
        >>> with torchmix.no_parameters():
        ...     model = torchmix.nn.Linear(100, 200)
        >>> print(model._parameters)
        AttributeError: 'Linear' object has no attribute '_parameters'
    """
    with unittest.mock.patch("torchmix.core._module.NO_PARAMS", True):
        yield
