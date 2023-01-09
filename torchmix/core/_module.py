import functools
import inspect
import os
import re
from collections.abc import Mapping, Sequence
from inspect import Signature
from pathlib import Path
from typing import Any, Dict, Optional

from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate
from hydra_zen.typing import Builds, Partial
from omegaconf import DictConfig, OmegaConf
from torch import nn
from typing_extensions import Self

from torchmix.core._builds import BuildMode, builds

NO_PARAMS = False
GLOBAL_KWARGS: Dict[str, Any] = {}


def _extract_args_from_signature(sig: Signature):
    args = tuple(
        name
        for name, param in sig.parameters.items()
        if param.default is param.empty
        and name != "self"
        and param.kind != param.VAR_KEYWORD
        # TODO: VAR_POSITIONAL must be handled as well.
    )
    return args


def _extract_kwargs_from_signature(sig: Signature):
    # Extracting keyword-only arguments and their default values
    # Cannot use param.kind here.
    # See https://stackoverflow.com/questions/57848612
    kwargs = {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not param.empty
    }
    return kwargs


def _underscore(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    name = name.replace("-", "_")
    return name.lower()


def _parse(value):
    if isinstance(value, Component):
        return value.config
    elif isinstance(value, Partial):
        return builds(
            value.func,
            *value.args,
            **value.keywords,
            zen_partial=True,
        )
    elif isinstance(value, Mapping):
        return builds(
            type(value),
            {k: _parse(v) for k, v in value.items()},
            populate_full_signature=False,
        )
    elif isinstance(value, str):
        return value
    elif isinstance(value, Sequence):
        return builds(
            type(value),
            [_parse(elem) for elem in value],
        )

    return value


class Component(nn.Module):
    """
    A PyTorch module wrapper that automates the process of generating and
    managing configurations for PyTorch modules.

    This class inherits from PyTorch's `nn.Module` and can be used as
    a drop-in replacement. It automatically generates configurations
    for any instance of the `Component` class or its subclasses and
    stores them in a `DictConfig` object.

    The generated configurations can be accessed via the `config` property and
    exported to a file using the `export` method.

    The `store` method can be used to store the configurations in
    `hydra`'s ConfigStore for use with `hydra`'s CLI.
    """

    _config: DictConfig
    _option_name: str

    build_mode: BuildMode = BuildMode.WITHOUT_ARGS

    def __new__(cls, *args, **kwargs):
        """
        Overrides the default behavior of `Component`'s constructor. This
        method is called before the actual `__init__` method and is used to
        build the `config` attribute of the `Component` instance.

        This method is responsible for generating the configuration for a
        `Component` instance before calling the original `__init__` method.
        It does this by creating a new `__init__` method that calls
        `self.builds` with the provided arguments after recursively parsing
        them with the `_parse` function.

        The `_parse` function converts any `Component` instances found in
        the arguments to their corresponding configurations, as well as
        handling any partial instances and wrapping dictionaries and sequences
        in their respective types.

        The resulting configuration is then stored in `self._config`. The new
        `__init__` method then calls the original `__init__` method with the
        provided arguments.

        When `no_parameters` context manager is active, this method will
        skip calling the original `__init__` method, and the `Component`
        instance will not have any additional attributes or behavior beyond
        the `config` attribute.

        This can be useful if you want to create a `Component` instance
        solely for the purpose of generating its configuration, without
        actually instantiating the module. For more details, see
        `torchmix.core._context.no_parameters` context manager.

        Returns:
            The `Component` instance.
        """
        old_init = cls.__init__
        old_sig = inspect.signature(old_init)
        default_args = _extract_args_from_signature(old_sig)
        default_kwargs = _extract_kwargs_from_signature(old_sig)

        def new_init(self, *args, **kwargs):
            if cls.build_mode is BuildMode.WITHOUT_ARGS:
                args_offset = len(args) - len(default_args)
                num_args_to_be_provided = len(default_args)
                kwargs_keys = list(default_kwargs)

                if args_offset < 0:
                    for i in range(-args_offset):
                        if default_args[i + len(args)] in kwargs:
                            num_args_to_be_provided -= 1
                        else:
                            raise TypeError(
                                f"{cls.__name__}.__init__() missing "
                                f"{-args_offset} "
                                "required positional arguments: "
                                f"""'{"', '".join(default_args[args_offset:])}'"""
                            )

                kwargs_as_pos = {
                    kwargs_keys[i]: args[i + len(default_args)]
                    for i in range(args_offset)
                }

                got_multiple_values = set(kwargs_as_pos).intersection(kwargs)

                if got_multiple_values:
                    raise TypeError(
                        f"{cls.__name__}.__init__() got multiple values for argument "
                        f"""'{"', '".join(got_multiple_values)}'"""
                    )

                default_kwargs.update(kwargs)
                default_kwargs.update(kwargs_as_pos)

                args = args[:num_args_to_be_provided]
                kwargs = default_kwargs

            args_list = list(args)
            for key in GLOBAL_KWARGS:
                if key in default_args[: len(args)]:
                    args_list[default_args.index(key)] = GLOBAL_KWARGS[key]
                elif key in kwargs:
                    kwargs[key] = GLOBAL_KWARGS[key]

            _args = ()
            _kwargs = {}

            for arg in args:
                _args += (_parse(arg),)

            for key, value in kwargs.items():
                _kwargs[key] = _parse(value)

            self._config = self.builds(*_args, **_kwargs)

            if not NO_PARAMS:
                nn.Module.__init__(self)
                old_init(self, *args, **kwargs)

            cls.__init__ = old_init

        new_init.__signature__ = old_sig.replace(
            parameters=list(old_sig.parameters.values())
        )

        cls.__init__ = new_init
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def builds(cls, *args, **kwargs):
        """
        Generates a configuration for an instance of the `Component` class
        or its subclasses.

        This method is used internally to generate the configuration for
        a `Component` instance. It should not be called directly.

        Returns:
            A `DictConfig` object containing the generated configuration for
            the instance.
        """
        if cls.build_mode is BuildMode.WITHOUT_ARGS:
            return builds(cls, **kwargs)(*args)
        elif cls.build_mode is BuildMode.WITH_ARGS:
            return builds(cls, *args, **kwargs)
        else:
            raise ValueError(
                "build_mode must be one of WITHOUT_ARGS or WITH_ARGS"
            )

    @classmethod
    def partial(cls, *args, **kwargs):
        """
        Returns a `functools.partial` object that can be used to partially
        instantiate the `Component` class or its subclasses.

        Returns:
            A `functools.partial` object that can be called like a function to
            create a `Component` instance.
        """
        return functools.partial(cls, *args, **kwargs)

    @property
    def config(self) -> Builds:
        """
        Returns the `DictConfig` object containing the configuration for the
        `Component` instance.

        This property allows users to access the generated configuration for
        the `Component` instance.
        """
        return self._config

    def export(
        self,
        path: Optional[str] = None,
    ) -> Self:  # type: ignore
        """
        Exports the configuration for the `Component` instance to a file.

        This method allows users to save the configuration for the
        `Component` instance to a file in YAML format.

        If no path is provided, the configuration will be saved to
        a file with the name of the class in the current working directory.

        Parameters:
            path: Path to the file where the configuration should be saved.

        Returns:
            The `Component` instance.
        """

        if path:
            _, extension = os.path.splitext(path)
            if extension:
                path = path
            else:
                if not self.option_name:
                    path = f"{path}.yaml"
                else:
                    Path(path).mkdir(parents=True, exist_ok=True)
                    path = f"{path}/{self.option_name}.yaml"
        else:
            if not self.option_name:
                path = path = f"{_underscore(type(self).__name__)}.yaml"
            else:
                path = self.option_name + ".yaml"

        OmegaConf.save(self.config, path)

        return self

    @property
    def option_name(self) -> Optional[str]:
        try:
            return self._option_name
        except AttributeError:
            return None

    def store(self, group: str, name: str) -> Self:  # type: ignore
        """
        Stores the configuration for the `Component` instance in
        `hydra`'s ConfigStore.
        """
        self._option_name = name

        cs = ConfigStore.instance()
        cs.store(group=group, name=name, node=self.config)

        return self

    def instantiate(self) -> Self:  # type: ignore
        """
        Recreates an instance of the Component from its configuration.

        This method recreates an instance of the `Component` by calling
        `hydra_zen.instantiate` on the `config` attribute of the
        `Component` instance. It returns the recreated instance.

        This can be useful if you want to instantiate a `Component`
        multiple times using the same configuration.

        Returns:
            The recreated instance of the `Component`.
        """
        return instantiate(self.config)
