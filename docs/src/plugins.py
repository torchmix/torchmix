import inspect
import shutil
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Optional

from docstring_parser import parse

import torchmix
from torchmix import Component
from torchmix.components.attention import AttentionPlugin
from torchmix.components.mlp import MLPPlugin

# root = Path("docs/pages")
# root.mkdir(parents=True, exist_ok=True)

plugins = Path("docs/pages/plugins")
shutil.rmtree(plugins)
plugins.mkdir(parents=True, exist_ok=True)


def signature(method: Any):
    signature = inspect.signature(method)
    params = signature.parameters

    method_signature = ", ".join(
        f"{param.name}: {param.annotation.__module__}.{param.annotation.__name__}"
        for param in params.values()
        if param.name not in ("self", "_")
    )

    return (
        f"({method_signature}) -> {signature.return_annotation.__module__}."
        f"{signature.return_annotation.__name__}"
    )


def write(file: TextIOWrapper):
    def _write(content: Optional[str] = None):
        try:
            file.write(content + "\n")
        except:
            file.write("\n")

    return _write


# index = root / "index.mdx"
# index.write_text("Welcome to torchmix!")

# with open(root / "_meta.json", "w") as f:


names = []
attention_plugin_method = [
    "pre_qkv",
    "post_qkv",
    "pre_split_heads",
    "post_split_heads",
    "pre_scaled_dot",
    "post_scaled_dot",
    "pre_softmax",
    "post_softmax",
    "pre_weighted_value",
    "post_weighted_value",
    "pre_combine_heads",
    "post_combine_heads",
    "pre_projection",
    "post_projection",
]
for name, obj in inspect.getmembers(torchmix.components.attention):
    try:
        if issubclass(obj, AttentionPlugin):
            print(obj)
            names.append((name, obj))
            doc = parse(obj.__doc__)
            with open(plugins / f"{name}.mdx", "w") as f:
                _w = write(f)
                _w('import { Callout } from "nextra-theme-docs"')
                _w('import { Tab, Tabs } from "nextra-theme-docs"')
                _w()
                _w(f"# {name} (Attention)")
                _w()
                _w(doc.short_description)
                _w()
                _w("```python copy")
                _w(doc.examples[0].description)
                _w("```")
                _w()
                _w(doc.long_description)
                _w()

                if doc.params:
                    _w("## Parameters")
                    _w()
                    for p in doc.params:
                        if p.default:
                            _w(
                                f"- `{p.arg_name} = {p.default}`: {p.description}"
                            )
                        else:
                            _w(f"- `{p.arg_name}`: {p.description}")

                    _w()

                # for _name, _obj in inspect.getmembers(obj):
                #     for method_name in attention_plugin_method:
                #         if (
                #             _obj.__qualname__
                #             != getattr(
                #                 AttentionPlugin, method_name
                #             ).__qualname__
                #         ):
                #             method = getattr(_obj, _name)
                #             _w(f"## {_name}")
                #             _w("```rust")  # Hack for syntax highlighting
                #             _w(f"{signature(method)}")
                #             _w("```")
                #             _w()
                #             forward_doc = parse(method.__doc__)
                #             if forward_doc.returns:
                #                 _w("### Returns")
                #                 _w(forward_doc.returns.description)

    except Exception as e:
        pass

for name, obj in inspect.getmembers(torchmix.components.mlp):
    try:
        if issubclass(obj, MLPPlugin):
            names.append((name, obj))
            doc = parse(obj.__doc__)
            with open(plugins / f"{name}.mdx", "w") as f:
                _w = write(f)
                _w('import { Callout } from "nextra-theme-docs"')
                _w('import { Tab, Tabs } from "nextra-theme-docs"')
                _w()
                _w(f"# {name} (MLP)")
                _w()
                _w(doc.short_description)
                _w()
                _w("```python copy")
                _w(doc.examples[0].description)
                _w("```")
                _w()
                _w(doc.long_description)
                _w()

                if doc.params:
                    _w("## Parameters")
                    _w()
                    for p in doc.params:
                        if p.default:
                            _w(
                                f"- `{p.arg_name} = {p.default}`: {p.description}"
                            )
                        else:
                            _w(f"- `{p.arg_name}`: {p.description}")

                    _w()

                # if hasattr(obj, "forward"):
                #     _w("## Forward")
                #     _w("```rust")  # Hack for syntax highlighting
                #     _w(f"{signature(obj.forward)}")
                #     _w("```")
                #     _w()
                #     forward_doc = parse(obj.forward.__doc__)
                #     if forward_doc.returns:
                #         _w("### Returns")
                #         _w(forward_doc.returns.description)

    except Exception as e:
        pass


with open(plugins / f"_meta.json", "w") as f:
    _w = write(f)
    _w("{")
    for i, (name, obj) in enumerate(names):
        if i != len(names) - 1:
            _w(f'    "{name}": "{obj.__name__}",')
        else:
            _w(f'    "{name}": "{obj.__name__}"')
    _w("}")
