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

components = Path("docs/pages/components")
shutil.rmtree(components)
components.mkdir(parents=True, exist_ok=True)


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
for name, obj in inspect.getmembers(torchmix.components):
    try:
        if issubclass(obj, Component):
            doc = parse(obj.__doc__)
            with open(components / f"{name}.mdx", "w") as f:
                _w = write(f)
                _w('import { Callout } from "nextra-theme-docs"')
                _w('import { Tab, Tabs } from "nextra-theme-docs"')
                _w()
                _w(f"# {name}")
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

                if hasattr(obj, "forward"):
                    _w("## Forward")
                    _w("```rust")  # Hack for syntax highlighting
                    _w(f"{signature(obj.forward)}")
                    _w("```")
                    _w()
                    forward_doc = parse(obj.forward.__doc__)
                    if forward_doc.returns:
                        _w("### Returns")
                        _w(forward_doc.returns.description)

            names.append((name, obj))

    except Exception as e:
        pass


for name, obj in inspect.getmembers(torchmix.components.attention):
    try:
        if issubclass(obj, Component) and not issubclass(obj, AttentionPlugin):
            doc = parse(obj.__doc__)
            with open(components / f"{name}.mdx", "w") as f:
                _w = write(f)
                _w('import { Callout } from "nextra-theme-docs"')
                _w('import { Tab, Tabs } from "nextra-theme-docs"')
                _w()
                _w(f"# {name}")
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

                if hasattr(obj, "forward"):
                    _w("## Forward")
                    _w("```rust")  # Hack for syntax highlighting
                    _w(f"{signature(obj.forward)}")
                    _w("```")
                    _w()
                    forward_doc = parse(obj.forward.__doc__)
                    if forward_doc.returns:
                        _w("### Returns")
                        _w(forward_doc.returns.description)

            names.append((name, obj))

    except Exception as e:
        pass

for name, obj in inspect.getmembers(torchmix.components.mlp):
    try:
        if issubclass(obj, Component) and not issubclass(obj, MLPPlugin):
            doc = parse(obj.__doc__)
            with open(components / f"{name}.mdx", "w") as f:
                _w = write(f)
                _w('import { Callout } from "nextra-theme-docs"')
                _w('import { Tab, Tabs } from "nextra-theme-docs"')
                _w()
                _w(f"# {name}")
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

                if hasattr(obj, "forward"):
                    _w("## Forward")
                    _w("```rust")  # Hack for syntax highlighting
                    _w(f"{signature(obj.forward)}")
                    _w("```")
                    _w()
                    forward_doc = parse(obj.forward.__doc__)
                    if forward_doc.returns:
                        _w("### Returns")
                        _w(forward_doc.returns.description)

            names.append((name, obj))

    except Exception as e:
        pass


with open(components / f"_meta.json", "w") as f:
    _w = write(f)
    _w("{")
    for i, (name, obj) in enumerate(names):
        if i != len(names) - 1:
            _w(f'    "{name}": "{obj.__name__}",')
        else:
            _w(f'    "{name}": "{obj.__name__}"')
    _w("}")
