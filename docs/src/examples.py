import inspect
import shutil
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Optional

from docstring_parser import parse

import torchmix
from torchmix import Component

# root = Path("docs/pages")
# root.mkdir(parents=True, exist_ok=True)

examples_src = Path("docs/examples")
examples = Path("docs/pages/examples")
shutil.rmtree(examples)
examples.mkdir(parents=True, exist_ok=True)


def write(file: TextIOWrapper):
    def _write(content: Optional[str] = None):
        try:
            file.write(content + "\n")
        except:
            file.write("\n")

    return _write


names = []
for path in examples_src.iterdir():
    if "__init__" not in str(path):
        with open(examples / f"{path.stem}.mdx", "w") as f:
            _w = write(f)
            _w(f"# {path.stem}")
            _w()
            _w("```python copy")
            with path.open("r") as src:
                f.write(src.read())
            _w("```")
        names.append(path.stem)


# names = []
# for name, obj in inspect.getmembers(torchmix.components):
#     try:
#         if issubclass(obj, Component):
#             doc = parse(obj.__doc__)
#             with open(components / f"{name}.mdx", "w") as f:
#                 _w = write(f)
#                 _w('import { Callout } from "nextra-theme-docs"')
#                 _w('import { Tab, Tabs } from "nextra-theme-docs"')
#                 _w()
#                 _w(f"# {name}")
#                 _w()
#                 _w(doc.short_description)
#                 _w()
#                 _w("```python copy")
#                 _w(doc.examples[0].description)
#                 _w("```")
#                 _w()
#                 _w(doc.long_description)
#                 _w()

#                 if doc.params:
#                     _w("## Parameters")
#                     _w()
#                     for p in doc.params:
#                         if p.default:
#                             _w(
#                                 f"- `{p.arg_name} = {p.default}`: {p.description}"
#                             )
#                         else:
#                             _w(f"- `{p.arg_name}`: {p.description}")

#                     _w()

#                 if hasattr(obj, "forward"):
#                     _w("## Forward")
#                     _w("```rust")  # Hack for syntax highlighting
#                     _w(f"{signature(obj.forward)}")
#                     _w("```")
#                     _w()
#                     forward_doc = parse(obj.forward.__doc__)
#                     if forward_doc.returns:
#                         _w("### Returns")
#                         _w(forward_doc.returns.description)

#             names.append((name, obj))

#     except Exception as e:
#         pass

with open(examples / f"_meta.json", "w") as f:
    _w = write(f)
    _w("{")
    for i, name in enumerate(names):
        if i != len(names) - 1:
            _w(f'    "{name}": "{name}",')
        else:
            _w(f'    "{name}": "{name}"')
    _w("}")
