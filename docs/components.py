#

import inspect
import shutil
from io import TextIOWrapper
from pathlib import Path
from typing import Optional

from docstring_parser import parse

import torchmix
from torchmix import MixModule

# root = Path("docs/pages")
# root.mkdir(parents=True, exist_ok=True)

components = Path("docs/pages/components")
shutil.rmtree(components)
components.mkdir(parents=True, exist_ok=True)


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


for name, obj in inspect.getmembers(torchmix.components):
    try:
        if issubclass(obj, MixModule):
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
                _w("## Parameters")
                _w()
                for p in doc.params:
                    _w(f"- `{p.arg_name} = {p.default}`: {p.description}")
                _w()
                _w("## Signatures")
                _w("```ts")
                _w(f"{inspect.signature(obj.forward)}")
                _w()

    except:
        pass
