import inspect
import shutil
from io import TextIOWrapper
from pathlib import Path
from types import ModuleType
from typing import Any, List, Optional, Tuple

from docstring_parser import parse
from torch.nn.modules.module import _forward_unimplemented

import torchmix
from torchmix import Component
from torchmix.components.attention import AttentionPlugin
from torchmix.components.feedforward import FeedforwardPlugin


def prepare_path(path: str):
    path = Path(f"docs/pages/{path}")
    path.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


COMPONENTS = prepare_path("components")
EXAMPLES = prepare_path("examples")
PLUGINS = prepare_path("plugins")


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


class Write:
    def __init__(self, file: TextIOWrapper, obj: Any):
        self.file = file
        self.obj = obj
        self.doc = parse(obj.__doc__)

    def __call__(self, content: Optional[str] = None):
        self.file.write(str(content or "") + "\n")

    def name(self, suffix: Optional[str] = None):
        name = self.obj.__name__ + (suffix or "")
        self(f"# {name}")
        self()

    def short_description(self):
        if hasattr(self.doc, "short_description"):
            self(self.doc.short_description)
            self()

    def long_description(self):
        if hasattr(self.doc, "long_description"):
            self(self.doc.long_description)
            self()

    def example(self):
        try:
            description = self.doc.examples[0].description
        except:
            return

        self("```python copy")
        self(description)
        self("```")
        self()

    def params(self):
        if hasattr(self.doc, "params") and self.doc.params:
            self("## Parameters")
            self()
            for p in self.doc.params:
                if p.default:
                    self(f"- `{p.arg_name} = {p.default}`: {p.description}")
                else:
                    self(f"- `{p.arg_name}`: {p.description}")
            self()

    def forward(self):
        if (
            hasattr(self.obj, "forward")
            and getattr(self.obj, "forward") is not _forward_unimplemented
        ):
            self("## Forward")
            self("```rust")  # Hack for syntax highlighting
            self(f"{signature(self.obj.forward)}")
            self("```")
            self()
            forward_doc = parse(self.obj.forward.__doc__)
            if forward_doc.returns:
                self("### Returns")
                self(forward_doc.returns.description)


def write_meta(path: Path, names: List[Tuple[str, Any]]):
    with open(path / f"_meta.json", "w") as f:
        f.write("{\n")
        for i, (name, obj) in enumerate(names):
            if i != len(names) - 1:
                f.write(f'    "{name}": "{obj.__name__ if obj else name}",\n')
            else:
                f.write(f'    "{name}": "{obj.__name__ if obj else name}"\n')
        f.write("}\n")


def write_main(
    module: ModuleType,
    path: Path,
    base: Any,
    suffix: Optional[str] = None,
    meta: bool = True,
    names: Optional[List[Tuple[str, Any]]] = None,
) -> List[Tuple[str, Any]]:
    if not names:
        names = []

    for name in module.__all__:
        obj = getattr(module, name)
        if inspect.isclass(obj) and issubclass(obj, base) and not obj is base:
            with open(path / f"{name}.mdx", "w") as f:
                w = Write(f, obj)
                w('import { Callout } from "nextra-theme-docs"')
                w('import { Tab, Tabs } from "nextra-theme-docs"')
                w()
                w.name(suffix)
                w.short_description()
                w.example()
                w.long_description()
                w.params()
                w.forward()

            names.append((name, obj))

    if meta:
        write_meta(path, names)
        return

    return names


def convert(source: Path, path: Path):
    names = []
    for file in source.iterdir():
        if "__init__" not in str(file):
            with open(path / f"{file.stem}.mdx", "w") as f:

                def _w(txt: Optional[str] = None):
                    f.write((txt or "") + "\n")

                _w(f"# {file.stem}")
                _w()
                _w("```python copy")
                with file.open("r") as src:
                    f.write(src.read())
                _w("```")

            names.append((file.stem, None))

    write_meta(path, names)


write_main(torchmix.components, COMPONENTS, base=Component)

names = write_main(
    torchmix, PLUGINS, base=AttentionPlugin, suffix=" (Attention)", meta=False
)
write_main(
    torchmix,
    PLUGINS,
    base=FeedforwardPlugin,
    suffix=" (Feedforward)",
    names=names,
)

convert(Path("docs/examples"), EXAMPLES)
