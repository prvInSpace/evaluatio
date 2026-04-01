"""
Generate static Markdown API documentation from NumPy-style docstrings.

Usage:
    uv run generate_api_docs.py

Needs to be run from the evaluatio-docs folder since imports are relative
"""

import re
from pathlib import Path

import griffe

# Root of your Python source (the directory that contains the evaluatio package)
PACKAGE_DIR = Path("../python-src")

# Where to write the generated .md files
OUTPUT_DIR = Path("./API")

# The package name to recursively iterate through
PACKAGE_NAME = "evaluatio"

# Name overrides for API pages
NAME_OVERRIDES = {
    "wer": "WER",
    "cer": "CER",
    "ued": "UED",
    "uer": "UER",
    "pier": "PIER",
    "ci": "CI",
}


def format_annotation(annotation) -> str:
    """Convert a griffe annotation object to a readable string."""
    if annotation is None:
        return ""
    return str(annotation)


def format_section(section) -> str:
    """Render a single NumPy docstring section to Markdown."""
    kind = section.kind.value  # e.g. 'parameters', 'returns', 'raises', 'notes', etc.

    if kind in ["parameters", "attributes"]:
        lines = [f"**{kind.title()}**\n"]
        for param in section.value:
            annotation = format_annotation(param.annotation)
            type_str = f" : `{annotation}`" if annotation else ""
            lines.append(f"- **`{param.name}`**{type_str}  ")
            if param.description:
                # Indent continuation lines
                desc = param.description.strip().replace("\n", "\n  ")
                lines.append(f"  {desc}")
        return "\n".join(lines)

    elif kind == "returns":
        lines = ["**Returns**\n"]
        for ret in section.value:
            annotation = format_annotation(ret.annotation)
            name_str = f"**`{ret.name}`** : " if ret.name else ""
            type_str = f"`{annotation}`" if annotation else ""
            lines.append(f"- {name_str}{type_str}  ")
            if ret.description:
                desc = ret.description.strip().replace("\n", "\n  ")
                lines.append(f"  {desc}")
        return "\n".join(lines)

    elif kind == "raises":
        lines = ["**Raises**\n"]
        for exc in section.value:
            lines.append(f"- **`{exc.annotation.name}`**  ")
            if exc.description:
                desc = exc.description.strip().replace("\n", "\n  ")
                lines.append(f"  {desc}")
        return "\n".join(lines)

    elif kind == "examples":
        lines = ["**Examples**\n"]
        for entry_kind, entry_value in section.value:
            if entry_kind.value == "examples":
                lines.append(f"```python\n{entry_value.strip()}\n```")
            else:
                lines.append(entry_value.strip())
        return "\n".join(lines)

    elif kind in ("notes", "warnings", "references", "see also"):
        title = kind.title()
        raw = section.value
        if isinstance(raw, list):
            raw = "\n".join(str(r) for r in raw)
        return f"**{title}**\n\n{raw.strip()}"

    elif kind == "admonition":
        admonition = section.value
        title = admonition.kind.replace("_", " ").title()
        print(f"      Admonition type: {title}")
        body = admonition.description.strip() if admonition.description else ""
        body = re.sub(r"\[([0-9]+)\]_", lambda m: f"<sup>{m.group(1)}</sup>", body)
        body = re.sub(r"\.\. \[([0-9]+)\]", lambda m: f"{m.group(1)}.", body)
        return f"**{title}**\n\n{body}"

    else:
        # Fallback for any other section kinds
        raw = section.value
        if isinstance(raw, list):
            raw = "\n".join(str(r) for r in raw)
        return f"**{kind.title()}**\n\n{str(raw).strip()}"


def render_docstring(doc: griffe.Docstring | None) -> str:
    """Render a full docstring to Markdown."""
    if doc is None:
        return "*No documentation available.*"

    parsed = doc.parsed
    if not parsed:
        # Fall back to raw text if parsing yields nothing
        return doc.value.strip() if doc.value else "*No documentation available.*"

    parts = []
    for section in parsed:
        print(f"    Adding section {section.kind}")
        if section.kind.value == "text":
            # Plain description text
            text = section.value
            if isinstance(text, list):
                text = "\n".join(str(t) for t in text)
            parts.append(text.strip())
        else:
            rendered = format_section(section)
            if rendered:
                parts.append(rendered)

    return "\n\n".join(parts)


def render_function(func: griffe.Function, heading_level: int = 2) -> str:
    """Render a single function to Markdown."""
    print(f"  Compiling function {func.name}")
    hashes = "#" * heading_level

    # Build signature
    params = []
    for param in func.parameters:
        annotation = format_annotation(param.annotation)
        default = f"={param.default}" if param.default is not None else ""
        type_str = f": {annotation}" if annotation else ""
        params.append(f"{param.name}{type_str}{default}")

    return_annotation = format_annotation(func.returns)
    return_str = f" -> {return_annotation}" if return_annotation else ""
    signature = f"{func.name}({', '.join(params)}){return_str}"

    lines = [
        f"{hashes} {func.name}",
        "",
        "```python",
        signature,
        "```",
        "",
        render_docstring(func.docstring),
    ]
    return "\n".join(lines)


def render_class(cls: griffe.Class, heading_level: int = 2) -> str:
    """Render a class and its public methods to Markdown."""
    hashes = "#" * heading_level
    lines = [
        f"{hashes} `{cls.name}`",
        "",
        render_docstring(cls.docstring),
    ]

    # Render public methods (excluding dunder methods other than __init__)
    if not is_dataclass(cls):
        for name, member in cls.members.items():
            if not isinstance(member, griffe.Function):
                continue
            if name.startswith("_") and name != "__init__":
                continue
            lines.append("")
            lines.append(render_function(member, heading_level=heading_level + 1))

    return "\n".join(lines)


def is_dataclass(cls: griffe.Class) -> bool:
    return any("dataclass" in str(dec.value) for dec in cls.decorators)


def render_module(obj: griffe.Module) -> str | None:
    """Load and render a module to a full Markdown page."""

    module_path = obj.name

    # Page title from module docstring or module name
    title = module_path.split(".")[-1]
    title = NAME_OVERRIDES.get(title, title.replace("_", " ").title())

    lines = ["---", f"title: {title}", "---", f"# `{obj.path}`", ""]

    if obj.docstring:
        lines.append(render_docstring(obj.docstring))
        lines.append("")

    # Separate functions and classes
    functions = [
        (name, m)
        for name, m in obj.members.items()
        if isinstance(m, griffe.Function) and not name.startswith("_")
    ]
    classes = [
        (name, m)
        for name, m in obj.members.items()
        if isinstance(m, griffe.Class) and not name.startswith("_")
    ]

    # If there are no functions or classes just skip the the module
    if not functions and not classes:
        return None

    if functions:
        lines.append("# Functions")
        lines.append("")
        for _, func in functions:
            lines.append(render_function(func, heading_level=2))
            lines.append("")

    if classes:
        lines.append("# Classes")
        lines.append("")
        for _, cls in classes:
            lines.append(render_class(cls, heading_level=2))
            lines.append("")

    return "\n".join(lines)


def walk_modules(module: griffe.Module):
    """Walk module and all submodules recursively."""

    module_path = module.path
    output_path = module.path.removeprefix("evaluatio.").replace(".", "/")

    print(f"Generating {module_path} -> {output_path}.md")
    try:
        content = render_module(module)
        if content:
            out_file = OUTPUT_DIR / f"{output_path}.md"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(content, encoding="utf-8")
            print(f"  Written to {out_file}\n")
    except Exception as e:
        print(f"  ERROR: {e}")
    submodules = [m for m in module.modules.values() if "_bindings" not in m.name]
    for submod in submodules:
        walk_modules(submod)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    package = griffe.load(
        PACKAGE_NAME,
        search_paths=[PACKAGE_DIR],
        docstring_parser="numpy",
    )
    walk_modules(package)  # type: ignore
    print("\nDone.")


if __name__ == "__main__":
    main()
