from importlib.metadata import entry_points
from pathlib import Path
import re

from pybtex.plugin import find_plugin
from pybtex.database import parse_file
from pybtex.backends.markdown import Backend as MarkdownBackend

APA = find_plugin("pybtex.style.formatting", "apa")()


class CleanMarkdownBackend(MarkdownBackend):
    """This whole class exist soly due to the fact that
    the markdown backend is too eager to escape characters
    which causes it to go into mathmode sometimes."""

    def format_str(self, text):
        text = text.replace("\\", "\\\\")
        for ch in ("*", "_", "[", "]", "`"):
            text = text.replace(ch, "\\" + ch)
        return text


MD = CleanMarkdownBackend()


def bib_to_apa6_markdown(bibfile):
    bibliography = parse_file(bibfile, "bibtex")
    formatted_bib = APA.format_bibliography(bibliography)
    lines = ["# Bibliography\n"]
    for entry in formatted_bib:
        lines.append(f"## {entry.label}")
        text = entry.text.render(MD)
        # Fixes a bug where a space is added after the year if
        # a month is also added
        text = re.sub(r"(\d{4})\s+,\s+", r"\1, ", text)
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


docs_folder = Path(__file__).parent
bibliography = docs_folder / "bibliography.bib"
bib_output = docs_folder / "bibliography.md"
bib_output.write_text(bib_to_apa6_markdown(bibliography))
