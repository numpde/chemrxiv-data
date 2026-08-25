#!/usr/bin/env python3
"""Validate collection-owned HTML structure without interpreting scientific text."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from html import escape, unescape
from html.parser import HTMLParser
from itertools import zip_longest
from pathlib import Path


INLINE_TAGS = {"title", "p", "caption", "th", "td", "footer"}
VOID_TAGS = {"link", "meta"}
MARKER_BEFORE_NUMBER = re.compile(r"(?:•◊|•|◊)([ \t]*)(?=[+−-]?(?:\d|\.\d))")
SOURCE_PRESENTATION_TERM_IN_LABEL = re.compile(r"\b(?:table|fig(?:ure)?)s?\b", re.IGNORECASE)
HTML_SCRIPT_ELEMENT = re.compile(r"<\s*/?\s*(?:sup|sub)\b", re.IGNORECASE)
BASELINE_UNIT_EXPONENT = re.compile(
    r"(?<![A-Za-z\[])"
    r"(?:[kMGT]?Wm|[cmnµμ]?m|[kMµμ]?g|gcat|mol|[mµμ]?L|[mµμ]?s|K|Pa|Hz|V|A|W|J|M|e?Å)"
    r"[−-][1-9]\d*(?![A-Za-z0-9])"
)
BASELINE_POSITIVE_UNIT_EXPONENT = re.compile(
    r"(?<![A-Za-z])(?:[cmnµμ]?m|[mµμ]?s|Å)[23](?![A-Za-z0-9])"
)
CARET_EXPONENT = re.compile(r"\^[+−-]?\d+")
BASELINE_POWER_OF_TEN_EXPONENT = re.compile(
    r"(?:\b10−[1-9]\d*|(?:×|·|\*)[ \t]*10-[1-9]\d*)"
)
INCOMPLETE_SUPERSCRIPT_EXPONENT = re.compile(r"(?:[−-][⁰¹²³⁴⁵⁶⁷⁸⁹]+|⁻\d+)")
NUCLEAR_ISOTOPE = r"(?:1H|6Li|7Li|11B|13C|15N|17O|19F|27Al|31P)"
BASELINE_NUCLEAR_ISOTOPE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    rf"{NUCLEAR_ISOTOPE}"
    r"(?=(?:[^<\n]{0,48}\b(?:NMR|PFG|HSQC|HMBC|HETCOR)\b|-labelled\b))"
    r"|7Li(?=[ \t]+longitudinal relaxation\b)"
    r")|(?<=[–-])15N(?![A-Za-z0-9])"
)
BASELINE_COUPLING_ORDER = re.compile(
    r"(?<![A-Za-z0-9])[1-9]J(?=[A-Za-z,]*[ \t]*(?:=|~))"
)
BASELINE_DEUTERATED_SOLVENT = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"(?:acetone|DMSO|MeOD|THF|tetrahydrofuran|toluene|tol|MeOH|"
    r"dichloromethane|tetrachloroethane|OMePh)-d[1-9]\d*"
    r"|d[1-9]\d*-(?:DMSO|THF|toluene|MeOH)"
    r"|CDCl3|C6D6|CD3CN|CD3OD|D2O|C2D2Cl4"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)
PAGE_ITEM = r"S?\d+(?:–S?\d+)?"


@dataclass(frozen=True)
class Problem:
    line: int
    message: str


@dataclass
class Node:
    tag: str
    attrs: list[tuple[str, str | None]]
    line: int
    children: list[Node | str] = field(default_factory=list)


class DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("#document", [], 1)
        self.stack = [self.root]
        self.problems: list[Problem] = []

    def problem(self, message: str) -> None:
        self.problems.append(Problem(self.getpos()[0], message))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag, attrs, self.getpos()[0])
        self.stack[-1].children.append(node)
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        if len(self.stack) == 1:
            self.problem(f"closing </{tag}> has no open element")
            return
        if self.stack[-1].tag != tag:
            self.problem(f"closing </{tag}> does not match open <{self.stack[-1].tag}>")
            return
        self.stack.pop()

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)

    def close(self) -> None:
        super().close()
        for node in self.stack[1:]:
            self.problems.append(Problem(node.line, f"element <{node.tag}> is not closed"))


def elements(node: Node) -> list[Node]:
    return [child for child in node.children if isinstance(child, Node)]


def has_significant_text(node: Node) -> bool:
    return any(isinstance(child, str) and child.strip() for child in node.children)


def text_content(node: Node) -> str:
    return "".join(text_content(child) if isinstance(child, Node) else child for child in node.children)


class SemanticValidator:
    def __init__(self) -> None:
        self.problems: list[Problem] = []

    def problem(self, node: Node, message: str) -> None:
        self.problems.append(Problem(node.line, message))

    def expect_children(self, node: Node, tags: list[str]) -> list[Node] | None:
        actual = elements(node)
        actual_tags = [child.tag for child in actual]
        if actual_tags != tags:
            expected = ", ".join(f"<{tag}>" for tag in tags)
            self.problem(node, f"<{node.tag}> must contain {expected} in that order")
            return None
        if has_significant_text(node):
            self.problem(node, f"<{node.tag}> cannot contain text directly")
        return actual

    def expect_attrs(self, node: Node, attrs: list[tuple[str, str | None]]) -> None:
        if node.attrs != attrs:
            expected = " ".join(f'{key}="{value}"' for key, value in attrs)
            suffix = f" {expected}" if expected else " no attributes"
            self.problem(node, f"<{node.tag}> must have{suffix}")

    def expect_plain_text(self, node: Node) -> None:
        if elements(node):
            self.problem(node, f"<{node.tag}> must contain text only")
        if not text_content(node).strip():
            self.problem(node, f"<{node.tag}> cannot be empty")

    def expect_paper_title_sentence_case(self, cite: Node) -> None:
        title = text_content(cite)
        if not title.strip():
            return
        title_word_initials = [
            next((character for character in word if character.isalpha()), None)
            for word in title.split()
        ]
        cased_word_initials = [initial for initial in title_word_initials if initial is not None]
        subsequent_word_initials = cased_word_initials[1:]
        lowercase_words = sum(initial.islower() for initial in subsequent_word_initials)
        if subsequent_word_initials and lowercase_words * 2 <= len(subsequent_word_initials):
            self.problem(cite, "paper title must use sentence case; most words after the first must start lowercase")

    def validate(self, root: Node) -> list[Problem]:
        document_children = elements(root)
        if len(document_children) != 1 or document_children[0].tag != "html":
            self.problem(root, "document must contain exactly one <html> element")
            return self.problems
        if has_significant_text(root):
            self.problem(root, "document cannot contain text outside <html>")

        html = document_children[0]
        self.expect_attrs(html, [("lang", "en")])
        html_children = self.expect_children(html, ["head", "body"])
        if html_children is None:
            return self.problems
        document_year = self.validate_head(html_children[0])
        self.validate_body(html_children[1], document_year)
        return self.problems

    def validate_head(self, head: Node) -> str | None:
        self.expect_attrs(head, [])
        children = self.expect_children(head, ["meta", "meta", "title", "link"])
        if children is None:
            return None
        charset, viewport, title, stylesheet = children
        self.expect_attrs(charset, [("charset", "utf-8")])
        self.expect_attrs(
            viewport,
            [("name", "viewport"), ("content", "width=device-width, initial-scale=1")],
        )
        self.expect_attrs(title, [])
        self.expect_attrs(stylesheet, [("rel", "stylesheet"), ("href", "../../style.css")])
        self.expect_plain_text(title)
        match = re.fullmatch(r"Compound characterization data — .+, (\d{4})", text_content(title))
        if not match:
            self.problem(title, "<title> must use 'Compound characterization data — credit, year'")
            return None
        return match.group(1)

    def validate_body(self, body: Node, document_year: str | None) -> None:
        self.expect_attrs(body, [])
        children = self.expect_children(body, ["main"])
        if children is None:
            return
        main = children[0]
        self.expect_attrs(main, [])
        main_children = elements(main)
        if has_significant_text(main):
            self.problem(main, "<main> cannot contain text directly")
        if not main_children:
            self.problem(main, "<main> must contain one <header> followed only by <article> elements")
            return
        header, *articles = main_children
        if header.tag != "header" or any(article.tag != "article" for article in articles):
            self.problem(main, "<main> must contain one <header> followed only by <article> elements")
            return
        self.validate_header(header, note_required=not articles, document_year=document_year)
        for article in articles:
            self.validate_article(article)

    def validate_header(self, header: Node, note_required: bool, document_year: str | None) -> None:
        self.expect_attrs(header, [])
        children = elements(header)
        invalid_children = len(children) not in (1, 2) or any(child.tag != "p" for child in children)
        if has_significant_text(header) or invalid_children:
            self.problem(header, "<header> must contain a bibliographic <p> and at most one note <p>")
            return
        self.validate_bibliography(children[0], document_year)
        for note in children[1:]:
            self.expect_attrs(note, [])
            self.expect_plain_text(note)
        if note_required and len(children) == 1:
            self.problem(header, "a document without <article> elements must include a nonempty header note")

    def validate_bibliography(self, paragraph: Node, document_year: str | None) -> None:
        self.expect_attrs(paragraph, [])
        children = paragraph.children
        node_children = [child for child in children if isinstance(child, Node)]
        if [child.tag for child in node_children] != ["cite", "time", "a"] or len(children) != 6:
            self.problem(
                paragraph,
                "bibliographic <p> must contain title <cite>, year <time>, and DOI <a> in the standard order",
            )
            return
        before, cite, after_cite, time, after_time, anchor = children
        if not all(isinstance(part, str) for part in (before, after_cite, after_time)):
            self.problem(paragraph, "bibliographic punctuation does not match the standard form")
            return
        assert isinstance(cite, Node) and isinstance(time, Node) and isinstance(anchor, Node)
        author_text = before.removesuffix(". “").strip()
        punctuation_is_valid = (
            bool(author_text)
            and before.endswith(". “")
            and after_cite == ".” ChemRxiv ("
            and after_time == "). "
        )
        if not punctuation_is_valid:
            self.problem(paragraph, "bibliographic punctuation does not match the standard form")
        for node in (cite, time, anchor):
            self.expect_plain_text(node)
        self.expect_attrs(cite, [])
        self.expect_paper_title_sentence_case(cite)
        year = text_content(time)
        self.expect_attrs(time, [("datetime", year)])
        if not re.fullmatch(r"\d{4}", year):
            self.problem(time, "bibliographic year must contain four digits")
        elif document_year is not None and year != document_year:
            self.problem(time, "bibliographic year must match the document title year")
        if len(anchor.attrs) != 1 or anchor.attrs[0][0] != "href" or anchor.attrs[0][1] is None:
            self.problem(anchor, "DOI <a> must have exactly one href attribute")
        else:
            href = anchor.attrs[0][1]
            assert href is not None
            if not href.startswith("https://doi.org/"):
                self.problem(anchor, "DOI href must be an absolute https://doi.org/ URL")
            if text_content(anchor) != href:
                self.problem(anchor, "DOI link text must equal its href")

    def validate_article(self, article: Node) -> None:
        self.expect_attrs(article, [])
        children = self.expect_children(article, ["table", "footer"])
        if children is None:
            return
        table, footer = children
        self.validate_table(table)
        self.validate_footer(footer)

    def validate_table(self, table: Node) -> None:
        self.expect_attrs(table, [])
        children = self.expect_children(table, ["caption", "tbody"])
        if children is None:
            return
        caption, tbody = children
        self.expect_attrs(caption, [])
        self.expect_plain_text(caption)
        self.expect_no_terminal_punctuation(caption, "caption")
        self.expect_attrs(tbody, [])
        rows = elements(tbody)
        if has_significant_text(tbody) or not rows or any(row.tag != "tr" for row in rows):
            self.problem(tbody, "<tbody> must contain one or more <tr> elements")
            return
        for row in rows:
            self.expect_attrs(row, [])
            cells = self.expect_children(row, ["th", "td"])
            if cells is None:
                continue
            label, value = cells
            self.expect_attrs(label, [("scope", "row")])
            self.expect_attrs(value, [])
            self.expect_plain_text(label)
            self.expect_plain_text(value)
            label_text = text_content(label).strip()
            if label_text == "Product":
                self.problem(label, "use Identifier for a compound identifier, not Product")
            if SOURCE_PRESENTATION_TERM_IN_LABEL.search(label_text):
                self.problem(label, "measurement labels must describe the measurement, not a source table or figure")
            self.expect_no_terminal_punctuation(label, "measurement label")
            self.expect_no_terminal_punctuation(value, "measurement text")

    def expect_no_terminal_punctuation(self, node: Node, field_name: str) -> None:
        if text_content(node).rstrip().endswith((".", ";", ",", ":")):
            self.problem(node, f"{field_name} must not end with punctuation")

    def validate_footer(self, footer: Node) -> None:
        self.expect_attrs(footer, [])
        references = self.footer_references(footer)
        if references is None:
            return

        labels = [text_content(source) for source, _locator in references]
        if labels not in (["Main"], ["SI"], ["Main", "SI"]):
            self.problem(footer, "footer sources must be Main and/or SI, in that order")
        for source, locator in references:
            if source.tag != "cite":
                self.problem(source, "footer sources must use <cite>")
                continue
            self.expect_attrs(source, [])
            self.expect_plain_text(source)
            self.validate_page_locator(footer, locator)

    def footer_references(self, footer: Node) -> list[tuple[Node, str]] | None:
        parts = footer.children
        if len(parts) == 2 and isinstance(parts[0], Node) and isinstance(parts[1], str):
            return [(parts[0], parts[1])]
        if (
            len(parts) == 4
            and isinstance(parts[0], Node)
            and isinstance(parts[1], str)
            and isinstance(parts[2], Node)
            and isinstance(parts[3], str)
        ):
            if not parts[1].endswith("; "):
                self.problem(footer, "separate Main and SI citations with a semicolon and one space")
                return None
            return [(parts[0], parts[1][:-2]), (parts[2], parts[3])]
        self.problem(footer, "<footer> must contain one or two source and page-locator pairs")
        return None

    def validate_page_locator(self, footer: Node, locator: Node | str) -> None:
        if not isinstance(locator, str):
            self.problem(footer, "each footer source must be followed by a page locator")
            return
        match = re.fullmatch(rf", (p|pp)\. ({PAGE_ITEM}(?:, {PAGE_ITEM})*)", locator)
        if match is None:
            self.problem(footer, "each footer source must be followed by a page locator")
            return
        page_kind, pages = match.groups()
        has_multiple_pages = ", " in pages or "–" in pages
        if page_kind == "p" and has_multiple_pages:
            self.problem(footer, "use pp. for a page range or list")
        elif page_kind == "pp" and not has_multiple_pages:
            self.problem(footer, "use p. for one page")


def validate_markers(line_number: int, line: str) -> list[Problem]:
    problems: list[Problem] = []
    rendered_line = unescape(line)
    if "●" in rendered_line:
        problems.append(Problem(line_number, "use the smaller marker •, not ●"))
    marker_runs = re.findall(r"[•◊](?:[ \t]*[•◊])+", rendered_line)
    if any(run != "•◊" for run in marker_runs):
        problems.append(Problem(line_number, "the only valid combined marker is •◊"))
    if any(match.group(1) != " " for match in MARKER_BEFORE_NUMBER.finditer(rendered_line)):
        problems.append(Problem(line_number, "a marker before a number must be followed by one ASCII space"))
    return problems


def find_invalid_notation(text: str, patterns: tuple[re.Pattern[str], ...]) -> list[str]:
    notations: list[str] = []
    for match in sorted(
        (match for pattern in patterns for match in pattern.finditer(text)),
        key=lambda match: match.start(),
    ):
        notation = match.group()
        if len(notation) > 32:
            notation = notation[:29] + "..."
        if notation not in notations:
            notations.append(notation)
    return notations


def describe_invalid_notation(notations: list[str]) -> str:
    shown = ", ".join(repr(notation) for notation in notations[:3])
    omitted = len(notations) - 3
    return shown + (f" and {omitted} more" if omitted > 0 else "")


def validate_typographic_scripts(line_number: int, line: str) -> list[Problem]:
    problems: list[Problem] = []
    rendered_line = unescape(line)
    if HTML_SCRIPT_ELEMENT.search(rendered_line):
        problems.append(
            Problem(line_number, "use Unicode superscript or subscript characters, not HTML elements")
        )
    invalid_exponents = find_invalid_notation(
        rendered_line,
        (
            BASELINE_UNIT_EXPONENT,
            BASELINE_POSITIVE_UNIT_EXPONENT,
            CARET_EXPONENT,
            BASELINE_POWER_OF_TEN_EXPONENT,
            INCOMPLETE_SUPERSCRIPT_EXPONENT,
        ),
    )
    if invalid_exponents:
        problems.append(
            Problem(
                line_number,
                "replace baseline exponent notation "
                f"{describe_invalid_notation(invalid_exponents)} with Unicode superscript "
                "characters, for example cm⁻¹, m², or 10⁵",
            )
        )
    invalid_prefixes = find_invalid_notation(
        rendered_line,
        (BASELINE_NUCLEAR_ISOTOPE, BASELINE_COUPLING_ORDER),
    )
    if invalid_prefixes:
        problems.append(
            Problem(
                line_number,
                "replace baseline nuclear or coupling notation "
                f"{describe_invalid_notation(invalid_prefixes)} with Unicode superscript "
                "characters, for example ¹H NMR or ³J = 7.2 Hz",
            )
        )
    invalid_subscripts = find_invalid_notation(
        rendered_line,
        (BASELINE_DEUTERATED_SOLVENT,),
    )
    if invalid_subscripts:
        problems.append(
            Problem(
                line_number,
                "replace baseline deuterated-solvent indices "
                f"{describe_invalid_notation(invalid_subscripts)} with Unicode subscript "
                "characters, for example CDCl₃, C₆D₆, or DMSO-d₆",
            )
        )
    return problems


def opening_tag(node: Node) -> str:
    attributes = "".join(
        f' {name}="{escape(value, quote=True)}"'
        for name, value in node.attrs
        if value is not None
    )
    return f"<{node.tag}{attributes}>"


def render_inline(node: Node) -> str:
    content = "".join(
        render_inline(child) if isinstance(child, Node) else escape(child, quote=False)
        for child in node.children
    )
    return f"{opening_tag(node)}{content}</{node.tag}>"


def separate_children(parent: Node, previous: Node, current: Node) -> bool:
    return (
        parent.tag == "main" and current.tag == "article"
        or parent.tag == "tbody" and previous.tag == current.tag == "tr"
        or parent.tag == "article" and previous.tag == "table" and current.tag == "footer"
    )


def render_element(node: Node, indentation: int = 0) -> list[str]:
    prefix = " " * indentation
    if node.tag in VOID_TAGS:
        return [prefix + opening_tag(node)]
    if node.tag in INLINE_TAGS:
        return [prefix + render_inline(node)]

    lines = [prefix + opening_tag(node)]
    child_indentation = indentation if node.tag == "html" else indentation + 2
    children = elements(node)
    for index, child in enumerate(children):
        if index and separate_children(node, children[index - 1], child):
            lines.append("")
        lines.extend(render_element(child, child_indentation))
    lines.append(prefix + f"</{node.tag}>")
    return lines


def render_document(root: Node) -> str:
    html = elements(root)[0]
    return "<!doctype html>\n" + "\n".join(render_element(html)) + "\n"


def validate_canonical_source(source: str, root: Node) -> list[Problem]:
    canonical = render_document(root)
    for line_number, (actual, expected) in enumerate(
        zip_longest(source.split("\n"), canonical.split("\n")),
        1,
    ):
        if actual == expected:
            continue
        if expected is None:
            message = "unexpected line after the canonical document"
        elif actual is None:
            message = "canonical document line is missing"
        else:
            message = "line does not match canonical HTML formatting"
        return [Problem(line_number, message)]
    return []


def validate_text(source: str) -> list[Problem]:
    parser = DocumentParser()
    parser.feed(source)
    parser.close()

    problems = list(parser.problems)
    semantic_problems: list[Problem] = []
    if not parser.problems:
        semantic_problems = SemanticValidator().validate(parser.root)
        problems.extend(semantic_problems)
    if not parser.problems and not semantic_problems:
        problems.extend(validate_canonical_source(source, parser.root))

    for line_number, line in enumerate(source.splitlines(), 1):
        problems.extend(validate_markers(line_number, line))
        problems.extend(validate_typographic_scripts(line_number, line))
    return sorted(problems, key=lambda problem: (problem.line, problem.message))


def collect_html_files(paths: list[Path]) -> tuple[list[Path], list[str]]:
    files: set[Path] = set()
    errors: list[str] = []
    for path in paths:
        if path.is_file():
            if path.suffix.lower() != ".html":
                errors.append(f"{path}:1: expected an HTML file")
            else:
                files.add(path)
        elif path.is_dir():
            found = {
                candidate
                for candidate in path.rglob("*.html")
                if candidate.name != "index.html"
            }
            if not found:
                errors.append(f"{path}:1: no HTML files found")
            files.update(found)
        else:
            errors.append(f"{path}:1: path was not found")
    return sorted(files), errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("ext")],
        help="HTML files or directories to validate (default: ext)",
    )
    args = parser.parse_args()
    files, path_errors = collect_html_files(args.paths)
    for error in path_errors:
        print(error, file=sys.stderr)

    failed = bool(path_errors)
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            print(f"{path}:1: {error}", file=sys.stderr)
            failed = True
            continue
        problems = validate_text(source)
        failed = failed or bool(problems)
        for problem in problems:
            print(f"{path}:{problem.line}: {problem.message}", file=sys.stderr)

    if failed:
        return 1
    print(f"Validated {len(files)} HTML file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
