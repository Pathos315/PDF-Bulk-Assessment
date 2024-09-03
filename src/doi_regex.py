from __future__ import annotations
import re


DOI_PATTERNS = [
    re.compile(r"doi[\s\.\:]{0,2}(10\.\d{4}[\d\:\.\-\/a-z]+)(?:[\s\n\"<]|$)"),
    re.compile(r"(10\.\d{4}[\d\:\.\-\/a-z]+)(?:[\s\n\"<]|$)"),
    re.compile(r"(10\.\d{4}[\:\.\-\/a-z]+[\:\.\-\d]+)(?:[\s\na-z\"<]|$)"),
    re.compile(
        r"https?://[ -~]*doi[ -~]*/(10\.\d{4,9}/[-._;()/:a-z0-9]+)(?:[\s\n\"<]|$)"
    ),
    re.compile(r"^(10\.\d{4,9}/[-._;()/:a-z0-9]+)$"),
]
ARXIV_PATTERNS = [
    re.compile(r"^(\d{4}\.\d+)(?:v\d+)?$"),
    re.compile(r"arxiv[\s]*\:[\s]*(\d{4}\.\d+)(?:v\d+)?(?:[\s\n\"<]|$)"),
    re.compile(r"(\d{4}\.\d+)(?:v\d+)?(?:\.pdf)"),
    re.compile(r"^(\d{4}\.\d+)(?:v\d+)?$"),
]

DOI_REGEX = re.compile(
    r"""(?xm)
  (?P<marker>   doi[:\/\s]{0,3})?
  (?P<prefix>
    (?P<namespace> 10)
    [.]
    (?P<registrant> \d{2,9})
  )
  (?P<sep>     [:\-\/\s\]])
  (?P<suffix>  [\-._;()\/:a-z0-9]+[a-z0-9])
  (?P<trailing> ([\s\n\"<.]|$))
"""
)

ARXIV_REGEX = re.compile(
    r"""(?x)
    (?P<marker>arxiv[:\/\s]{0,3})?  # Marker (optional)
    (?P<identifier>\d{4}\.\d+)       # Identifier (mandatory)
    (?:v\d+)?                        # Version (optional)
    (?P<trailing>\.pdf)?$            # Trailing '.pdf' (optional)
""",
    flags=re.IGNORECASE,  # Using IGNORECASE flag to make the regex case-insensitive
)

IDENTIFIER_PATTERNS = {
    "doi": DOI_PATTERNS,
    "arxiv": ARXIV_PATTERNS,
}


def standardize_identifier(identifier: str, pattern_key: str) -> str:
    """
    Standardize a DOI or arXiv identifier by removing any marker, lowercase, and applying a consistent separator
    """
    regex = DOI_REGEX if pattern_key == "doi" else ARXIV_REGEX
    meta: dict[str, str] = next(regex.finditer(identifier.casefold()), {}).groupdict()  # type: ignore

    if pattern_key == "doi":
        return format_doi(meta)
    return (
        f"{meta['identifier']}{meta.get('trailing', '.pdf')}"
        if {"identifier"}.issubset(meta)
        else ""
    )


def format_doi(meta: dict[str, str]) -> str:
    """Format a DOI from its components."""
    return (
        f"10.{meta['registrant']}/{meta['suffix']}"
        if {"registrant", "suffix"}.issubset(meta)
        else ""
    )


def extract_identifier(text: str) -> str:
    """
    Extract DOI or arXiv identifier from a string.

    Args:
        text (str): The text to search for an identifier.

    Returns:
        Optional[str]: The standardized identifier if found, otherwise None.
    """
    text_lower = text.casefold()

    for pattern_key, patterns in IDENTIFIER_PATTERNS.items():
        if not (
            identifier := find_identifier(text_lower, patterns, pattern_key)
        ):
            continue
        return identifier
    return ""


def find_identifier(
    text: str, patterns: list[re.Pattern[str]], pattern_key: str
) -> str:
    """
    Find and standardize an identifier using a list of patterns.

    Args:
        text (str): The text to search in.
        patterns (List[Pattern[str]]): List of regex patterns to try.
        pattern_key (str): The type of identifier ('doi' or 'arxiv').

    Returns:
        Optional[str]: The standardized identifier if found, otherwise None.
    """

    group_index = 0 if pattern_key == "arxiv" else 1
    for pattern in patterns:
        if not (this_match := pattern.search(text)):
            continue
        if not (meta := this_match.group(group_index)):
            continue
        return standardize_identifier(meta, pattern_key)
    return ""
