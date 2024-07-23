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


def standardize_identifier(identifier: str, pattern_key: str) -> str | None:
    """
    Standardize a DOI or arXiv identifier by removing any marker, lowercase, and applying a consistent separator
    """
    meta = {}
    regex = DOI_REGEX if pattern_key == "doi" else ARXIV_REGEX

    for matches in regex.finditer(identifier.casefold()):
        meta.update(matches.groupdict())

    if pattern_key == "doi":
        return (
            None
            if any(key not in meta for key in ["registrant", "suffix"])
            else f"10.{meta['registrant']}/{meta['suffix']}"
        )
    return None if "identifier" not in meta else meta["identifier"]


def extract_identifier(
    text: str,
) -> str | None:
    """Extract doi or arXiv identifier from a string"""
    for pattern_key, pattern_list in IDENTIFIER_PATTERNS.items():
        for pattern in pattern_list:
            if match := pattern.search(text.casefold()):
                if pattern_key == "arxiv" and (arxiv_meta := match.group(0)):
                    return standardize_identifier(arxiv_meta, pattern_key)
                elif doi_meta := match.group(1):
                    # For DOI, standardize and return it
                    return standardize_identifier(doi_meta, pattern_key)
    return None
