"""
dedup_utils.py — Automated entity deduplication for node_data.

Merges entity entries that refer to the same real-world entity using:
  1. Prefix / substring containment (one name is entirely inside the other)
  2. Fuzzy string similarity via rapidfuzz (fallback: thefuzz)

Only entities that share the same Node_type are ever merged.
When merging, the longer (more complete) name wins as the canonical key.
"""

# ---------------------------------------------------------------------------
# Fuzzy-match library — prefer rapidfuzz, fall back to thefuzz
# ---------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz as _fuzz
    _ratio = _fuzz.token_sort_ratio   # handles word-order differences
except ImportError:
    try:
        from thefuzz import fuzz as _fuzz
        _ratio = _fuzz.token_sort_ratio
    except ImportError:
        raise ImportError(
            "Install rapidfuzz (preferred) or thefuzz to use dedup_utils.\n"
            "  pip install rapidfuzz"
        )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FUZZY_THRESHOLD = 85   # similarity score (0-100) to consider two names the same
                        # Conservative: avoids merging genuinely different characters


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_substring_match(short: str, long: str) -> bool:
    """
    Return True when `short` appears as a whole-word substring inside `long`.
    E.g. "sam" in "sam gamgee", but NOT "am" in "sam gamgee".
    """
    import re
    pattern = r'\b' + re.escape(short) + r'\b'
    return bool(re.search(pattern, long))


def _merge_into(canonical: dict, duplicate: dict) -> dict:
    """
    Merge `duplicate` attributes into `canonical` and return the result.

    Rules:
      - Reference_Count: sum
      - First_Apperance: keep the one that sorts earliest (lexicographic on
        "Book X, Chapter Y" strings is correct because both fields use the
        same format)
      - Last_Apperance: keep the one that sorts latest
      - Confidence: keep the higher value
      - Everything else (Node_type, ID, Description, Label): keep canonical's
    """
    merged = canonical.copy()

    # Reference_Count
    canon_rc = canonical.get("Reference_Count", 1)
    dup_rc    = duplicate.get("Reference_Count", 1)
    merged["Reference_Count"] = canon_rc + dup_rc

    # First_Apperance — earliest wins
    canon_first = canonical.get("First_Apperance", "[N/A]")
    dup_first   = duplicate.get("First_Apperance", "[N/A]")
    if canon_first == "[N/A]":
        merged["First_Apperance"] = dup_first
    elif dup_first != "[N/A]":
        merged["First_Apperance"] = min(canon_first, dup_first)

    # Last_Apperance — latest wins
    canon_last = canonical.get("Last_Apperance", "[N/A]")
    dup_last   = duplicate.get("Last_Apperance", "[N/A]")
    if canon_last == "[N/A]":
        merged["Last_Apperance"] = dup_last
    elif dup_last != "[N/A]":
        merged["Last_Apperance"] = max(canon_last, dup_last)

    # Confidence — keep the highest observed
    canon_conf = canonical.get("Confidence", 0.0)
    dup_conf   = duplicate.get("Confidence", 0.0)
    merged["Confidence"] = max(canon_conf, dup_conf)

    return merged


def _pick_canonical(key_a: str, key_b: str):
    """
    Between two entity keys, return (canonical_key, duplicate_key).
    The longer name is more complete and wins as the canonical form.
    Tie-break: alphabetical order for determinism.
    """
    if len(key_a) > len(key_b):
        return key_a, key_b
    elif len(key_b) > len(key_a):
        return key_b, key_a
    else:
        return (key_a, key_b) if key_a <= key_b else (key_b, key_a)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def deduplicate_entities(node_data: dict) -> dict:
    """
    Deduplicate node_data by merging entries that refer to the same entity.

    Parameters
    ----------
    node_data : dict
        Keyed by canonical lowercase entity name.
        Each value is an attribute dict with at least:
          Node_type, ID, Label, Reference_Count,
          First_Apperance, Last_Apperance, Confidence

    Returns
    -------
    dict
        A cleaned copy of node_data with duplicates merged.
    """
    # Work on a shallow copy of keys; we'll build a fresh output dict
    keys = list(node_data.keys())

    # Union-Find structure: maps every key to its canonical representative
    parent: dict[str, str] = {k: k for k in keys}

    def find(k: str) -> str:
        """Path-compressed find."""
        while parent[k] != k:
            parent[k] = parent[parent[k]]   # path compression
            k = parent[k]
        return k

    def union(a: str, b: str) -> None:
        """Merge the groups of a and b, making the longer name the root."""
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        canonical, duplicate = _pick_canonical(ra, rb)
        parent[duplicate] = canonical

    # ----------------------------------------------------------------
    # Pass 1: build merge groups
    # ----------------------------------------------------------------
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key_i = keys[i]
            key_j = keys[j]

            attr_i = node_data[key_i]
            attr_j = node_data[key_j]

            # Type guard — never merge across entity types
            if attr_i.get("Node_type") != attr_j.get("Node_type"):
                continue

            should_merge = False

            # --- Prefix / substring check ---
            shorter, longer = (
                (key_i, key_j) if len(key_i) <= len(key_j) else (key_j, key_i)
            )
            if shorter != longer and _is_substring_match(shorter, longer):
                should_merge = True

            # --- Fuzzy similarity check ---
            if not should_merge:
                score = _ratio(key_i, key_j)
                if score >= FUZZY_THRESHOLD:
                    should_merge = True

            if should_merge:
                union(key_i, key_j)

    # ----------------------------------------------------------------
    # Pass 2: collect groups
    # ----------------------------------------------------------------
    from collections import defaultdict
    groups: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        groups[find(k)].append(k)

    # ----------------------------------------------------------------
    # Pass 3: merge each group into its canonical entry
    # ----------------------------------------------------------------
    cleaned: dict = {}

    for root, members in groups.items():
        # The root is already the longest name (enforced by union's _pick_canonical)
        canonical_key = root
        merged_attrs = node_data[canonical_key].copy()

        for member in members:
            if member == canonical_key:
                continue
            merged_attrs = _merge_into(merged_attrs, node_data[member])

        # Ensure Label reflects the canonical (title-cased) name for readability
        # but keep the key lowercase, consistent with the rest of node_data
        if "Label" in merged_attrs and merged_attrs["Label"].lower() != canonical_key:
            merged_attrs["Label"] = canonical_key.title()

        cleaned[canonical_key] = merged_attrs

    return cleaned
