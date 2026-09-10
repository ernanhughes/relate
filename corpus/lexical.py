"""Cheap lexical-overlap features for RELATE DEV.

NOTE: the frozen RELATE v0.1 build (spec section 5) computes lexical_overlap as a
Jaccard over *lemmatized* content tokens. DEV uses a lemmatizer-free
approximation (lowercase, depunctuate, drop a short stopword list) so the
validators can run with stdlib only. The approximation is conservative for the
negation constraint (>= 0.7) because lemmatization only raises overlap.
"""
from __future__ import annotations

import re

_STOP = {
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "with", "and",
    "or", "but", "is", "are", "was", "were", "be", "been", "being", "as", "that",
    "this", "these", "those", "it", "its", "from", "into", "than", "then", "so",
    "not", "no", "did", "do", "does", "has", "have", "had", "will", "would",
    "s", "t",
}
_TOKEN = re.compile(r"[a-z0-9]+")


def _stem(w: str) -> str:
    """Light suffix strip — a lemmatizer stand-in for DEV so that inflected
    forms of the same content word (reject / rejected, acquire / acquired,
    lower / lowered) collapse. The frozen build uses a real lemmatizer."""
    for suf in ("ided", "ches", "shes", "sses", "ies"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)] + ("y" if suf == "ies" else ("" if suf in ("ches", "shes", "sses") else "id"))
    for suf in ("ing", "ed", "es", "s", "d"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            w = w[: -len(suf)]
            break
    if w.endswith("e") and len(w) > 3:   # acquire/acquir, lose/los -> collapse with -ed forms
        w = w[:-1]
    return w


def content_tokens(text: str) -> list[str]:
    return [_stem(w) for w in _TOKEN.findall(text.lower()) if w not in _STOP]


def jaccard(a: str, b: str) -> float:
    ta, tb = set(content_tokens(a)), set(content_tokens(b))
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def char_3grams(text: str) -> set[str]:
    s = re.sub(r"\s+", " ", text.lower().strip())
    return {s[i : i + 3] for i in range(max(0, len(s) - 2))}


def dice_3gram(a: str, b: str) -> float:
    ga, gb = char_3grams(a), char_3grams(b)
    if not ga and not gb:
        return 1.0
    return 2 * len(ga & gb) / (len(ga) + len(gb))


def entity_overlap(entities_a: list[str], entities_b: list[str]) -> int:
    return len(set(e.lower() for e in entities_a) & set(e.lower() for e in entities_b))
