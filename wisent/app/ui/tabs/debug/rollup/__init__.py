"""Roll an lm-eval task/subtask name up to its canonical top-level benchmark.

Three reverse-map sources, tried in order of authority:
  1. extractor registry — task names sharing one extractor ref are one
     benchmark family; the canonical member among them is the rollup target.
     Catches families whose lm-eval names don't prefix-match the canonical
     name (e.g. arabic_leaderboard_arabic_mt_hellaswag -> the canonical
     arabic_leaderboard_complete, both served by ArabicLeaderboardComplete).
  2. GROUP_TASK_EXPANSIONS — the static group->subtasks table.
  3. name-prefix stripping — collapse <canonical>_<suffix> at any depth.
"""
import json
import os


def canonical_benchmarks() -> list:
    """The canonical top-level benchmarks from benchmark_tags.json (the tagged
    list shipped with the wisent package)."""
    import wisent.support
    p = os.path.join(os.path.dirname(wisent.support.__file__), "examples",
                     "scripts", "benchmark_tags.json")
    with open(p, "r") as f:
        return sorted(json.load(f).keys())


_EXTRACTOR_REV: dict = {}
_GROUP_REV: dict = {}


def _extractor_reverse(canon_set: frozenset) -> dict:
    """subtask -> canonical sibling, from the extractor registry. Task names
    that map to the same extractor ref form one benchmark family; map every
    non-canonical member to a canonical member of its family (preferring a
    canonical that prefixes the name, else the shortest canonical name —
    deterministic when a family has >1 canonical config like _complete/_light)."""
    key = id(canon_set)
    if key in _EXTRACTOR_REV:
        return _EXTRACTOR_REV[key]
    from wisent.extractors.lm_eval.lm_extractor_registry import _REGISTRY
    by_ref: dict = {}
    for name, ref in _REGISTRY.items():
        by_ref.setdefault(ref, []).append(name)
    rev: dict = {}
    for names in by_ref.values():
        canon_here = [n for n in names if n in canon_set]
        if not canon_here:
            continue
        for n in names:
            if n in canon_set:
                continue
            owner = next((c for c in canon_here if n.startswith(c)), None)
            rev[n] = owner or min(canon_here, key=len)
    _EXTRACTOR_REV[key] = rev
    return rev


def _group_reverse() -> dict:
    """subtask -> top-level group from the GROUP_TASK_EXPANSIONS static table.
    Consulted only when the extractor registry doesn't resolve a name."""
    if not _GROUP_REV:
        from wisent.core.utils.infra_tools.data.loaders.lm_eval.\
            _lm_loader_task_mapping import GROUP_TASK_EXPANSIONS
        for g, subs in GROUP_TASK_EXPANSIONS.items():
            for s in subs:
                _GROUP_REV.setdefault(s, g)
    return _GROUP_REV


def rollup_to_canonical(task: str, canon_set):
    """Roll a task up to its canonical top-level benchmark. Order: exact
    canonical; extractor-registry sibling; GROUP_TASK_EXPANSIONS group (if
    canonical); strip trailing _segments to the longest canonical prefix.
    Returns None if no canonical owner."""
    cs = canon_set if isinstance(canon_set, frozenset) else frozenset(canon_set)
    if task in cs:
        return task
    e = _extractor_reverse(cs).get(task)
    if e:
        return e
    g = _group_reverse().get(task)
    if g and g in cs:
        return g
    s = task
    while s:
        if s in cs:
            return s
        if "_" not in s:
            return None
        s = s.rsplit("_", 1)[0]
    return None
