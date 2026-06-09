from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


def load_id_map(path: str) -> Tuple[Dict[str, int], List[str]]:
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty file: {path}")
    n = int(lines[0].strip())
    mapping: Dict[str, int] = {}
    names: List[Optional[str]] = [None] * n
    for ln in lines[1:]:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split("\t")
        if len(parts) != 2:
            continue
        name, idx_s = parts
        idx = int(idx_s)
        mapping[name] = idx
        if 0 <= idx < n:
            names[idx] = name
    if any(x is None for x in names):
        missing = sum(1 for x in names if x is None)
        raise ValueError(f"incomplete id map: {path}, missing={missing}")
    return mapping, [x for x in names if x is not None]


def load_triples_hrt(path: str) -> List[Tuple[int, int, int]]:
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty file: {path}")
    n = int(lines[0].strip())
    out: List[Tuple[int, int, int]] = []
    for ln in lines[1:]:
        ln = ln.strip()
        if not ln:
            continue
        h_s, t_s, r_s = ln.split()
        out.append((int(h_s), int(t_s), int(r_s)))
    if len(out) != n:
        raise ValueError(f"triple count mismatch: {path}, header={n}, actual={len(out)}")
    return out


@dataclass
class FilterIndex:
    tails_by_hr: Dict[Tuple[int, int], Set[int]]
    heads_by_rt: Dict[Tuple[int, int], Set[int]]


def build_filter_index(triples: Sequence[Tuple[int, int, int]]) -> FilterIndex:
    tails_by_hr: Dict[Tuple[int, int], Set[int]] = {}
    heads_by_rt: Dict[Tuple[int, int], Set[int]] = {}
    for h, t, r in triples:
        tails_by_hr.setdefault((h, r), set()).add(t)
        heads_by_rt.setdefault((r, t), set()).add(h)
    return FilterIndex(tails_by_hr=tails_by_hr, heads_by_rt=heads_by_rt)

