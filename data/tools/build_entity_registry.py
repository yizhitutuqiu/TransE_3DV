from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _write_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
            n += 1
    return n


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


def _progress(it: Iterable[Any], *, total: Optional[int], desc: str) -> Iterable[Any]:
    if _tqdm is None:
        return it
    return _tqdm(it, total=total, desc=desc)


def _normalize_ws(s: str) -> str:
    s = s.replace("\u00a0", " ").replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _canonical_key(s: str) -> str:
    s = _normalize_ws(s)
    if not s:
        return s
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._+\-:/()]*", s):
        s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _method_clean(name: str) -> Optional[str]:
    t = _canonical_key(name)
    t = t.strip(" \t\n\r\"'`")
    t = re.sub(r"\s+", " ", t)
    t = t.strip()
    if not t:
        return None
    if "http" in t or "www." in t:
        return None
    if len(t) < 3 or len(t) > 80:
        return None
    if re.fullmatch(r"[-_./:()]+", t):
        return None
    stop = {
        "method",
        "model",
        "approach",
        "framework",
        "network",
        "architecture",
        "pipeline",
        "baseline",
    }
    if t in stop:
        return None
    return t


def _default_alias_rules() -> Dict[str, str]:
    return {
        "3dgs": "3d gaussian splatting",
        "3d gaussian splatting": "3d gaussian splatting",
        "gaussian splatting": "3d gaussian splatting",
        "nerf": "nerf",
        "neural radiance field": "nerf",
        "neural radiance fields": "nerf",
        "mip-nerf": "mip-nerf",
        "instant-ngp": "instant-ngp",
        "splatting": "3d gaussian splatting",
        "sfm": "structure from motion",
        "structure from motion": "structure from motion",
        "mvs": "multi-view stereo",
        "multi-view stereo": "multi-view stereo",
        "slam": "slam",
    }


def _load_alias_rules(path: Path) -> Dict[str, str]:
    if path.exists():
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                        out[_canonical_key(k)] = _canonical_key(v)
                return out
        except Exception:
            pass
    d = _default_alias_rules()
    path.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out = {_canonical_key(k): _canonical_key(v) for k, v in d.items()}
    return out


def _map_canonical(typ: str, canonical: str, alias_rules: Dict[str, str]) -> Optional[str]:
    if typ == "Method":
        c = _method_clean(canonical)
        if c is None:
            return None
        return alias_rules.get(c, c)

    if typ == "Metric" and canonical.isalpha() and canonical.upper() == canonical and len(canonical) <= 4:
        return canonical

    if typ in {"Dataset", "Venue", "Task"}:
        return _canonical_key(canonical)

    return canonical


def _entity_id(typ: str, canonical: str) -> str:
    return f"{typ}:{canonical}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seed_registry_path",
        type=str,
        default=str((_data_dir() / "preprocessed" / "text" / "entities" / "entity_registry_seed.jsonl").resolve()),
    )
    ap.add_argument(
        "--doc_entities_path",
        type=str,
        default=str((_data_dir() / "preprocessed" / "text" / "entities" / "doc_entities.jsonl").resolve()),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str((_data_dir() / "preprocessed" / "kg" / "entities").resolve()),
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max_aliases_each", type=int, default=30)
    args = ap.parse_args()

    seed_path = Path(args.seed_registry_path).resolve()
    doc_entities_path = Path(args.doc_entities_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    alias_rules_path = out_dir / "alias_rules.json"
    alias_rules = _load_alias_rules(alias_rules_path)

    doc_freq: Dict[Tuple[str, str], int] = Counter()
    mention_freq: Dict[Tuple[str, str, str], int] = Counter()

    total_docs = _count_lines(doc_entities_path)
    for d in _progress(_iter_jsonl(doc_entities_path), total=total_docs, desc="registry:scan_docs"):
        ents = d.get("entities")
        if not isinstance(ents, list):
            continue
        seen_doc: Set[Tuple[str, str]] = set()
        for e in ents:
            if not isinstance(e, dict):
                continue
            typ = e.get("type")
            canonical = e.get("canonical")
            mention = e.get("mention")
            if not isinstance(typ, str) or not isinstance(canonical, str):
                continue
            if not isinstance(mention, str):
                mention = canonical
            mc = _map_canonical(typ, canonical, alias_rules)
            if mc is None:
                continue
            k = (typ, mc)
            if k not in seen_doc:
                doc_freq[k] += 1
                seen_doc.add(k)
            mention_freq[(typ, mc, _normalize_ws(mention))] += 1

    seeds: List[Tuple[str, str]] = []
    total_seed = _count_lines(seed_path)
    for it in _progress(_iter_jsonl(seed_path), total=total_seed, desc="registry:scan_seed"):
        typ = it.get("type")
        canonical = it.get("canonical")
        if not isinstance(typ, str) or not isinstance(canonical, str) or not typ or not canonical:
            continue
        mc = _map_canonical(typ, canonical, alias_rules)
        if mc is None:
            continue
        seeds.append((typ, mc))

    uniq = set(seeds)
    for k in doc_freq.keys():
        uniq.add(k)

    registry: List[Dict[str, Any]] = []
    by_type = Counter()

    for typ, canonical in sorted(uniq, key=lambda x: (x[0], x[1])):
        mids = [(m, c) for (t, mc, m), c in mention_freq.items() if t == typ and mc == canonical]
        mids.sort(key=lambda x: x[1], reverse=True)
        display_name = mids[0][0] if mids else canonical
        aliases: List[str] = []
        for m, _c in mids[: args.max_aliases_each]:
            aliases.append(m)
        entry = {
            "entity_id": _entity_id(typ, canonical),
            "type": typ,
            "canonical": canonical,
            "display_name": display_name,
            "aliases": aliases,
            "stats": {"doc_freq": int(doc_freq.get((typ, canonical), 0))},
            "created_at": _utc_now_iso(),
        }
        registry.append(entry)
        by_type[typ] += 1

    out_path = out_dir / "entity_registry.jsonl"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} exists, pass --overwrite")

    _write_jsonl(out_path, registry)
    stats = {
        "created_at": _utc_now_iso(),
        "seed_registry_path": str(seed_path),
        "doc_entities_path": str(doc_entities_path),
        "out_dir": str(out_dir),
        "entity_registry_path": str(out_path),
        "entity_count": len(registry),
        "by_type": dict(by_type),
        "alias_rules_path": str(alias_rules_path),
    }
    _write_json(out_dir / "stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

