from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError("config must be a mapping")
    return obj


def _expand(v: Any) -> Any:
    if isinstance(v, str):
        return os.path.expandvars(v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def _resolve_path(p: Any, *, base: Path) -> Path:
    if p is None:
        return base
    s = str(p).strip()
    if not s:
        return base
    pp = Path(s).expanduser()
    if pp.is_absolute():
        return pp.resolve()
    cand = (Path.cwd() / pp).resolve()
    if cand.exists():
        return cand
    return (base / pp).resolve()


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _entity_type(name: str) -> str:
    if ":" not in name:
        return ""
    return name.split(":", 1)[0].strip()


def _load_entity_meta(documents_path: Path) -> Dict[str, Dict[str, Any]]:
    entity_meta: Dict[str, Dict[str, Any]] = {}
    if not documents_path.exists():
        return entity_meta
    for d in _read_jsonl(documents_path):
        meta = d.get("metadata")
        if not isinstance(meta, dict):
            continue
        if d.get("doc_type") == "paper":
            aid = meta.get("arxiv_id")
            if isinstance(aid, str) and aid.strip():
                k = f"Paper:{aid.strip()}"
                entity_meta[k] = {
                    "citation_count": meta.get("citation_count"),
                    "reference_count": meta.get("reference_count"),
                    "year": meta.get("year"),
                    "title": d.get("title"),
                }
        elif d.get("doc_type") == "readme":
            full = meta.get("repo_full_name") or meta.get("full_name")
            if isinstance(full, str) and full.strip():
                k = f"Repo:{full.strip()}"
                entity_meta[k] = {
                    "repo_stargazers_count": meta.get("repo_stargazers_count"),
                    "repo_forks_count": meta.get("repo_forks_count"),
                    "repo_open_issues_count": meta.get("repo_open_issues_count"),
                    "repo_updated_at": meta.get("repo_updated_at"),
                    "repo_created_at": meta.get("repo_created_at"),
                }
    return entity_meta


def _load_entity_names(entity2id_path: Path) -> List[str]:
    names: List[str] = []
    with entity2id_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            names.append(parts[0])
    return names


def _iter_triples(paths: Sequence[Path]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        if not p.exists():
            continue
        yield from _read_jsonl(p)


def _to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return int(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float):
        return float(v)
    if isinstance(v, int):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(repo_root / "tools" / "export_neo4j.yaml"))
    args = ap.parse_args()

    cfg = _expand(_load_yaml(_resolve_path(args.config, base=repo_root)))
    paths = cfg.get("paths", {}) or {}

    data_dir = _resolve_path(paths.get("data_dir", "data/preprocessed/final"), base=repo_root)
    entity2id_path = _resolve_path(paths.get("entity2id_path", str(data_dir / "entity2id.txt")), base=repo_root)
    documents_path = _resolve_path(paths.get("documents_path", "data/preprocessed/text/documents.jsonl"), base=repo_root)
    out_dir = _resolve_path(paths.get("out_dir", "data/vis/neo4j"), base=repo_root)
    triples_paths_raw = paths.get("triples_paths")
    if triples_paths_raw is None:
        triples_paths = [data_dir / "triples_train.jsonl", data_dir / "triples_test.jsonl"]
    else:
        triples_paths = [_resolve_path(p, base=repo_root) for p in (triples_paths_raw or [])]

    neo_cfg = cfg.get("neo4j", {}) or {}
    nodes_csv = out_dir / str(neo_cfg.get("nodes_csv", "nodes.csv"))
    rels_csv = out_dir / str(neo_cfg.get("rels_csv", "relationships.csv"))
    delimiter = str(neo_cfg.get("delimiter", ","))

    out_dir.mkdir(parents=True, exist_ok=True)

    ent_names = _load_entity_names(entity2id_path)
    meta = _load_entity_meta(documents_path)

    node_rows: List[Dict[str, Any]] = []
    for name in ent_names:
        t = _entity_type(name)
        m = meta.get(name, {})
        row: Dict[str, Any] = {
            ":ID": name,
            "name": name,
            "type": t,
            ":LABEL": t or "Entity",
            "citation_count:long": _to_int(m.get("citation_count")),
            "reference_count:long": _to_int(m.get("reference_count")),
            "year:long": _to_int(m.get("year")),
            "repo_stargazers_count:long": _to_int(m.get("repo_stargazers_count")),
            "repo_forks_count:long": _to_int(m.get("repo_forks_count")),
            "repo_open_issues_count:long": _to_int(m.get("repo_open_issues_count")),
            "repo_updated_at": m.get("repo_updated_at"),
            "repo_created_at": m.get("repo_created_at"),
            "title": m.get("title"),
        }
        node_rows.append(row)

    rel_rows: List[Dict[str, Any]] = []
    for tr in _iter_triples(triples_paths):
        h = tr.get("h")
        r = tr.get("r")
        t = tr.get("t")
        if not (isinstance(h, str) and isinstance(r, str) and isinstance(t, str)):
            continue
        rel_rows.append(
            {
                ":START_ID": h,
                ":END_ID": t,
                ":TYPE": r,
                "doc_id": tr.get("doc_id"),
                "source": tr.get("source"),
                "confidence:float": _to_float(tr.get("confidence")),
            }
        )

    node_fieldnames = list(node_rows[0].keys()) if node_rows else [":ID", "name", "type", ":LABEL"]
    rel_fieldnames = list(rel_rows[0].keys()) if rel_rows else [":START_ID", ":END_ID", ":TYPE"]

    with nodes_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=node_fieldnames, delimiter=delimiter)
        w.writeheader()
        for row in node_rows:
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})

    with rels_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rel_fieldnames, delimiter=delimiter)
        w.writeheader()
        for row in rel_rows:
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})

    meta_path = out_dir / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "created_at": _iso_now(),
                "nodes_csv": str(nodes_csv),
                "rels_csv": str(rels_csv),
                "triples_paths": [str(p) for p in triples_paths],
                "documents_path": str(documents_path),
                "node_count": len(node_rows),
                "rel_count": len(rel_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

