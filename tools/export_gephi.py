from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


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


def _iter_triples(paths: Sequence[Path]) -> Iterable[Tuple[str, str, str, float]]:
    for p in paths:
        if not p.exists():
            continue
        for tr in _read_jsonl(p):
            h = tr.get("h")
            r = tr.get("r")
            t = tr.get("t")
            if not (isinstance(h, str) and isinstance(r, str) and isinstance(t, str)):
                continue
            conf = tr.get("confidence")
            try:
                w = float(conf) if conf is not None else 1.0
            except Exception:
                w = 1.0
            yield (h, r, t, w)


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


def _score_for(name: str, meta: Dict[str, Dict[str, Any]], metric: str) -> float:
    m = meta.get(name, {})
    if not isinstance(m, dict):
        return 0.0
    v = m.get(metric)
    fv = _to_float(v)
    return float(fv) if fv is not None else 0.0


def _bfs_expand(seeds: Sequence[str], *, neighbors: Dict[str, Set[str]], hops: int, limit: int) -> Set[str]:
    if limit <= 0:
        return set()
    seen: Set[str] = set()
    q = deque([(s, 0) for s in seeds])
    for s in seeds:
        seen.add(s)
        if len(seen) >= limit:
            return seen
    while q and len(seen) < limit:
        cur, d = q.popleft()
        if d >= hops:
            continue
        for nb in neighbors.get(cur, set()):
            if nb in seen:
                continue
            seen.add(nb)
            if len(seen) >= limit:
                return seen
            q.append((nb, d + 1))
    return seen


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(repo_root / "tools" / "export_gephi.yaml"))
    args = ap.parse_args()

    cfg = _expand(_load_yaml(_resolve_path(args.config, base=repo_root)))
    paths = cfg.get("paths", {}) or {}
    sample_cfg = cfg.get("sample", {}) or {}
    gephi_cfg = cfg.get("gephi", {}) or {}

    data_dir = _resolve_path(paths.get("data_dir", "data/preprocessed/final"), base=repo_root)
    entity2id_path = _resolve_path(paths.get("entity2id_path", str(data_dir / "entity2id.txt")), base=repo_root)
    documents_path = _resolve_path(paths.get("documents_path", "data/preprocessed/text/documents.jsonl"), base=repo_root)
    out_dir = _resolve_path(paths.get("out_dir", "data/vis/gephi"), base=repo_root)
    triples_paths_raw = paths.get("triples_paths")
    if triples_paths_raw is None:
        triples_paths = [data_dir / "triples_train.jsonl", data_dir / "triples_test.jsonl"]
    else:
        triples_paths = [_resolve_path(p, base=repo_root) for p in (triples_paths_raw or [])]

    nodes_csv = out_dir / str(gephi_cfg.get("nodes_csv", "nodes.csv"))
    edges_csv = out_dir / str(gephi_cfg.get("edges_csv", "edges.csv"))

    out_dir.mkdir(parents=True, exist_ok=True)

    ent_names = _load_entity_names(entity2id_path)
    meta = _load_entity_meta(documents_path)

    edges: List[Tuple[str, str, str, float]] = list(_iter_triples(triples_paths))
    neighbors: Dict[str, Set[str]] = {}
    for h, r, t, w in edges:
        neighbors.setdefault(h, set()).add(t)
        neighbors.setdefault(t, set()).add(h)

    rng = random.Random(int(sample_cfg.get("seed", 42)))
    mode = str(sample_cfg.get("mode", "seed_bfs")).strip() or "seed_bfs"

    seeds: List[str] = []
    selected: Set[str] = set()
    type_counts: Dict[str, int] = {}

    if mode == "stratified_by_type":
        per_type_limit = int(sample_cfg.get("per_type_limit", 300))
        types = [str(x) for x in (sample_cfg.get("types") or ["Paper", "Method", "Repo", "Dataset"])]
        for t in types:
            cand = [n for n in ent_names if _entity_type(n) == t]
            if not cand:
                type_counts[t] = 0
                continue
            k = min(max(0, per_type_limit), len(cand))
            pick = rng.sample(cand, k) if k < len(cand) else list(cand)
            selected.update(pick)
            type_counts[t] = len(pick)
        seeds = sorted(selected)[:50]
    else:
        node_limit = int(sample_cfg.get("node_limit", 500))
        hops = int(sample_cfg.get("expand_hops", 1))
        top_papers = int(sample_cfg.get("top_papers_by_citation", 50))
        top_repos = int(sample_cfg.get("top_repos_by_stars", 50))
        seed_entities = [str(x) for x in (sample_cfg.get("seed_entities") or [])]

        papers = [n for n in ent_names if _entity_type(n) == "Paper"]
        repos = [n for n in ent_names if _entity_type(n) == "Repo"]

        papers_sorted = sorted(papers, key=lambda n: _score_for(n, meta, "citation_count"), reverse=True)
        repos_sorted = sorted(repos, key=lambda n: _score_for(n, meta, "repo_stargazers_count"), reverse=True)

        for x in seed_entities:
            if x in set(ent_names):
                seeds.append(x)
        seeds.extend(papers_sorted[: max(0, top_papers)])
        seeds.extend(repos_sorted[: max(0, top_repos)])
        if not seeds and ent_names:
            seeds.append(rng.choice(ent_names))
        seeds = list(dict.fromkeys(seeds))

        selected = _bfs_expand(seeds, neighbors=neighbors, hops=max(0, hops), limit=max(1, node_limit))

    edge_rows: List[Dict[str, Any]] = []
    for h, r, t, w in edges:
        if h not in selected or t not in selected:
            continue
        edge_rows.append({"Source": h, "Target": t, "Type": "Directed", "Label": r, "Weight": w})

    node_rows: List[Dict[str, Any]] = []
    for n in sorted(selected):
        t = _entity_type(n) or "Entity"
        m = meta.get(n, {})
        node_rows.append(
            {
                "Id": n,
                "Label": n,
                "Type": t,
                "citation_count": m.get("citation_count"),
                "repo_stars": m.get("repo_stargazers_count"),
                "year": m.get("year"),
            }
        )

    with nodes_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(node_rows[0].keys()) if node_rows else ["Id", "Label", "Type"])
        w.writeheader()
        for row in node_rows:
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})

    with edges_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(edge_rows[0].keys()) if edge_rows else ["Source", "Target", "Type", "Label", "Weight"])
        w.writeheader()
        for row in edge_rows:
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})

    meta_path = out_dir / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "created_at": _iso_now(),
                "nodes_csv": str(nodes_csv),
                "edges_csv": str(edges_csv),
                "triples_paths": [str(p) for p in triples_paths],
                "documents_path": str(documents_path),
                "node_count": len(node_rows),
                "edge_count": len(edge_rows),
                "mode": mode,
                "seed_count": len(seeds),
                "seeds": seeds[:50],
                "type_counts": type_counts,
                "node_limit": int(sample_cfg.get("node_limit", 500)) if mode != "stratified_by_type" else None,
                "expand_hops": int(sample_cfg.get("expand_hops", 1)) if mode != "stratified_by_type" else None,
                "per_type_limit": int(sample_cfg.get("per_type_limit", 300)) if mode == "stratified_by_type" else None,
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
