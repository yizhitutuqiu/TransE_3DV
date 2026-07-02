from __future__ import annotations

import argparse
import json
import os
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _etype(name: str) -> str:
    if ":" not in name:
        return ""
    return name.split(":", 1)[0]


def _short(name: str) -> str:
    if ":" not in name:
        return name
    return name.split(":", 1)[1]


def _load_triples(paths: Sequence[Path]) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for p in paths:
        for it in _iter_jsonl(p):
            h = it.get("h")
            r = it.get("r")
            t = it.get("t")
            if isinstance(h, str) and isinstance(r, str) and isinstance(t, str):
                out.append((h, r, t))
    return out


def _find_minimal_pattern(
    triples: Sequence[Tuple[str, str, str]],
    *,
    prefer_paper: Optional[str],
) -> Tuple[str, str, str, str, str]:
    proposes: Dict[str, Set[str]] = defaultdict(set)
    implements: Dict[str, Set[str]] = defaultdict(set)
    uses: Dict[str, Set[str]] = defaultdict(set)
    cites: Dict[str, Set[str]] = defaultdict(set)

    for h, r, t in triples:
        if r == "paper_proposes_method":
            proposes[h].add(t)
        elif r == "repo_implements_method":
            implements[t].add(h)
        elif r == "method_uses_dataset":
            uses[h].add(t)
        elif r == "paper_cites_paper":
            cites[h].add(t)

    def pick_for_paper(paper: str) -> Optional[Tuple[str, str, str, str, str, Tuple[int, int, int]]]:
        ms = proposes.get(paper) or set()
        if not ms:
            return None
        cs = cites.get(paper) or set()
        if not cs:
            return None
        best: Optional[Tuple[str, str, str, str, str, Tuple[int, int, int]]] = None
        for m in ms:
            rs = implements.get(m) or set()
            ds = uses.get(m) or set()
            if not rs or not ds:
                continue
            repo = sorted(rs)[0]
            dataset = sorted(ds)[0]
            cite_p = sorted(cs)[0]
            deg = (len(rs), len(ds), len(cs))
            cand = (paper, m, repo, dataset, cite_p, deg)
            if best is None or deg < best[-1]:
                best = cand
        return best

    if prefer_paper:
        got = pick_for_paper(prefer_paper)
        if got is not None:
            p, m, repo, ds, cite_p, _ = got
            return p, m, repo, ds, cite_p

    best_any: Optional[Tuple[str, str, str, str, str, Tuple[int, int, int]]] = None
    for p in proposes.keys():
        got = pick_for_paper(p)
        if got is None:
            continue
        if best_any is None or got[-1] < best_any[-1]:
            best_any = got
    if best_any is None:
        raise RuntimeError("no compact pattern found that covers 4 relations + 4 types")
    p, m, repo, ds, cite_p, _ = best_any
    return p, m, repo, ds, cite_p


def _find_minimal_pattern_excluding(
    triples: Sequence[Tuple[str, str, str]],
    *,
    prefer_paper: Optional[str],
    exclude_nodes: Set[str],
) -> Tuple[str, str, str, str, str]:
    proposes: Dict[str, Set[str]] = defaultdict(set)
    implements: Dict[str, Set[str]] = defaultdict(set)
    uses: Dict[str, Set[str]] = defaultdict(set)
    cites: Dict[str, Set[str]] = defaultdict(set)

    for h, r, t in triples:
        if r == "paper_proposes_method":
            proposes[h].add(t)
        elif r == "repo_implements_method":
            implements[t].add(h)
        elif r == "method_uses_dataset":
            uses[h].add(t)
        elif r == "paper_cites_paper":
            cites[h].add(t)

    def pick_for_paper(paper: str) -> Optional[Tuple[str, str, str, str, str, Tuple[int, int, int]]]:
        ms = proposes.get(paper) or set()
        if not ms:
            return None
        cs = cites.get(paper) or set()
        if not cs:
            return None
        best: Optional[Tuple[str, str, str, str, str, Tuple[int, int, int]]] = None
        for m in sorted(ms):
            rs = implements.get(m) or set()
            ds = uses.get(m) or set()
            if not rs or not ds:
                continue
            repo = sorted(rs)[0]
            dataset = sorted(ds)[0]
            cite_p = sorted(cs)[0]
            nodes = {paper, m, repo, dataset, cite_p}
            if nodes & exclude_nodes:
                continue
            deg = (len(rs), len(ds), len(cs))
            cand = (paper, m, repo, dataset, cite_p, deg)
            if best is None or deg < best[-1]:
                best = cand
        return best

    if prefer_paper:
        got = pick_for_paper(prefer_paper)
        if got is not None:
            p, m, repo, ds, cite_p, _ = got
            return p, m, repo, ds, cite_p

    best_any: Optional[Tuple[str, str, str, str, str, Tuple[int, int, int]]] = None
    for p in sorted(proposes.keys()):
        got = pick_for_paper(p)
        if got is None:
            continue
        if best_any is None or got[-1] < best_any[-1]:
            best_any = got
    if best_any is None:
        raise RuntimeError("no compact pattern found that covers 4 relations + 4 types (after exclude)")
    p, m, repo, ds, cite_p, _ = best_any
    return p, m, repo, ds, cite_p


def _build_adj(
    triples: Sequence[Tuple[str, str, str]],
    *,
    allowed_relations: Set[str],
    allowed_types: Set[str],
) -> DefaultDict[str, List[Tuple[str, str, str]]]:
    adj: DefaultDict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for h, r, t in triples:
        if r not in allowed_relations:
            continue
        if _etype(h) not in allowed_types or _etype(t) not in allowed_types:
            continue
        adj[h].append((h, r, t))
        adj[t].append((h, r, t))
    for k in list(adj.keys()):
        adj[k].sort(key=lambda x: (x[1], x[0], x[2]))
    return adj


def _bfs_pick_bg(
    adj: Dict[str, List[Tuple[str, str, str]]],
    *,
    core_nodes: Set[str],
    seeds: Sequence[str],
    bg_target: int,
    core_degree_cap: int,
    bg_degree_cap: int,
    initial_degrees: Optional[Dict[str, int]] = None,
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    seen: Set[str] = set(seeds)
    bg_nodes: List[str] = []
    edges: List[Tuple[str, str, str]] = []

    deg: Dict[str, int] = defaultdict(int)
    if isinstance(initial_degrees, dict):
        for k, v in initial_degrees.items():
            if isinstance(k, str) and isinstance(v, int):
                deg[k] = int(v)

    q: List[str] = list(seeds)
    i = 0
    while i < len(q) and len(bg_nodes) < bg_target:
        cur = q[i]
        i += 1

        cap_cur = int(core_degree_cap if cur in core_nodes else bg_degree_cap)
        if deg.get(cur, 0) >= cap_cur:
            continue

        for h, r, t in adj.get(cur, []):
            if len(bg_nodes) >= bg_target:
                break
            other = t if h == cur else h

            if other not in seen:
                if other in core_nodes:
                    seen.add(other)
                else:
                    seen.add(other)
                    bg_nodes.append(other)
                    q.append(other)

            def _cap(x: str) -> int:
                return int(core_degree_cap if x in core_nodes else bg_degree_cap)

            def can_add_endpoint(x: str) -> bool:
                return deg.get(x, 0) < _cap(x)

            if not (can_add_endpoint(h) and can_add_endpoint(t)):
                continue

            edges.append((h, r, t))
            deg[h] += 1
            deg[t] += 1

            if deg.get(cur, 0) >= cap_cur:
                break

    return bg_nodes, edges


def render(
    *,
    prefer_paper: str,
    triples: Sequence[Tuple[str, str, str]],
    out_path: Path,
    bg_nodes_target: int,
    core_degree_cap: int,
    bg_degree_cap: int,
    core_scale: float,
    spring_k: float,
    spring_iter: int,
    bg_ring_radius: float,
    bg_jitter: float,
    seed: int,
    exclude_core_nodes: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    import networkx as nx

    def _set_zorder(obj: Any, z: int) -> None:
        if obj is None:
            return
        if isinstance(obj, dict):
            for v in obj.values():
                _set_zorder(v, z)
            return
        if isinstance(obj, (list, tuple)):
            for v in obj:
                _set_zorder(v, z)
            return
        m = getattr(obj, "set_zorder", None)
        if callable(m):
            m(z)

    exclude = exclude_core_nodes or set()
    prefer_s = str(prefer_paper).strip()
    prefer_name = f"Paper:{prefer_s}" if prefer_s and not prefer_s.startswith("Paper:") else prefer_s

    if exclude:
        p, m, repo, ds, cite_p = _find_minimal_pattern_excluding(triples, prefer_paper=(prefer_name if prefer_name else None), exclude_nodes=exclude)
    else:
        p, m, repo, ds, cite_p = _find_minimal_pattern(triples, prefer_paper=(prefer_name if prefer_name else None))

    allowed_relations = {"paper_proposes_method", "repo_implements_method", "method_uses_dataset", "paper_cites_paper"}
    allowed_types = {"Paper", "Method", "Repo", "Dataset"}
    adj = _build_adj(triples, allowed_relations=allowed_relations, allowed_types=allowed_types)

    core_nodes_list = [p, m, repo, ds, cite_p]
    core_nodes = set(core_nodes_list)
    core_edges = [
        (p, "paper_proposes_method", m),
        (repo, "repo_implements_method", m),
        (m, "method_uses_dataset", ds),
        (p, "paper_cites_paper", cite_p),
    ]

    core_deg: Dict[str, int] = defaultdict(int)
    for h, _, t in core_edges:
        core_deg[h] += 1
        core_deg[t] += 1

    core_cap = int(core_degree_cap)
    bg_cap = int(bg_degree_cap)
    for _, d in list(core_deg.items()):
        if d > core_cap:
            core_cap = d

    bg_nodes, bg_edges = _bfs_pick_bg(
        adj,
        core_nodes=core_nodes,
        seeds=core_nodes_list,
        bg_target=max(0, int(bg_nodes_target)),
        core_degree_cap=max(1, core_cap),
        bg_degree_cap=max(1, bg_cap),
        initial_degrees=dict(core_deg),
    )

    G = nx.DiGraph()
    for n in core_nodes_list:
        G.add_node(n, etype=_etype(n), is_core=True)
    for h, r, t in core_edges:
        G.add_edge(h, t, rel=r, is_core=True)
    for n in bg_nodes:
        G.add_node(n, etype=_etype(n), is_core=False)
    for h, r, t in bg_edges:
        if h not in G.nodes() or t not in G.nodes():
            continue
        if G.has_edge(h, t):
            continue
        G.add_edge(h, t, rel=r, is_core=False)

    s = float(core_scale)
    if s <= 0:
        s = 1.0
    pos_core = {
        p: (0.0 * s, 0.0 * s),
        m: (1.8 * s, 0.0 * s),
        repo: (1.8 * s, 1.2 * s),
        ds: (1.8 * s, -1.2 * s),
        cite_p: (-1.8 * s, 0.0 * s),
    }
    fixed = list(pos_core.keys())

    rng = random.Random(int(seed))
    pos_init = dict(pos_core)
    bg_only = [n for n in G.nodes() if n not in pos_init]
    bg_only.sort()
    ring = float(bg_ring_radius)
    if ring <= 0:
        ring = 2.9 * s
    jitter = float(bg_jitter)
    if jitter < 0:
        jitter = 0.0
    if bg_only:
        for idx, n in enumerate(bg_only):
            ang = 2.0 * math.pi * (idx / max(1, len(bg_only)))
            rr = ring * (0.92 + 0.16 * rng.random())
            pos_init[n] = (
                rr * math.cos(ang) + rng.uniform(-jitter, jitter),
                rr * math.sin(ang) + rng.uniform(-jitter, jitter),
            )

    n_total = max(1, int(G.number_of_nodes()))
    k = float(spring_k)
    if k <= 0:
        k = max(0.55, (2.2 * s) / (n_total**0.5))
    iters = max(80, int(spring_iter))
    scale = max(2.6, ring * 1.15)
    pos = nx.spring_layout(
        G.to_undirected(),
        seed=int(seed),
        pos=pos_init,
        fixed=fixed,
        k=k,
        iterations=iters,
        scale=scale,
        center=(0.0, 0.0),
    )

    type_colors = {
        "Paper": "#ef4444",
        "Method": "#2563eb",
        "Dataset": "#10b981",
        "Repo": "#f59e0b",
        "": "#94a3b8",
    }

    fig = plt.figure(figsize=(10.5, 6.5), dpi=220)
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title("Compact Subgraph (4 types + 4 relations)", fontsize=14)

    bg_set = set(bg_nodes)
    if bg_set:
        bg_nodes_artist = nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=sorted(bg_set),
            node_color="#d1d5db",
            node_size=520,
            alpha=0.85,
            linewidths=1.2,
            edgecolors="#9ca3af",
            ax=ax,
        )
        _set_zorder(bg_nodes_artist, 1)

    core_by_type: Dict[str, List[str]] = defaultdict(list)
    for n in core_nodes_list:
        core_by_type[_etype(n)].append(n)
    for typ, nodes in sorted(core_by_type.items(), key=lambda x: x[0]):
        core_nodes_artist = nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=type_colors.get(typ, "#94a3b8"),
            node_size=2100 if typ == "Paper" else 1300,
            alpha=0.95,
            linewidths=2.0,
            edgecolors="white",
            ax=ax,
            label=typ or "Other",
        )
        _set_zorder(core_nodes_artist, 4)

    bg_edgelist = [(u, v) for (u, v) in G.edges() if not bool(G[u][v].get("is_core"))]
    if bg_edgelist:
        bg_edges_artist = nx.draw_networkx_edges(
            G,
            pos,
            edgelist=bg_edgelist,
            width=1.2,
            alpha=0.40,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            edge_color="#9ca3af",
            ax=ax,
            connectionstyle="arc3,rad=0.10",
        )
        _set_zorder(bg_edges_artist, 1)

    core_edgelist = [(u, v) for (u, v) in G.edges() if bool(G[u][v].get("is_core"))]
    core_edges_artist = nx.draw_networkx_edges(
        G,
        pos,
        edgelist=core_edgelist,
        width=2.2,
        alpha=0.7,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        edge_color="#6b7280",
        ax=ax,
        connectionstyle="arc3,rad=0.10",
    )
    _set_zorder(core_edges_artist, 3)

    labels = {n: _short(n) for n in core_nodes_list}
    core_label_artist = nx.draw_networkx_labels(G, pos, labels=labels, font_size=10.5, font_color="#111827", ax=ax)
    _set_zorder(core_label_artist, 6)

    edge_labels = {(u, v): G[u][v].get("rel", "") for u, v in core_edgelist}
    core_edge_label_artist = nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=9.2,
        font_color="#111827",
        ax=ax,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#e5e7eb", "alpha": 0.95},
        rotate=False,
    )
    _set_zorder(core_edge_label_artist, 5)

    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=10)
    note = (
        f"bg_nodes={len(bg_nodes)} (target={int(bg_nodes_target)}), "
        f"core_degree_cap={int(core_cap)}, bg_degree_cap={int(bg_cap)}"
    )
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=9, color="#374151", va="bottom")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "out_path": str(out_path),
        "core_nodes": core_nodes_list,
        "core_edges": core_edges,
        "bg_nodes": bg_nodes,
        "bg_edges": bg_edges,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefer_paper", type=str, default="1812.03828")
    ap.add_argument("--triples_train", type=str, default="data/preprocessed/final/triples_train.jsonl")
    ap.add_argument("--triples_test", type=str, default="data/preprocessed/final/triples_test.jsonl")
    ap.add_argument("--bg_nodes", type=int, default=25)
    ap.add_argument("--core_degree_cap", type=int, default=5)
    ap.add_argument("--bg_degree_cap", type=int, default=5)
    ap.add_argument("--core_scale", type=float, default=1.25)
    ap.add_argument("--spring_k", type=float, default=0.0)
    ap.add_argument("--spring_iter", type=int, default=260)
    ap.add_argument("--bg_ring_radius", type=float, default=0.0)
    ap.add_argument("--bg_jitter", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="output/vis")
    ap.add_argument("--out_name", type=str, default="compact_4x4_min_bg.png")
    args = ap.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    import networkx as nx

    root = Path(__file__).resolve().parents[1]
    train_path = (root / str(args.triples_train)).resolve() if not Path(str(args.triples_train)).is_absolute() else Path(str(args.triples_train)).resolve()
    test_path = (root / str(args.triples_test)).resolve() if not Path(str(args.triples_test)).is_absolute() else Path(str(args.triples_test)).resolve()

    triples = _load_triples([train_path, test_path])
    out_dir = (root / str(args.out_dir)).resolve() if not Path(str(args.out_dir)).is_absolute() else Path(str(args.out_dir)).resolve()
    out_path = out_dir / str(args.out_name)
    render(
        prefer_paper=str(args.prefer_paper),
        triples=triples,
        out_path=out_path,
        bg_nodes_target=int(args.bg_nodes),
        core_degree_cap=int(args.core_degree_cap),
        bg_degree_cap=int(args.bg_degree_cap),
        core_scale=float(args.core_scale),
        spring_k=float(args.spring_k),
        spring_iter=int(args.spring_iter),
        bg_ring_radius=float(args.bg_ring_radius),
        bg_jitter=float(args.bg_jitter),
        seed=int(args.seed),
        exclude_core_nodes=set(),
    )

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
