from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


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


def _load_paper_titles(documents_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for d in _iter_jsonl(documents_path):
        if d.get("doc_type") != "paper":
            continue
        doc_id = d.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id.startswith("arxiv:"):
            continue
        aid = doc_id.split(":", 1)[1].strip()
        if not aid:
            continue
        title = d.get("title")
        if isinstance(title, str) and title.strip():
            out[f"Paper:{aid}"] = title.strip()
    return out


def _keyword_from_title(title: str) -> str:
    t = str(title or "").strip()
    if not t:
        return ""
    if ":" in t:
        left = t.split(":", 1)[0].strip()
        if 2 <= len(left) <= 28:
            return left
    toks = [x for x in t.replace("(", " ").replace(")", " ").replace(",", " ").split() if x]
    if not toks:
        return ""
    for n in (1, 2, 3):
        cand = " ".join(toks[:n]).strip()
        if 2 <= len(cand) <= 24:
            return cand
    return toks[0][:24]


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


def _bfs_expand(
    triples: Sequence[Tuple[str, str, str]],
    *,
    seeds: Sequence[str],
    allowed_relations: Set[str],
    allowed_types: Set[str],
    max_new_nodes: int,
) -> Tuple[Set[str], List[Tuple[str, str, str]]]:
    adj: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for h, r, t in triples:
        if r not in allowed_relations:
            continue
        if _etype(h) not in allowed_types or _etype(t) not in allowed_types:
            continue
        adj[h].append((h, r, t))
        adj[t].append((h, r, t))

    seen: Set[str] = set(seeds)
    q: List[str] = list(seeds)
    edges: List[Tuple[str, str, str]] = []
    new_cnt = 0
    i = 0
    while i < len(q) and new_cnt < max_new_nodes:
        cur = q[i]
        i += 1
        for h, r, t in adj.get(cur, []):
            other = t if h == cur else h
            if other not in seen:
                if new_cnt >= max_new_nodes:
                    break
                seen.add(other)
                q.append(other)
                new_cnt += 1
                edges.append((h, r, t))
    return seen, edges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefer_paper", type=str, default="1812.03828")
    ap.add_argument("--triples_train", type=str, default="data/preprocessed/final/triples_train.jsonl")
    ap.add_argument("--triples_test", type=str, default="data/preprocessed/final/triples_test.jsonl")
    ap.add_argument("--documents_path", type=str, default="data/preprocessed/text/documents.jsonl")
    ap.add_argument("--bg_nodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="output/vis")
    ap.add_argument("--out_name", type=str, default="compact_4x4_fixed_bg20.png")
    args = ap.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    import networkx as nx

    root = Path(__file__).resolve().parents[1]
    train_path = (root / str(args.triples_train)).resolve() if not Path(str(args.triples_train)).is_absolute() else Path(str(args.triples_train)).resolve()
    test_path = (root / str(args.triples_test)).resolve() if not Path(str(args.triples_test)).is_absolute() else Path(str(args.triples_test)).resolve()
    docs_path = (root / str(args.documents_path)).resolve() if not Path(str(args.documents_path)).is_absolute() else Path(str(args.documents_path)).resolve()

    triples = _load_triples([train_path, test_path])
    prefer = str(args.prefer_paper).strip()
    prefer_name = f"Paper:{prefer}" if prefer and not prefer.startswith("Paper:") else prefer
    p, m, repo, ds, cite_p = _find_minimal_pattern(triples, prefer_paper=(prefer_name if prefer_name else None))

    allowed_relations = {"paper_proposes_method", "repo_implements_method", "method_uses_dataset", "paper_cites_paper"}
    allowed_types = {"Paper", "Method", "Repo", "Dataset"}
    core_nodes = [p, m, repo, ds, cite_p]
    core_edges = [
        (p, "paper_proposes_method", m),
        (repo, "repo_implements_method", m),
        (m, "method_uses_dataset", ds),
        (p, "paper_cites_paper", cite_p),
    ]

    paper_titles = _load_paper_titles(docs_path) if docs_path.exists() else {}

    bg_nodes = max(0, int(args.bg_nodes))
    _, bfs_edges = _bfs_expand(
        triples,
        seeds=core_nodes,
        allowed_relations=allowed_relations,
        allowed_types=allowed_types,
        max_new_nodes=bg_nodes,
    )

    G = nx.DiGraph()
    for n in core_nodes:
        G.add_node(n, etype=_etype(n), is_core=True)
    for h, r, t in core_edges:
        G.add_node(h, etype=_etype(h), is_core=True)
        G.add_node(t, etype=_etype(t), is_core=True)
        G.add_edge(h, t, rel=r, is_core=True)
    for h, r, t in bfs_edges:
        if not G.has_node(h):
            G.add_node(h, etype=_etype(h), is_core=False)
        if not G.has_node(t):
            G.add_node(t, etype=_etype(t), is_core=False)
        if not G.has_edge(h, t):
            G.add_edge(h, t, rel=r, is_core=False)

    pos_core = {
        p: (0.0, 0.0),
        m: (1.8, 0.0),
        repo: (1.8, 1.2),
        ds: (1.8, -1.2),
        cite_p: (-1.8, 0.0),
    }
    fixed = list(pos_core.keys())

    rng = random.Random(int(args.seed))
    pos_init = dict(pos_core)
    for n in G.nodes():
        if n not in pos_init:
            pos_init[n] = (rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6))

    pos = nx.spring_layout(G, seed=int(args.seed), pos=pos_init, fixed=fixed, k=0.65, iterations=120, scale=None, center=(0.0, 0.0))

    limx = 4.2
    limy = 3.2
    for n in G.nodes():
        if n in fixed:
            continue
        x, y = pos[n]
        pos[n] = (limx * math.tanh(float(x) / limx), limy * math.tanh(float(y) / limy))

    type_fill = {
        "Paper": "#ef4444",
        "Method": "#2563eb",
        "Dataset": "#10b981",
        "Repo": "#f59e0b",
        "": "#94a3b8",
    }
    rel_color = {
        "paper_proposes_method": "#2563eb",
        "repo_implements_method": "#f59e0b",
        "method_uses_dataset": "#10b981",
        "paper_cites_paper": "#ef4444",
    }

    fig = plt.figure(figsize=(10.5, 6.5), dpi=240)
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title("Compact Subgraph (4 types + 4 relations)", fontsize=14)

    core_set = {n for n in G.nodes() if bool(G.nodes[n].get("is_core"))}
    bg_set = set(G.nodes()) - core_set

    if bg_set:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=sorted(bg_set),
            node_color="#f1f5f9",
            node_size=520,
            alpha=0.70,
            linewidths=1.0,
            edgecolors="#e2e8f0",
            ax=ax,
        )

    core_by_type: Dict[str, List[str]] = defaultdict(list)
    for n in core_nodes:
        core_by_type[_etype(n)].append(n)
    for typ, nodes in sorted(core_by_type.items(), key=lambda x: x[0]):
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=type_fill.get(typ, "#94a3b8"),
            node_size=2200 if typ == "Paper" else 1400,
            alpha=0.98,
            linewidths=2.0,
            edgecolors="white",
            ax=ax,
            label=typ,
        )

    bg_edges = [(u, v) for u, v in G.edges() if not bool(G[u][v].get("is_core"))]
    if bg_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=bg_edges,
            width=1.1,
            alpha=0.25,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            edge_color="#cbd5e1",
            ax=ax,
            connectionstyle="arc3,rad=0.10",
        )

    rel_to_edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for u, v in G.edges():
        if not bool(G[u][v].get("is_core")):
            continue
        rel_to_edges[str(G[u][v].get("rel") or "")].append((u, v))

    for rel, eds in sorted(rel_to_edges.items(), key=lambda x: x[0]):
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=eds,
            width=3.0,
            alpha=0.85,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            edge_color=rel_color.get(rel, "#6b7280"),
            ax=ax,
            connectionstyle="arc3,rad=0.10",
        )

    for n in sorted(bg_set):
        x, y = pos[n]
        ax.text(x, y + 0.02, "", fontsize=1, color="#cbd5e1", ha="center", va="center", alpha=0.0)

    paper_kw: Dict[str, str] = {}
    for pn in [p, cite_p]:
        title = paper_titles.get(pn, "")
        kw = _keyword_from_title(title) if title else ""
        if kw:
            paper_kw[pn] = kw

    label_offsets = {
        p: (-0.05, 0.28, "center", "bottom"),
        cite_p: (0.0, 0.28, "center", "bottom"),
        m: (0.05, -0.30, "center", "top"),
        repo: (0.0, 0.30, "center", "bottom"),
        ds: (0.0, -0.30, "center", "top"),
    }

    for n in core_nodes:
        x, y = pos[n]
        dx, dy, ha, va = label_offsets.get(n, (0.0, 0.20, "center", "bottom"))
        typ = _etype(n)
        if typ == "Paper":
            txt = _short(n)
            kw = paper_kw.get(n, "")
            if kw:
                txt = f"{txt}\n{kw}"
            ax.text(
                x + dx,
                y + dy,
                txt,
                fontsize=10.0,
                color="#111827",
                ha=ha,
                va=va,
                zorder=6,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#e5e7eb", "alpha": 0.92},
            )
        elif typ == "Repo":
            ax.text(
                x + dx,
                y + dy,
                _short(n),
                fontsize=9.6,
                color="#111827",
                ha=ha,
                va=va,
                zorder=6,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#e5e7eb", "alpha": 0.92},
            )
        else:
            ax.text(
                x + dx,
                y + dy,
                _short(n),
                fontsize=9.6,
                color="#111827",
                ha=ha,
                va=va,
                zorder=6,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#e5e7eb", "alpha": 0.92},
            )

    handles_types, labels_types = ax.get_legend_handles_labels()
    if handles_types and labels_types:
        ax.legend(handles_types, labels_types, loc="upper left", frameon=True, framealpha=0.95, fontsize=10)

    y0 = 0.96
    x0 = 0.02
    ax.text(x0, y0, "Relation colors:", transform=ax.transAxes, fontsize=9.5, color="#374151", va="top")
    y = y0 - 0.03
    for rel, col in [
        ("paper_proposes_method", rel_color["paper_proposes_method"]),
        ("repo_implements_method", rel_color["repo_implements_method"]),
        ("method_uses_dataset", rel_color["method_uses_dataset"]),
        ("paper_cites_paper", rel_color["paper_cites_paper"]),
    ]:
        ax.text(x0, y, "━━", transform=ax.transAxes, fontsize=11, color=col, va="top")
        ax.text(x0 + 0.035, y, rel, transform=ax.transAxes, fontsize=8.8, color="#374151", va="top")
        y -= 0.028

    out_dir = (root / str(args.out_dir)).resolve() if not Path(str(args.out_dir)).is_absolute() else Path(str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / str(args.out_name)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

