from __future__ import annotations

import argparse
import json
import math
import os
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


def _truncate(s: str, n: int) -> str:
    s = str(s or "").strip()
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)].rstrip() + "…"


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
    ap.add_argument("--out_dir", type=str, default="output/vis")
    ap.add_argument("--out_name", type=str, default="compact_4x4.png")
    ap.add_argument("--bg_nodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
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
    core_edges = {
        (p, m),
        (repo, m),
        (m, ds),
        (p, cite_p),
    }

    all_titles = _load_paper_titles(docs_path) if docs_path.exists() else {}

    bg_n = max(0, int(args.bg_nodes))
    picked_nodes, picked_edges = _bfs_expand(
        triples,
        seeds=core_nodes,
        allowed_relations=allowed_relations,
        allowed_types=allowed_types,
        max_new_nodes=bg_n,
    )

    G = nx.DiGraph()
    for n in core_nodes:
        G.add_node(n, etype=_etype(n))
    G.add_edge(p, m, rel="paper_proposes_method")
    G.add_edge(repo, m, rel="repo_implements_method")
    G.add_edge(m, ds, rel="method_uses_dataset")
    G.add_edge(p, cite_p, rel="paper_cites_paper")

    for h, r, t in picked_edges:
        G.add_node(h, etype=_etype(h))
        G.add_node(t, etype=_etype(t))
        if not G.has_edge(h, t):
            G.add_edge(h, t, rel=r)

    pos = {
        p: (0.0, 0.0),
        m: (1.8, 0.0),
        repo: (1.8, 1.2),
        ds: (1.8, -1.2),
        cite_p: (-1.8, 0.0),
    }

    type_colors = {
        "Paper": "#ef4444",
        "Method": "#2563eb",
        "Dataset": "#10b981",
        "Repo": "#f59e0b",
        "": "#94a3b8",
    }

    fixed = list(pos.keys())
    pos_init = dict(pos)
    rng = __import__("random").Random(int(args.seed))
    for n in G.nodes():
        if n not in pos_init:
            pos_init[n] = (rng.uniform(-0.7, 0.7), rng.uniform(-0.7, 0.7))
    pos = nx.spring_layout(G, seed=int(args.seed), pos=pos_init, fixed=fixed, k=0.95, iterations=200, scale=None, center=(0.0, 0.0))
    limx = 4.8
    limy = 3.6
    for n in G.nodes():
        if n in fixed:
            continue
        x, y = pos[n]
        x = limx * math.tanh(float(x) / limx)
        y = limy * math.tanh(float(y) / limy)
        pos[n] = (x, y)

    fig = plt.figure(figsize=(12.0, 7.2), dpi=220)
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title("Compact Subgraph (4 types + 4 relations)", fontsize=14)

    core_set = set(core_nodes)
    bg_set = set(G.nodes()) - core_set

    def draw_group(nodes: List[str], *, alpha: float, is_bg: bool) -> None:
        node_types: Dict[str, List[str]] = defaultdict(list)
        for n in nodes:
            node_types[G.nodes[n].get("etype", "")].append(n)
        for typ, ns in sorted(node_types.items(), key=lambda x: x[0]):
            if is_bg:
                node_color = "#f1f5f9"
                edgecolors = "#e2e8f0"
                node_size = 520
                lw = 1.0
            else:
                node_color = type_colors.get(typ, "#94a3b8")
                edgecolors = "white"
                node_size = 2300 if typ == "Paper" else 1450
                lw = 2.0
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=ns,
                node_color=node_color,
                node_size=node_size,
                alpha=alpha,
                linewidths=lw,
                edgecolors=edgecolors,
                ax=ax,
                label=(typ if not is_bg else None),
            )

    if bg_set:
        draw_group(sorted(bg_set), alpha=0.75, is_bg=True)
    draw_group(core_nodes, alpha=0.98, is_bg=False)

    bg_edges = [(u, v) for (u, v) in G.edges() if (u, v) not in core_edges]
    core_edges_list = [e for e in G.edges() if e in core_edges]

    if bg_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=bg_edges,
            width=1.2,
            alpha=0.35,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            edge_color="#cbd5e1",
            ax=ax,
            connectionstyle="arc3,rad=0.10",
        )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=core_edges_list,
        width=2.4,
        alpha=0.8,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        edge_color="#6b7280",
        ax=ax,
        connectionstyle="arc3,rad=0.10",
    )

    cx = sum(pos[n][0] for n in core_nodes) / max(1, len(core_nodes))
    cy = sum(pos[n][1] for n in core_nodes) / max(1, len(core_nodes))
    core_pos = [pos[n] for n in core_nodes]
    for n in sorted(bg_set):
        x, y = pos[n]
        min_d = min(((x - xx) ** 2 + (y - yy) ** 2) ** 0.5 for (xx, yy) in core_pos)
        if min_d < 0.25:
            continue
        txt = _short(n)
        dx = x - cx
        dy = y - cy
        norm = (dx * dx + dy * dy) ** 0.5
        if norm > 1e-6:
            ox = dx / norm * 0.06
            oy = dy / norm * 0.06
        else:
            ox, oy = (0.02, 0.02)
        ax.text(
            x + ox,
            y + oy,
            txt,
            fontsize=5.2,
            color="#cbd5e1",
            alpha=0.35,
            ha="center",
            va="center",
            zorder=1,
        )

    core_label_style: Dict[str, Tuple[float, float, str, str]] = {
        p: (-0.05, 0.26, "center", "bottom"),
        cite_p: (0.0, 0.26, "center", "bottom"),
        m: (0.05, -0.30, "center", "top"),
        repo: (0.0, 0.30, "center", "bottom"),
        ds: (0.0, -0.30, "center", "top"),
    }
    for n in core_nodes:
        x, y = pos[n]
        dx, dy, ha, va = core_label_style.get(n, (0.0, 0.14, "center", "bottom"))
        txt = _short(n)
        typ = _etype(n)
        if typ == "Paper":
            title = all_titles.get(n, "")
            if title:
                txt = f"{_short(n)}\n{_truncate(title, 44)}"
        elif typ == "Repo":
            txt = _truncate(_short(n), 28)
        elif typ == "Method":
            txt = _truncate(_short(n), 22)
        ax.text(
            x + dx,
            y + dy,
            txt,
            fontsize=9.4,
            color="#111827",
            ha=ha,
            va=va,
            zorder=6,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#e5e7eb", "alpha": 0.92},
        )

    rel_alias = {
        "paper_proposes_method": "proposes",
        "repo_implements_method": "implements",
        "method_uses_dataset": "uses",
        "paper_cites_paper": "cites",
    }
    edge_labels = {(u, v): rel_alias.get(G[u][v].get("rel", ""), G[u][v].get("rel", "")) for u, v in core_edges_list}
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=9.2,
        font_color="#111827",
        ax=ax,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#e5e7eb", "alpha": 0.95},
        rotate=False,
    )

    handles, labels = ax.get_legend_handles_labels()
    uniq = []
    seen = set()
    for h, l in zip(handles, labels, strict=False):
        if l and l not in seen:
            seen.add(l)
            uniq.append((h, l))
    if uniq:
        ax.legend([h for h, _ in uniq], [l for _, l in uniq], loc="upper left", frameon=True, framealpha=0.95, fontsize=10)

    note = f"picked: {p} | {m} | {repo} | {ds} | cites {cite_p} | bg_nodes={len(bg_set)}"
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=9, color="#374151", va="bottom")

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
