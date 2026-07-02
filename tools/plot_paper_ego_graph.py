from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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


def _load_edges(triple_paths: List[Path], *, center: str) -> List[Tuple[str, str, str]]:
    edges: List[Tuple[str, str, str]] = []
    for p in triple_paths:
        for it in _iter_jsonl(p):
            h = it.get("h")
            r = it.get("r")
            t = it.get("t")
            if not isinstance(h, str) or not isinstance(r, str) or not isinstance(t, str):
                continue
            if h != center and t != center:
                continue
            edges.append((h, r, t))
    return edges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper_id", type=str, default="1812.03828")
    ap.add_argument("--triples_train", type=str, default="data/preprocessed/final/triples_train.jsonl")
    ap.add_argument("--triples_test", type=str, default="data/preprocessed/final/triples_test.jsonl")
    ap.add_argument("--out_dir", type=str, default="output/vis")
    ap.add_argument("--out_name", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    import networkx as nx

    root = Path(__file__).resolve().parents[1]
    train_path = (root / str(args.triples_train)).resolve() if not Path(str(args.triples_train)).is_absolute() else Path(str(args.triples_train)).resolve()
    test_path = (root / str(args.triples_test)).resolve() if not Path(str(args.triples_test)).is_absolute() else Path(str(args.triples_test)).resolve()

    center = f"Paper:{str(args.paper_id).strip()}"
    edges = _load_edges([train_path, test_path], center=center)
    if not edges:
        raise SystemExit(f"no 1-hop triples found for {center} in {train_path} / {test_path}")

    G = nx.DiGraph()
    for h, r, t in edges:
        G.add_node(h, etype=_etype(h))
        G.add_node(t, etype=_etype(t))

        if G.has_edge(h, t):
            rs = G[h][t].get("rels", [])
            if r not in rs:
                rs.append(r)
            G[h][t]["rels"] = rs
        else:
            G.add_edge(h, t, rels=[r])

    pos_init: Dict[str, Tuple[float, float]] = {center: (0.0, 0.0)}
    pos = nx.spring_layout(G, seed=int(args.seed), pos=pos_init, fixed=[center])

    type_colors = {
        "Paper": "#ef4444",
        "Method": "#2563eb",
        "Dataset": "#10b981",
        "Repo": "#f59e0b",
        "Task": "#8b5cf6",
        "Venue": "#06b6d4",
        "Metric": "#64748b",
        "": "#94a3b8",
    }

    node_types: Dict[str, List[str]] = {}
    for n in G.nodes():
        node_types.setdefault(G.nodes[n].get("etype", ""), []).append(n)

    fig = plt.figure(figsize=(15, 10), dpi=180)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f"Ego Graph (1-hop): {center}", fontsize=14)
    ax.axis("off")

    for typ, nodes in sorted(node_types.items(), key=lambda x: x[0]):
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=type_colors.get(typ, "#94a3b8"),
            node_size=1700 if typ == "Paper" else 900,
            alpha=0.95,
            linewidths=1.8,
            edgecolors="white",
            ax=ax,
            label=typ or "Other",
        )

    nx.draw_networkx_edges(
        G,
        pos,
        width=1.8,
        alpha=0.7,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        edge_color="#6b7280",
        ax=ax,
        connectionstyle="arc3,rad=0.08",
    )

    labels = {n: _short(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8.5, font_color="#111827", ax=ax)

    edge_labels = {(u, v): "\\n".join(G[u][v].get("rels", [])) for u, v in G.edges()}
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7.8,
        font_color="#111827",
        ax=ax,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#e5e7eb", "alpha": 0.9},
        rotate=False,
    )

    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9)

    out_dir = (root / str(args.out_dir)).resolve() if not Path(str(args.out_dir)).is_absolute() else Path(str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = str(args.out_name).strip() or f"paper_{args.paper_id}_ego.png"
    out_path = out_dir / out_name
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

