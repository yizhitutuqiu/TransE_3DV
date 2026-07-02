from __future__ import annotations

import argparse
import json
import math
import os
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


def _html_template(payload_json: str, title: str) -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <link rel="stylesheet" href="https://unpkg.com/vis-network/styles/vis-network.min.css">
  <style>
    html, body {{ width: 100%; height: 100%; margin: 0; padding: 0; background: #ffffff; }}
    #wrap {{ width: 100%; height: 100%; display: flex; flex-direction: column; }}
    #bar {{ padding: 10px 14px; border-bottom: 1px solid #e5e7eb; font-family: ui-sans-serif, system-ui; font-size: 14px; color: #111827; }}
    #net {{ flex: 1; }}
    .kv {{ color: #374151; }}
  </style>
</head>
<body>
  <div id="wrap">
    <div id="bar">
      <span>{title}</span>
      <span class="kv" style="margin-left:14px;">Drag nodes to adjust layout. Scroll to zoom.</span>
    </div>
    <div id="net"></div>
  </div>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <script>
    const payload = {payload_json};
    const nodes = new vis.DataSet(payload.nodes);
    const edges = new vis.DataSet(payload.edges);
    const container = document.getElementById('net');
    const data = {{ nodes, edges }};
    const options = {{
      autoResize: true,
      interaction: {{
        dragNodes: true,
        dragView: true,
        zoomView: true,
        hover: true,
        multiselect: true
      }},
      physics: {{
        enabled: false
      }},
      nodes: {{
        shape: 'dot'
      }},
      edges: {{
        smooth: {{
          enabled: true,
          type: 'continuous'
        }},
        arrows: {{
          to: {{ enabled: true, scaleFactor: 0.8 }}
        }},
        font: {{
          size: 12,
          strokeWidth: 3,
          strokeColor: '#ffffff'
        }}
      }}
    }};
    const network = new vis.Network(container, data, options);
    network.once('afterDrawing', () => {{
      network.fit({{ animation: false, padding: 30 }});
    }});
  </script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefer_paper", type=str, default="1812.03828")
    ap.add_argument("--triples_train", type=str, default="data/preprocessed/final/triples_train.jsonl")
    ap.add_argument("--triples_test", type=str, default="data/preprocessed/final/triples_test.jsonl")
    ap.add_argument("--bg_nodes", type=int, default=25)
    ap.add_argument("--core_degree_cap", type=int, default=5)
    ap.add_argument("--bg_degree_cap", type=int, default=5)
    ap.add_argument("--core_scale", type=float, default=1.8)
    ap.add_argument("--spring_k", type=float, default=0.0)
    ap.add_argument("--spring_iter", type=int, default=260)
    ap.add_argument("--bg_ring_radius", type=float, default=0.0)
    ap.add_argument("--bg_jitter", type=float, default=0.12)
    ap.add_argument("--pos_scale_px", type=float, default=220.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="output/vis")
    ap.add_argument("--out_name", type=str, default="compact_4x4_min_bg.html")
    args = ap.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import networkx as nx

    root = Path(__file__).resolve().parents[1]
    train_path = (root / str(args.triples_train)).resolve() if not Path(str(args.triples_train)).is_absolute() else Path(str(args.triples_train)).resolve()
    test_path = (root / str(args.triples_test)).resolve() if not Path(str(args.triples_test)).is_absolute() else Path(str(args.triples_test)).resolve()
    triples = _load_triples([train_path, test_path])

    prefer = str(args.prefer_paper).strip()
    prefer_name = f"Paper:{prefer}" if prefer and not prefer.startswith("Paper:") else prefer
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
    core_cap = int(args.core_degree_cap)
    bg_cap = int(args.bg_degree_cap)
    for _, d in list(core_deg.items()):
        if d > core_cap:
            core_cap = d

    bg_nodes, bg_edges = _bfs_pick_bg(
        adj,
        core_nodes=core_nodes,
        seeds=core_nodes_list,
        bg_target=max(0, int(args.bg_nodes)),
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

    s = float(args.core_scale)
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

    rng = random.Random(int(args.seed))
    pos_init = dict(pos_core)
    bg_only = [n for n in G.nodes() if n not in pos_init]
    bg_only.sort()
    ring = float(args.bg_ring_radius)
    if ring <= 0:
        ring = 2.9 * s
    jitter = float(args.bg_jitter)
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
    k = float(args.spring_k)
    if k <= 0:
        k = max(0.55, (2.2 * s) / (n_total**0.5))
    iters = max(80, int(args.spring_iter))
    scale = max(2.6, ring * 1.15)
    pos = nx.spring_layout(
        G.to_undirected(),
        seed=int(args.seed),
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
    bg_node_fill = "#9ca3af"
    bg_node_border = "#6b7280"

    nodes: List[Dict[str, Any]] = []
    px = float(args.pos_scale_px)
    if px <= 0:
        px = 220.0
    for n in G.nodes():
        is_core = bool(G.nodes[n].get("is_core"))
        typ = str(G.nodes[n].get("etype") or "")
        x, y = pos.get(n, (0.0, 0.0))
        if is_core:
            color = type_colors.get(typ, "#94a3b8")
            size = 34 if typ == "Paper" else 26
            label = _short(n)
            fixed_xy = {"x": True, "y": True}
            font = {"size": 18, "color": "#111827", "strokeWidth": 4, "strokeColor": "#ffffff"}
            border = "#ffffff"
        else:
            color = bg_node_fill
            size = 18
            label = ""
            fixed_xy = False
            font = {"size": 14, "color": "#6b7280", "strokeWidth": 4, "strokeColor": "#ffffff"}
            border = bg_node_border
        nodes.append(
            {
                "id": n,
                "label": label,
                "title": n,
                "x": float(x) * px,
                "y": float(y) * px,
                "fixed": fixed_xy,
                "color": {"background": color, "border": border},
                "size": size,
                "font": font,
            }
        )

    edges: List[Dict[str, Any]] = []
    for u, v in G.edges():
        rel = str(G[u][v].get("rel") or "")
        is_core = bool(G[u][v].get("is_core"))
        if is_core:
            color = "#374151"
            width = 3
            opacity = 0.85
        else:
            color = "#9ca3af"
            width = 1
            opacity = 0.45
        edges.append(
            {
                "from": u,
                "to": v,
                "label": rel if is_core else "",
                "arrows": "to",
                "color": {"color": color, "opacity": opacity},
                "width": width,
                "smooth": {"enabled": True, "type": "continuous"},
            }
        )

    payload = {"nodes": nodes, "edges": edges}
    title = f"Compact 4x4 + BG ({len(bg_nodes)} nodes)"
    html = _html_template(json.dumps(payload, ensure_ascii=False), title=title)

    out_dir = (root / str(args.out_dir)).resolve() if not Path(str(args.out_dir)).is_absolute() else Path(str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / str(args.out_name)
    out_path.write_text(html, encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

