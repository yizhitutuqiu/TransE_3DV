from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_tail_result(
    rows: List[Dict[str, Any]],
    *,
    h: str,
    r: str,
) -> Optional[Dict[str, Any]]:
    for it in rows:
        if not isinstance(it, dict):
            continue
        if it.get("type") != "tail":
            continue
        if str(it.get("h") or "") != h:
            continue
        if str(it.get("r") or "") != r:
            continue
        topk = it.get("topk")
        if isinstance(topk, list):
            return it
    return None


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _write_html(
    out_path: Path,
    *,
    title: str,
    center_label: str,
    edges: List[Tuple[str, float]],
    radius: float,
) -> None:
    labels = [t for t, _ in edges]
    scores = [float(s) for _, s in edges]
    n = max(1, len(labels))

    min_s = min(scores) if scores else 0.0
    max_s = max(scores) if scores else 1.0

    def norm(s: float) -> float:
        if max_s <= min_s:
            return 0.5
        return (float(s) - min_s) / (max_s - min_s)

    nodes: List[Dict[str, Any]] = []
    nodes.append({"id": center_label, "x": 0.0, "y": 0.0, "type": "Method", "score": None})

    radius = float(radius)
    if radius <= 0:
        radius = 1.2
    for i, (lab, sc) in enumerate(edges):
        ang = (2.0 * math.pi * i) / n
        x = radius * math.cos(ang)
        y = radius * math.sin(ang)
        nodes.append({"id": lab, "x": x, "y": y, "type": "Dataset", "score": float(sc)})

    edge_segments_x: List[float] = []
    edge_segments_y: List[float] = []
    edge_hover: List[str] = []
    for lab, sc in edges:
        edge_segments_x.extend([0.0, next(x["x"] for x in nodes if x["id"] == lab), None])
        edge_segments_y.extend([0.0, next(x["y"] for x in nodes if x["id"] == lab), None])
        edge_hover.append(f"{center_label} → {lab}<br>score={float(sc):.4f}")

    node_x = [x["x"] for x in nodes]
    node_y = [x["y"] for x in nodes]
    node_text = [x["id"] for x in nodes]
    node_type = [x["type"] for x in nodes]

    sizes: List[float] = []
    colors: List[str] = []
    for t, sc in zip(node_type, [x["score"] for x in nodes], strict=False):
        if t == "Method":
            sizes.append(38.0)
            colors.append("#2563eb")
        else:
            s = float(sc) if isinstance(sc, (int, float)) else max_s
            nn = 1.0 - norm(s)
            sizes.append(_clamp(14.0 + nn * 18.0, 14.0, 34.0))
            colors.append("#10b981")

    lim = max(1.0, radius * 1.35)
    payload = {
        "title": title,
        "center": center_label,
        "min_score": min_s,
        "max_score": max_s,
        "lim": lim,
        "nodes": {"x": node_x, "y": node_y, "text": node_text, "size": sizes, "color": colors},
        "edges": {"x": edge_segments_x, "y": edge_segments_y},
    }

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, "Noto Sans", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif; margin: 16px; }}
    .meta {{ color: #6b7280; font-size: 12px; margin: 6px 0 12px 0; }}
    #plot {{ width: 100%; height: 760px; border: 1px solid #e5e7eb; border-radius: 10px; }}
  </style>
</head>
<body>
  <h2 style="margin:0;">{title}</h2>
  <div class="meta">center={center_label} · edges={len(edges)} · score range=[{min_s:.4f}, {max_s:.4f}] · score 越小越可信</div>
  <div id="plot"></div>
  <script>
    const p = {json.dumps(payload, ensure_ascii=False)};
    const edgeTrace = {{
      x: p.edges.x,
      y: p.edges.y,
      mode: "lines",
      line: {{ width: 2, color: "rgba(107,114,128,0.55)" }},
      hoverinfo: "skip",
      showlegend: false
    }};
    const nodeTrace = {{
      x: p.nodes.x,
      y: p.nodes.y,
      mode: "markers+text",
      text: p.nodes.text,
      textposition: "top center",
      textfont: {{ size: 12, color: "#111827" }},
      marker: {{
        size: p.nodes.size,
        color: p.nodes.color,
        line: {{ width: 2, color: "white" }}
      }},
      hovertemplate: "%{{text}}<extra></extra>",
      showlegend: false
    }};
    const layout = {{
      margin: {{ l: 20, r: 20, t: 20, b: 20 }},
      xaxis: {{ visible: false, range: [-p.lim, p.lim] }},
      yaxis: {{ visible: false, range: [-p.lim, p.lim], scaleanchor: "x", scaleratio: 1 }},
    }};
    Plotly.newPlot("plot", [edgeTrace, nodeTrace], layout, {{displayModeBar: true}});
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_path", type=str, required=True)
    ap.add_argument("--h", type=str, default="Method:nerf")
    ap.add_argument("--r", type=str, default="method_uses_dataset")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--radius", type=float, default=1.2)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--out_name", type=str, default="nerf_method_uses_dataset_graph.html")
    args = ap.parse_args()

    results_path = Path(args.results_path).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name

    data = _read_json(results_path)
    if not isinstance(data, list):
        raise SystemExit("results.json must be a list")

    item = _find_tail_result(data, h=str(args.h), r=str(args.r))
    if item is None:
        raise SystemExit(f"no tail result found for h={args.h} r={args.r}")

    topk = item.get("topk") or []
    edges: List[Tuple[str, float]] = []
    for x in topk[: max(1, int(args.k))]:
        if not isinstance(x, dict):
            continue
        t = x.get("t")
        s = x.get("score")
        if isinstance(t, str) and isinstance(s, (int, float)):
            edges.append((t, float(s)))
    if not edges:
        raise SystemExit("no edges to plot")

    title = f"{args.h} —[{args.r}]→ (top{len(edges)})"
    _write_html(out_path, title=title, center_label=str(args.h), edges=edges, radius=float(args.radius))
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
