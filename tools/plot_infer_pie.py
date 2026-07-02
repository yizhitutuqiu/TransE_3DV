from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_tail_result(rows: List[Dict[str, Any]], *, h: str, r: str) -> Optional[Dict[str, Any]]:
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


def _topk_edges(item: Dict[str, Any], *, k: int) -> List[Tuple[str, float]]:
    topk = item.get("topk") or []
    edges: List[Tuple[str, float]] = []
    for x in topk:
        if not isinstance(x, dict):
            continue
        t = x.get("t")
        s = x.get("score")
        if isinstance(t, str) and isinstance(s, (int, float)):
            edges.append((t, float(s)))
    edges.sort(key=lambda x: x[1])
    return edges[: max(1, int(k))]


def _label(name: str) -> str:
    s = str(name or "").strip()
    for p in ("Dataset:", "Method:", "Repo:", "Paper:"):
        if s.startswith(p):
            return s[len(p) :]
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_path", type=str, required=True)
    ap.add_argument("--h", type=str, default="Method:nerf")
    ap.add_argument("--r", type=str, default="method_uses_dataset")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out_path", type=str, default="")
    args = ap.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    results_path = Path(args.results_path).resolve()
    out_path = Path(args.out_path).resolve() if args.out_path else (results_path.parent / "nerf_method_uses_dataset_top8_pie.png")

    data = _read_json(results_path)
    if not isinstance(data, list):
        raise SystemExit("results.json must be a list")

    item = _find_tail_result(data, h=str(args.h), r=str(args.r))
    if item is None:
        raise SystemExit(f"no tail result found for h={args.h} r={args.r}")

    edges = _topk_edges(item, k=int(args.k))
    labels = [_label(t) for t, _ in edges]
    scores = [float(s) for _, s in edges]
    max_s = max(scores) if scores else 1.0
    weights = [(max_s - s) + 1e-6 for s in scores]

    fig = plt.figure(figsize=(8.6, 6.2), dpi=160)
    ax = fig.add_subplot(1, 1, 1)
    ax.pie(weights, labels=labels, autopct="%1.1f%%", startangle=90, pctdistance=0.75, labeldistance=1.05)
    ax.axis("equal")
    ax.set_title(f"{args.h} —[{args.r}]→ top{len(edges)} (slice ~ better score)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

