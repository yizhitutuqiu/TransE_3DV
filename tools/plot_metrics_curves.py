from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_metrics_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
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
                out.append(obj)
    return out


def _extract_series(rows: List[Dict[str, Any]]) -> Tuple[List[int], List[float], List[float]]:
    epochs: List[int] = []
    loss: List[float] = []
    mean_rank: List[float] = []
    for r in rows:
        ep = r.get("epoch")
        lo = r.get("avg_loss")
        mr = r.get("mean_rank")
        if not isinstance(ep, int):
            continue
        if not isinstance(lo, (int, float)):
            continue
        if not isinstance(mr, (int, float)):
            continue
        epochs.append(int(ep))
        loss.append(float(lo))
        mean_rank.append(float(mr))
    return epochs, loss, mean_rank


def _write_html(out_path: Path, *, title: str, epochs: List[int], loss: List[float], mean_rank: List[float]) -> None:
    payload = {
        "epochs": epochs,
        "loss": loss,
        "mean_rank": mean_rank,
    }
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, "Noto Sans", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif; margin: 16px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
    @media (min-width: 1100px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; }}
    .title {{ font-size: 14px; color: #111827; margin: 0 0 10px 0; }}
    .meta {{ font-size: 12px; color: #6b7280; margin: 0 0 12px 0; }}
    .plot {{ width: 100%; height: 420px; }}
  </style>
</head>
<body>
  <h2 style="margin: 0 0 6px 0;">{title}</h2>
  <p class="meta">Generated from metrics.jsonl (epochs={len(epochs)})</p>
  <div class="grid">
    <div class="card">
      <p class="title">avg_loss vs epoch</p>
      <div id="plot_loss" class="plot"></div>
    </div>
    <div class="card">
      <p class="title">mean_rank vs epoch</p>
      <div id="plot_mr" class="plot"></div>
    </div>
  </div>
  <script>
    const data = {json.dumps(payload, ensure_ascii=False)};
    Plotly.newPlot("plot_loss", [{{
      x: data.epochs,
      y: data.loss,
      type: "scatter",
      mode: "lines+markers",
      name: "avg_loss",
      line: {{ width: 2 }},
      marker: {{ size: 6 }}
    }}], {{
      margin: {{ l: 50, r: 20, t: 10, b: 40 }},
      xaxis: {{ title: "epoch" }},
      yaxis: {{ title: "avg_loss" }}
    }}, {{displayModeBar: true}});

    Plotly.newPlot("plot_mr", [{{
      x: data.epochs,
      y: data.mean_rank,
      type: "scatter",
      mode: "lines+markers",
      name: "mean_rank",
      line: {{ width: 2 }},
      marker: {{ size: 6 }}
    }}], {{
      margin: {{ l: 60, r: 20, t: 10, b: 40 }},
      xaxis: {{ title: "epoch" }},
      yaxis: {{ title: "mean_rank" }}
    }}, {{displayModeBar: true}});
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--out_name", type=str, default="metrics_curves.html")
    ap.add_argument("--title", type=str, default="Training Curves")
    args = ap.parse_args()

    metrics_path = Path(args.metrics_path).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else metrics_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name

    rows = _read_metrics_jsonl(metrics_path)
    epochs, loss, mean_rank = _extract_series(rows)
    if not epochs:
        raise SystemExit(f"no usable rows found in {metrics_path}")

    _write_html(out_path, title=args.title, epochs=epochs, loss=loss, mean_rank=mean_rank)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

