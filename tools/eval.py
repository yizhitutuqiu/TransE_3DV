from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT.parent))

from zstp_final.train.diagnose_transe import main as _diagnose_main


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError("eval.yaml must be a mapping")
    return obj


def _resolve_path(p: Any, *, base: Path) -> Path:
    if p is None:
        return base
    s = str(p).strip()
    if not s:
        return base
    pp = Path(s).expanduser()
    if pp.is_absolute():
        return pp.resolve()
    return (base / pp).resolve()


def _infer_int(v: Any) -> int:
    if v is None:
        return 0
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return int(v)
    s = str(v).strip().lower()
    if not s or s == "auto":
        return 0
    return int(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(Path(__file__).with_suffix(".yaml")))
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config).resolve())
    model_cfg = cfg.get("model", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}
    output_cfg = cfg.get("output", {}) or {}

    ckpt_path = str(_resolve_path(model_cfg.get("ckpt_path", "checkpoints/transe_v1/best.pt"), base=_PROJECT_ROOT))
    data_dir_raw = str(data_cfg.get("data_dir", "auto"))
    data_dir = str(_resolve_path(data_dir_raw, base=_PROJECT_ROOT)) if data_dir_raw and str(data_dir_raw).strip().lower() != "auto" else ""
    device_cfg = str(model_cfg.get("device", "cpu"))
    device = device_cfg
    if device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        try:
            _ = torch.tensor([0], device=torch.device(device))
        except Exception:
            device = "cpu"
    embedding_dim_cfg = _infer_int(model_cfg.get("embedding_dim", "auto"))
    p_norm_cfg = _infer_int(model_cfg.get("p_norm", "auto"))
    batch_size = int(eval_cfg.get("batch_size", 512))

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not data_dir:
        ckpt_args = ckpt.get("args", {}) or {}
        cand = str(ckpt_args.get("data_dir", "")).strip()
        if cand:
            p = Path(cand).expanduser()
            if p.is_absolute() and p.exists():
                data_dir = str(p.resolve())

    if not data_dir:
        data_dir = str(_resolve_path("data/preprocessed/final", base=_PROJECT_ROOT))

    st = ckpt.get("model_state", {}) or {}
    ent_w = st.get("ent.weight", None)
    inferred_dim = int(ent_w.shape[1]) if hasattr(ent_w, "shape") and len(ent_w.shape) == 2 else 0
    ckpt_args = ckpt.get("args", {}) or {}
    inferred_dim = inferred_dim or _infer_int(ckpt_args.get("embedding_dim", 0))
    inferred_p = _infer_int(ckpt_args.get("p_norm", 0))

    embedding_dim = inferred_dim or embedding_dim_cfg or 100
    p_norm = inferred_p or p_norm_cfg or 1

    ent2id_path = Path(data_dir).resolve() / "entity2id.txt"
    rel2id_path = Path(data_dir).resolve() / "relation2id.txt"
    if ent2id_path.exists():
        from zstp_final.utils.data import load_id_map

        ent2id, _ = load_id_map(str(ent2id_path))
        rel2id, _ = load_id_map(str(rel2id_path))
        rel_w = st.get("rel.weight", None)
        if hasattr(ent_w, "shape") and int(ent_w.shape[0]) != len(ent2id):
            raise ValueError(f"entity2id size {len(ent2id)} != ckpt ent.size {int(ent_w.shape[0])}")
        if hasattr(rel_w, "shape") and len(rel_w.shape) == 2:
            ckpt_rel_n = int(rel_w.shape[0])
            if ckpt_rel_n not in {len(rel2id), len(rel2id) * 2}:
                raise ValueError(f"relation2id size {len(rel2id)} not compatible with ckpt rel.size {ckpt_rel_n}")

    out_root = _resolve_path(output_cfg.get("out_root", "output/eval"), base=_PROJECT_ROOT)
    run_dir = out_root / _ts()
    run_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "diagnose_transe.py",
        "--data_dir",
        data_dir,
        "--ckpt",
        ckpt_path,
        "--device",
        device,
        "--embedding_dim",
        str(embedding_dim),
        "--p_norm",
        str(p_norm),
        "--batch_size",
        str(batch_size),
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        from io import StringIO
        import contextlib

        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            _diagnose_main()
        report = buf.getvalue()
    finally:
        sys.argv = old_argv

    (run_dir / "report.json").write_text(report, encoding="utf-8")
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "config_path": str(Path(args.config).resolve()),
                "ckpt_path": ckpt_path,
                "data_dir": data_dir,
                "device": device,
                "embedding_dim": embedding_dim,
                "p_norm": p_norm,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
