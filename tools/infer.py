from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT.parent))

from zstp_final.utils.data import build_filter_index, build_ids_by_type, relation_type_constraint, load_id_map, load_triples_hrt
from zstp_final.utils.transe import TransE


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError("infer.yaml must be a mapping")
    return obj


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _resolve_id(val: Any, mapping: Dict[str, int], kind: str) -> int:
    if isinstance(val, int):
        return int(val)
    if isinstance(val, str) and val.isdigit():
        return int(val)
    if isinstance(val, str):
        if val in mapping:
            return int(mapping[val])
    raise KeyError(f"unknown {kind}: {val}")


def _score_all_tails(
    model: TransE,
    *,
    h_id: int,
    r_id: int,
    num_entities: int,
    device: torch.device,
    batch_size: int,
    mask_t: Optional[torch.Tensor],
) -> torch.Tensor:
    model.eval()
    h = torch.tensor([h_id], device=device, dtype=torch.long)
    r = torch.tensor([r_id], device=device, dtype=torch.long)
    ent_ids = torch.arange(num_entities, device=device, dtype=torch.long)
    scores: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, num_entities, batch_size):
            cand = ent_ids[start : start + batch_size]
            sc = model.score(h.expand_as(cand), r.expand_as(cand), cand)
            scores.append(sc)
    out = torch.cat(scores, dim=0)
    if mask_t is not None:
        out = out.masked_fill(mask_t, float("inf"))
    return out


def _score_all_heads(
    model: TransE,
    *,
    t_id: int,
    r_id: int,
    num_entities: int,
    device: torch.device,
    batch_size: int,
    mask_h: Optional[torch.Tensor],
) -> torch.Tensor:
    model.eval()
    t = torch.tensor([t_id], device=device, dtype=torch.long)
    r = torch.tensor([r_id], device=device, dtype=torch.long)
    ent_ids = torch.arange(num_entities, device=device, dtype=torch.long)
    scores: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, num_entities, batch_size):
            cand = ent_ids[start : start + batch_size]
            sc = model.score(cand, r.expand_as(cand), t.expand_as(cand))
            scores.append(sc)
    out = torch.cat(scores, dim=0)
    if mask_h is not None:
        out = out.masked_fill(mask_h, float("inf"))
    return out


def _topk(scores: torch.Tensor, *, k: int) -> List[Tuple[int, float]]:
    k = min(int(k), int(scores.numel()))
    vals, idx = torch.topk(scores, k=k, largest=False)
    out: List[Tuple[int, float]] = []
    for i in range(k):
        out.append((int(idx[i].item()), float(vals[i].item())))
    return out


def _neighbors(model: TransE, *, ent_id: int, num_entities: int, k: int, device: torch.device) -> List[Tuple[int, float]]:
    model.eval()
    with torch.no_grad():
        emb = model.ent.weight.data.to(device)
        q = emb[ent_id : ent_id + 1]
        d = torch.linalg.vector_norm(emb - q, ord=2, dim=-1)
        d[ent_id] = float("inf")
        return _topk(d, k=k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(Path(__file__).with_suffix(".yaml")))
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config).resolve())
    model_cfg = cfg.get("model", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    infer_cfg = cfg.get("infer", {}) or {}
    output_cfg = cfg.get("output", {}) or {}

    ckpt_path = Path(model_cfg.get("ckpt_path", "")).expanduser().resolve()
    data_dir_s = str(data_cfg.get("data_dir", ""))
    data_dir = Path(data_dir_s).expanduser().resolve() if data_dir_s and data_dir_s != "auto" else None
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    device_s = str(model_cfg.get("device", "auto"))
    if device_s == "auto":
        device_s = "cuda" if torch.cuda.is_available() else "cpu"
    if device_s.startswith("cuda"):
        try:
            _ = torch.tensor([0], device=torch.device(device_s))
        except Exception:
            device_s = "cpu"
    device = torch.device(device_s)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    if data_dir is None:
        ckpt_args = ckpt.get("args", {}) or {}
        data_dir = Path(str(ckpt_args.get("data_dir", ""))).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(str(data_dir))

    ent2id, ent_names = load_id_map(str(data_dir / "entity2id.txt"))
    rel2id, rel_names = load_id_map(str(data_dir / "relation2id.txt"))
    num_entities = len(ent2id)
    num_relations_base = len(rel2id)

    st = ckpt.get("model_state", {}) or {}
    ent_w = st.get("ent.weight", None)
    rel_w = st.get("rel.weight", None)
    if hasattr(ent_w, "shape") and int(ent_w.shape[0]) != num_entities:
        raise ValueError(f"entity2id size {num_entities} != ckpt ent.size {int(ent_w.shape[0])}")
    ckpt_rel_n = int(rel_w.shape[0]) if hasattr(rel_w, "shape") and len(rel_w.shape) == 2 else 0
    if ckpt_rel_n not in {num_relations_base, num_relations_base * 2}:
        raise ValueError(f"relation2id size {num_relations_base} not compatible with ckpt rel.size {ckpt_rel_n}")
    num_relations = ckpt_rel_n or num_relations_base

    inferred_dim = int(ent_w.shape[1]) if hasattr(ent_w, "shape") and len(ent_w.shape) == 2 else 0
    ckpt_args = ckpt.get("args", {}) or {}
    inferred_dim = inferred_dim or int(ckpt_args.get("embedding_dim") or 0)
    inferred_p = int(ckpt_args.get("p_norm") or 0)

    embedding_dim_cfg = str(model_cfg.get("embedding_dim", "auto")).strip().lower()
    p_norm_cfg = str(model_cfg.get("p_norm", "auto")).strip().lower()
    embedding_dim = inferred_dim or (0 if embedding_dim_cfg == "auto" else int(embedding_dim_cfg or 0)) or 100
    p_norm = inferred_p or (0 if p_norm_cfg == "auto" else int(p_norm_cfg or 0)) or 1

    if num_relations == num_relations_base * 2:
        rel_names_base = list(rel_names)
        rel_names = rel_names_base + [f"{x}__inv" for x in rel_names_base]
        rel2id = dict(rel2id)
        for i, base in enumerate(rel_names_base):
            rel2id[f"{base}__inv"] = i + num_relations_base

    model = TransE(num_entities=num_entities, num_relations=num_relations, embedding_dim=embedding_dim, p_norm=p_norm).to(device)
    model.load_state_dict(ckpt["model_state"])

    top_k = int(infer_cfg.get("top_k", 10))
    batch_size = int(infer_cfg.get("batch_size", 512))
    filtered = bool(infer_cfg.get("filtered", True))

    train_triples = load_triples_hrt(str(data_dir / "train2id.txt")) if (data_dir / "train2id.txt").exists() else []
    test_triples = load_triples_hrt(str(data_dir / "test2id.txt")) if (data_dir / "test2id.txt").exists() else []
    filter_index = build_filter_index(list(train_triples) + list(test_triples))

    queries = cfg.get("queries", []) or []
    if not isinstance(queries, list):
        raise ValueError("queries must be a list")

    out_root = Path(output_cfg.get("out_root", str(_PROJECT_ROOT / "output" / "infer"))).expanduser().resolve()
    run_dir = out_root / _ts()
    _ensure_dir(run_dir)

    results: List[Dict[str, Any]] = []
    ids_by_type = build_ids_by_type(ent_names)

    for q in queries:
        if not isinstance(q, dict):
            continue
        qtype = str(q.get("type", "")).strip().lower()
        if qtype == "tail":
            h_id = _resolve_id(q.get("h"), ent2id, "entity")
            r_id = _resolve_id(q.get("r"), rel2id, "relation")
            mask_t = None
            cand_type = q.get("candidate_type")
            if cand_type is None:
                c = relation_type_constraint(rel_names[r_id])
                cand_type = c[1] if c is not None else None
            if cand_type:
                allowed = set(ids_by_type.get(str(cand_type), []))
                type_mask = torch.ones(num_entities, device=device, dtype=torch.bool)
                if allowed:
                    type_mask[list(allowed)] = False
                mask_t = type_mask
            if filtered:
                filt_mask = torch.zeros(num_entities, device=device, dtype=torch.bool)
                for tt in filter_index.tails_by_hr.get((h_id, r_id), set()):
                    filt_mask[tt] = True
                t_keep = q.get("t")
                if t_keep is not None:
                    t_id = _resolve_id(t_keep, ent2id, "entity")
                    filt_mask[t_id] = False
                mask_t = filt_mask if mask_t is None else (mask_t | filt_mask)
            scores = _score_all_tails(
                model, h_id=h_id, r_id=r_id, num_entities=num_entities, device=device, batch_size=batch_size, mask_t=mask_t
            )
            top = _topk(scores, k=int(q.get("k", top_k)))
            results.append(
                {
                    "type": "tail",
                    "h": ent_names[h_id],
                    "r": rel_names[r_id],
                    "topk": [{"t": ent_names[i], "score": s} for i, s in top],
                }
            )
        elif qtype == "head":
            t_id = _resolve_id(q.get("t"), ent2id, "entity")
            r_id = _resolve_id(q.get("r"), rel2id, "relation")
            mask_h = None
            cand_type = q.get("candidate_type")
            if cand_type is None:
                c = relation_type_constraint(rel_names[r_id])
                cand_type = c[0] if c is not None else None
            if cand_type:
                allowed = set(ids_by_type.get(str(cand_type), []))
                type_mask = torch.ones(num_entities, device=device, dtype=torch.bool)
                if allowed:
                    type_mask[list(allowed)] = False
                mask_h = type_mask
            if filtered:
                filt_mask = torch.zeros(num_entities, device=device, dtype=torch.bool)
                for hh in filter_index.heads_by_rt.get((r_id, t_id), set()):
                    filt_mask[hh] = True
                h_keep = q.get("h")
                if h_keep is not None:
                    h_id = _resolve_id(h_keep, ent2id, "entity")
                    filt_mask[h_id] = False
                mask_h = filt_mask if mask_h is None else (mask_h | filt_mask)
            scores = _score_all_heads(
                model, t_id=t_id, r_id=r_id, num_entities=num_entities, device=device, batch_size=batch_size, mask_h=mask_h
            )
            top = _topk(scores, k=int(q.get("k", top_k)))
            results.append(
                {
                    "type": "head",
                    "t": ent_names[t_id],
                    "r": rel_names[r_id],
                    "topk": [{"h": ent_names[i], "score": s} for i, s in top],
                }
            )
        elif qtype == "score":
            h_id = _resolve_id(q.get("h"), ent2id, "entity")
            r_id = _resolve_id(q.get("r"), rel2id, "relation")
            t_id = _resolve_id(q.get("t"), ent2id, "entity")
            with torch.no_grad():
                sc = float(model.score(torch.tensor([h_id], device=device), torch.tensor([r_id], device=device), torch.tensor([t_id], device=device)).item())
            results.append({"type": "score", "h": ent_names[h_id], "r": rel_names[r_id], "t": ent_names[t_id], "score": sc})
        elif qtype == "neighbors":
            e_id = _resolve_id(q.get("entity"), ent2id, "entity")
            k = int(q.get("k", top_k))
            nb = _neighbors(model, ent_id=e_id, num_entities=num_entities, k=k, device=device)
            results.append({"type": "neighbors", "entity": ent_names[e_id], "topk": [{"entity": ent_names[i], "distance": s} for i, s in nb]})

    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_path": str(Path(args.config).resolve()),
        "ckpt_path": str(ckpt_path),
        "data_dir": str(data_dir),
        "num_entities": num_entities,
        "num_relations": num_relations,
        "device": str(device),
        "embedding_dim": embedding_dim,
        "p_norm": p_norm,
    }

    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
