from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT.parent))

from zstp_final.train.evaluate import evaluate_filtered
from zstp_final.utils.data import build_filter_index, load_id_map, load_triples_hrt
from zstp_final.utils.transe import TransE


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError("api.yaml must be a mapping")
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


def _resolve_id(val: Any, mapping: Dict[str, int], kind: str) -> int:
    if isinstance(val, int):
        return int(val)
    if isinstance(val, str) and val.isdigit():
        return int(val)
    if isinstance(val, str) and val in mapping:
        return int(mapping[val])
    raise KeyError(f"unknown {kind}: {val}")


def _topk(scores: torch.Tensor, *, k: int) -> List[Tuple[int, float]]:
    k = min(int(k), int(scores.numel()))
    vals, idx = torch.topk(scores, k=k, largest=False)
    out: List[Tuple[int, float]] = []
    for i in range(k):
        out.append((int(idx[i].item()), float(vals[i].item())))
    return out


@dataclass
class ModelInfo:
    ckpt_path: str
    data_dir: str
    device: str
    embedding_dim: int
    p_norm: int
    num_entities: int
    num_relations: int


@dataclass
class ZSTPAPI:
    model: TransE
    info: ModelInfo
    ent2id: Dict[str, int]
    rel2id: Dict[str, int]
    ent_names: List[str]
    rel_names: List[str]
    ids_by_type: Dict[str, List[int]]
    filter_index: Any
    train_triples: List[Tuple[int, int, int]]
    test_triples: List[Tuple[int, int, int]]

    @classmethod
    def from_config(cls, config_path: str | Path = "api.yaml") -> "ZSTPAPI":
        cfg_path = _resolve_path(config_path, base=_PROJECT_ROOT)
        cfg = _load_yaml(cfg_path)
        model_cfg = cfg.get("model", {}) or {}
        infer_cfg = cfg.get("infer", {}) or {}
        eval_cfg = cfg.get("eval", {}) or {}

        ckpt_path = _resolve_path(model_cfg.get("ckpt_path", "checkpoints/transe_v1/best.pt"), base=_PROJECT_ROOT)
        data_dir = _resolve_path(model_cfg.get("data_dir", "data/preprocessed/final"), base=_PROJECT_ROOT)
        if not ckpt_path.exists():
            raise FileNotFoundError(str(ckpt_path))
        if not data_dir.exists():
            raise FileNotFoundError(str(data_dir))

        device_s = str(model_cfg.get("device", "auto")).strip() or "auto"
        if device_s == "auto":
            device_s = "cuda" if torch.cuda.is_available() else "cpu"
        if device_s.startswith("cuda"):
            try:
                _ = torch.tensor([0], device=torch.device(device_s))
            except Exception:
                device_s = "cpu"
        device = torch.device(device_s)

        embedding_dim = int(model_cfg.get("embedding_dim", 100))
        p_norm = int(model_cfg.get("p_norm", 1))

        ckpt = torch.load(str(ckpt_path), map_location=device)

        ent2id, ent_names = load_id_map(str(data_dir / "entity2id.txt"))
        rel2id, rel_names = load_id_map(str(data_dir / "relation2id.txt"))
        num_entities = len(ent2id)
        num_relations = len(rel2id)

        st = ckpt.get("model_state", {}) or {}
        ent_w = st.get("ent.weight", None)
        rel_w = st.get("rel.weight", None)
        if hasattr(ent_w, "shape") and int(ent_w.shape[0]) != num_entities:
            raise ValueError(f"entity2id size {num_entities} != ckpt ent.size {int(ent_w.shape[0])}")
        if hasattr(rel_w, "shape") and int(rel_w.shape[0]) != num_relations:
            raise ValueError(f"relation2id size {num_relations} != ckpt rel.size {int(rel_w.shape[0])}")

        model = TransE(num_entities=num_entities, num_relations=num_relations, embedding_dim=embedding_dim, p_norm=p_norm).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        ids_by_type: Dict[str, List[int]] = {}
        for i, n in enumerate(ent_names):
            et = n.split(":", 1)[0] if ":" in n else ""
            ids_by_type.setdefault(et, []).append(i)

        train_triples = load_triples_hrt(str(data_dir / "train2id.txt")) if (data_dir / "train2id.txt").exists() else []
        test_triples = load_triples_hrt(str(data_dir / "test2id.txt")) if (data_dir / "test2id.txt").exists() else []
        filter_index = build_filter_index(list(train_triples) + list(test_triples))

        info = ModelInfo(
            ckpt_path=str(ckpt_path),
            data_dir=str(data_dir),
            device=str(device),
            embedding_dim=embedding_dim,
            p_norm=p_norm,
            num_entities=num_entities,
            num_relations=num_relations,
        )

        api = cls(
            model=model,
            info=info,
            ent2id=ent2id,
            rel2id=rel2id,
            ent_names=ent_names,
            rel_names=rel_names,
            ids_by_type=ids_by_type,
            filter_index=filter_index,
            train_triples=list(train_triples),
            test_triples=list(test_triples),
        )
        api._default_top_k = int(infer_cfg.get("top_k", 10))
        api._default_batch_size = int(infer_cfg.get("batch_size", 512))
        api._default_filtered = bool(infer_cfg.get("filtered", True))
        api._default_eval_batch_size = int(eval_cfg.get("batch_size", 256))
        return api

    def model_info(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self.info.__dict__, ensure_ascii=False))

    def relations(self) -> List[str]:
        return list(self.rel_names)

    def search_entities(self, *, q: str = "", entity_type: str = "", limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        q = str(q or "").strip().lower()
        entity_type = str(entity_type or "").strip()
        limit = max(1, min(int(limit), 2000))
        offset = max(0, int(offset))

        ids = list(range(self.info.num_entities))
        if entity_type:
            ids = list(self.ids_by_type.get(entity_type, []))

        out: List[Dict[str, Any]] = []
        for i in ids:
            name = self.ent_names[i]
            if q and q not in name.lower():
                continue
            out.append({"id": i, "name": name})
        return {"count": len(out), "items": out[offset : offset + limit], "offset": offset, "limit": limit}

    def score(self, *, h: Any, r: Any, t: Any) -> Dict[str, Any]:
        h_id = _resolve_id(h, self.ent2id, "entity")
        r_id = _resolve_id(r, self.rel2id, "relation")
        t_id = _resolve_id(t, self.ent2id, "entity")
        device = torch.device(self.info.device)
        with torch.no_grad():
            sc = self.model.score(
                torch.tensor([h_id], device=device, dtype=torch.long),
                torch.tensor([r_id], device=device, dtype=torch.long),
                torch.tensor([t_id], device=device, dtype=torch.long),
            )[0]
        return {"h": self.ent_names[h_id], "r": self.rel_names[r_id], "t": self.ent_names[t_id], "score": float(sc.item())}

    def predict_tail(
        self,
        *,
        h: Any,
        r: Any,
        k: Optional[int] = None,
        candidate_type: str = "",
        filtered: Optional[bool] = None,
        batch_size: Optional[int] = None,
        keep_t: Any = None,
    ) -> Dict[str, Any]:
        h_id = _resolve_id(h, self.ent2id, "entity")
        r_id = _resolve_id(r, self.rel2id, "relation")
        device = torch.device(self.info.device)

        kk = int(self._default_top_k if k is None else k)
        bb = int(self._default_batch_size if batch_size is None else batch_size)
        ff = bool(self._default_filtered if filtered is None else filtered)
        kk = max(1, kk)
        bb = max(1, bb)

        mask = None
        candidate_type = str(candidate_type or "").strip()
        if candidate_type:
            allowed = set(self.ids_by_type.get(candidate_type, []))
            type_mask = torch.ones(self.info.num_entities, device=device, dtype=torch.bool)
            if allowed:
                type_mask[list(allowed)] = False
            mask = type_mask

        if ff:
            filt_mask = torch.zeros(self.info.num_entities, device=device, dtype=torch.bool)
            for tt in self.filter_index.tails_by_hr.get((h_id, r_id), set()):
                filt_mask[tt] = True
            if keep_t is not None:
                t_id = _resolve_id(keep_t, self.ent2id, "entity")
                filt_mask[t_id] = False
            mask = filt_mask if mask is None else (mask | filt_mask)

        ent_ids = torch.arange(self.info.num_entities, device=device, dtype=torch.long)
        scores: List[torch.Tensor] = []
        h0 = torch.tensor([h_id], device=device, dtype=torch.long)
        r0 = torch.tensor([r_id], device=device, dtype=torch.long)
        with torch.no_grad():
            for start in range(0, self.info.num_entities, bb):
                cand = ent_ids[start : start + bb]
                sc = self.model.score(h0.expand_as(cand), r0.expand_as(cand), cand)
                scores.append(sc)
        out = torch.cat(scores, dim=0)
        if mask is not None:
            out = out.masked_fill(mask, float("inf"))
        top = _topk(out, k=kk)
        return {"h": self.ent_names[h_id], "r": self.rel_names[r_id], "topk": [{"t": self.ent_names[i], "score": s} for i, s in top]}

    def predict_head(
        self,
        *,
        t: Any,
        r: Any,
        k: Optional[int] = None,
        candidate_type: str = "",
        filtered: Optional[bool] = None,
        batch_size: Optional[int] = None,
        keep_h: Any = None,
    ) -> Dict[str, Any]:
        t_id = _resolve_id(t, self.ent2id, "entity")
        r_id = _resolve_id(r, self.rel2id, "relation")
        device = torch.device(self.info.device)

        kk = int(self._default_top_k if k is None else k)
        bb = int(self._default_batch_size if batch_size is None else batch_size)
        ff = bool(self._default_filtered if filtered is None else filtered)
        kk = max(1, kk)
        bb = max(1, bb)

        mask = None
        candidate_type = str(candidate_type or "").strip()
        if candidate_type:
            allowed = set(self.ids_by_type.get(candidate_type, []))
            type_mask = torch.ones(self.info.num_entities, device=device, dtype=torch.bool)
            if allowed:
                type_mask[list(allowed)] = False
            mask = type_mask

        if ff:
            filt_mask = torch.zeros(self.info.num_entities, device=device, dtype=torch.bool)
            for hh in self.filter_index.heads_by_rt.get((r_id, t_id), set()):
                filt_mask[hh] = True
            if keep_h is not None:
                h_id = _resolve_id(keep_h, self.ent2id, "entity")
                filt_mask[h_id] = False
            mask = filt_mask if mask is None else (mask | filt_mask)

        ent_ids = torch.arange(self.info.num_entities, device=device, dtype=torch.long)
        scores: List[torch.Tensor] = []
        t0 = torch.tensor([t_id], device=device, dtype=torch.long)
        r0 = torch.tensor([r_id], device=device, dtype=torch.long)
        with torch.no_grad():
            for start in range(0, self.info.num_entities, bb):
                cand = ent_ids[start : start + bb]
                sc = self.model.score(cand, r0.expand_as(cand), t0.expand_as(cand))
                scores.append(sc)
        out = torch.cat(scores, dim=0)
        if mask is not None:
            out = out.masked_fill(mask, float("inf"))
        top = _topk(out, k=kk)
        return {"t": self.ent_names[t_id], "r": self.rel_names[r_id], "topk": [{"h": self.ent_names[i], "score": s} for i, s in top]}

    def neighbors(self, *, entity: Any, k: int = 10) -> Dict[str, Any]:
        ent_id = _resolve_id(entity, self.ent2id, "entity")
        device = torch.device(self.info.device)
        kk = max(1, int(k))
        with torch.no_grad():
            emb = self.model.ent.weight.data.to(device)
            q = emb[ent_id : ent_id + 1]
            d = torch.linalg.vector_norm(emb - q, ord=2, dim=-1)
            d[ent_id] = float("inf")
            top = _topk(d, k=kk)
        return {"entity": self.ent_names[ent_id], "topk": [{"entity": self.ent_names[i], "distance": s} for i, s in top]}

    def evaluate(self, *, split: str = "test", batch_size: Optional[int] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        bb = int(self._default_eval_batch_size if batch_size is None else batch_size)
        bb = max(1, bb)
        split_s = str(split).strip().lower()
        if split_s not in {"test", "train"}:
            split_s = "test"
        triples = self.test_triples if split_s == "test" else self.train_triples
        if limit is not None:
            triples = triples[: max(0, int(limit))]
        r = evaluate_filtered(
            self.model,
            triples,
            filter_index=self.filter_index,
            num_entities=self.info.num_entities,
            device=torch.device(self.info.device),
            batch_size=bb,
        )
        return {"split": split_s, "count": len(triples), "mrr": r.mrr, "hits1": r.hits1, "hits3": r.hits3, "hits10": r.hits10, "mean_rank": r.mean_rank}


def load_api(config_path: str | Path = "api.yaml") -> ZSTPAPI:
    return ZSTPAPI.from_config(config_path)
