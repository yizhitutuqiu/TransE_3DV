from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT.parent))

from zstp_final.train.evaluate import evaluate_filtered
from zstp_final.utils.data import build_filter_index, build_ids_by_type, relation_type_constraint, load_id_map, load_triples_hrt
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
    cand = (Path.cwd() / pp).resolve()
    if cand.exists():
        return cand
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
    entity_meta: Dict[str, Dict[str, Any]]
    filter_index: Any
    heads_by_r: Dict[int, List[int]]
    tails_by_r: Dict[int, List[int]]
    triple_set: Set[Tuple[int, int, int]]
    train_triples: List[Tuple[int, int, int]]
    test_triples: List[Tuple[int, int, int]]

    @classmethod
    def from_config(cls, config_path: str | Path = "api.yaml") -> "ZSTPAPI":
        cfg_path = _resolve_path(config_path, base=_PROJECT_ROOT)
        cfg = _load_yaml(cfg_path)
        model_cfg = cfg.get("model", {}) or {}
        meta_cfg = cfg.get("metadata", {}) or {}
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

        ckpt = torch.load(str(ckpt_path), map_location=device)

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
        inferred_dim = inferred_dim or _infer_int(ckpt_args.get("embedding_dim"))
        inferred_p = _infer_int(ckpt_args.get("p_norm"))

        embedding_dim_cfg = _infer_int(model_cfg.get("embedding_dim", "auto"))
        p_norm_cfg = _infer_int(model_cfg.get("p_norm", "auto"))
        embedding_dim = inferred_dim or embedding_dim_cfg or 100
        p_norm = inferred_p or p_norm_cfg or 1

        if num_relations == num_relations_base * 2:
            rel_names_base = list(rel_names)
            rel_names = rel_names_base + [f"{x}__inv" for x in rel_names_base]
            rel2id = dict(rel2id)
            for i, base in enumerate(rel_names_base):
                rel2id[f"{base}__inv"] = i + num_relations_base

        model = TransE(num_entities=num_entities, num_relations=num_relations, embedding_dim=embedding_dim, p_norm=p_norm).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        ids_by_type = build_ids_by_type(ent_names)

        entity_meta: Dict[str, Dict[str, Any]] = {}
        docs_path_raw = meta_cfg.get("documents_path", "data/preprocessed/text/documents.jsonl")
        docs_path = _resolve_path(docs_path_raw, base=_PROJECT_ROOT) if str(docs_path_raw).strip().lower() != "auto" else None
        if docs_path is not None and docs_path.exists():
            with docs_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(d, dict):
                        continue
                    meta = d.get("metadata")
                    if not isinstance(meta, dict):
                        continue
                    if d.get("doc_type") == "paper":
                        aid = meta.get("arxiv_id")
                        if isinstance(aid, str) and aid.strip():
                            k = f"Paper:{aid.strip()}"
                            entity_meta[k] = {
                                "citation_count": meta.get("citation_count"),
                                "reference_count": meta.get("reference_count"),
                                "year": meta.get("year"),
                                "title": d.get("title"),
                            }
                    if d.get("doc_type") == "readme":
                        full = meta.get("repo_full_name") or meta.get("full_name")
                        if isinstance(full, str) and full.strip():
                            k = f"Repo:{full.strip()}"
                            entity_meta[k] = {
                                "repo_stargazers_count": meta.get("repo_stargazers_count"),
                                "repo_forks_count": meta.get("repo_forks_count"),
                                "repo_open_issues_count": meta.get("repo_open_issues_count"),
                                "repo_updated_at": meta.get("repo_updated_at"),
                                "repo_created_at": meta.get("repo_created_at"),
                            }

        train_triples = load_triples_hrt(str(data_dir / "train2id.txt")) if (data_dir / "train2id.txt").exists() else []
        test_triples = load_triples_hrt(str(data_dir / "test2id.txt")) if (data_dir / "test2id.txt").exists() else []
        all_triples = list(train_triples) + list(test_triples)
        filter_index = build_filter_index(all_triples)
        heads_by_r_set: Dict[int, set] = {}
        tails_by_r_set: Dict[int, set] = {}
        for h, t, r in all_triples:
            heads_by_r_set.setdefault(int(r), set()).add(int(h))
            tails_by_r_set.setdefault(int(r), set()).add(int(t))
        heads_by_r = {k: sorted(v) for k, v in heads_by_r_set.items()}
        tails_by_r = {k: sorted(v) for k, v in tails_by_r_set.items()}
        triple_set = {(int(h), int(r), int(t)) for (h, t, r) in all_triples}

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
            entity_meta=entity_meta,
            filter_index=filter_index,
            heads_by_r=heads_by_r,
            tails_by_r=tails_by_r,
            triple_set=triple_set,
            train_triples=list(train_triples),
            test_triples=list(test_triples),
        )
        api._default_top_k = int(infer_cfg.get("top_k", 10))
        api._default_batch_size = int(infer_cfg.get("batch_size", 512))
        api._default_filtered = bool(infer_cfg.get("filtered", True))
        api._default_infer_mode = str(infer_cfg.get("mode", "hybrid")).strip().lower() or "hybrid"
        api._default_eval_batch_size = int(eval_cfg.get("batch_size", 256))
        api._default_include_entity_meta = bool(meta_cfg.get("include_in_search", True))
        return api

    def model_info(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self.info.__dict__, ensure_ascii=False))

    def relations(self) -> List[str]:
        return list(self.rel_names)

    def search_entities(
        self,
        *,
        q: str = "",
        entity_type: str = "",
        limit: int = 50,
        offset: int = 0,
        include_meta: Optional[bool] = None,
    ) -> Dict[str, Any]:
        q = str(q or "").strip().lower()
        entity_type = str(entity_type or "").strip()
        limit = max(1, min(int(limit), 2000))
        offset = max(0, int(offset))
        inc = self._default_include_entity_meta if include_meta is None else bool(include_meta)

        ids = list(range(self.info.num_entities))
        if entity_type:
            ids = list(self.ids_by_type.get(entity_type, []))

        out: List[Dict[str, Any]] = []
        for i in ids:
            name = self.ent_names[i]
            if q and q not in name.lower():
                continue
            it: Dict[str, Any] = {"id": i, "name": name}
            if inc:
                m = self.entity_meta.get(name)
                if isinstance(m, dict) and m:
                    it["meta"] = m
            out.append(it)
        return {"count": len(out), "items": out[offset : offset + limit], "offset": offset, "limit": limit}

    def get_entity(self, val: Any, *, include_meta: Optional[bool] = None) -> Dict[str, Any]:
        ent_id = _resolve_id(val, self.ent2id, "entity")
        name = self.ent_names[ent_id]
        inc = self._default_include_entity_meta if include_meta is None else bool(include_meta)
        out: Dict[str, Any] = {"id": ent_id, "name": name}
        if inc:
            m = self.entity_meta.get(name)
            if isinstance(m, dict) and m:
                out["meta"] = m
        return out

    def query_entities(
        self,
        *,
        entity_type: str = "",
        where: Optional[List[Dict[str, Any]]] = None,
        relation: Optional[Dict[str, Any]] = None,
        order_by: str = "",
        order: str = "desc",
        limit: int = 50,
        offset: int = 0,
        include_meta: Optional[bool] = None,
    ) -> Dict[str, Any]:
        entity_type = str(entity_type or "").strip()
        limit = max(1, min(int(limit), 5000))
        offset = max(0, int(offset))
        inc = self._default_include_entity_meta if include_meta is None else bool(include_meta)

        ids = list(range(self.info.num_entities))
        if entity_type:
            ids = list(self.ids_by_type.get(entity_type, []))

        if isinstance(relation, dict) and relation:
            r_val = relation.get("r")
            r_id = _resolve_id(r_val, self.rel2id, "relation")
            role = str(relation.get("role", "either")).strip().lower()
            other = relation.get("other")
            if other is None:
                if role == "head":
                    allowed_ids = set(self.heads_by_r.get(r_id, []))
                elif role == "tail":
                    allowed_ids = set(self.tails_by_r.get(r_id, []))
                else:
                    allowed_ids = set(self.heads_by_r.get(r_id, [])) | set(self.tails_by_r.get(r_id, []))
            else:
                other_id = _resolve_id(other, self.ent2id, "entity")
                if role == "tail":
                    allowed_ids = set(self.filter_index.heads_by_rt.get((r_id, other_id), set()))
                else:
                    allowed_ids = set(self.filter_index.tails_by_hr.get((other_id, r_id), set()))
            ids = [i for i in ids if i in allowed_ids]

        def _get_field(eid: int, field: str) -> Any:
            field = str(field or "").strip()
            if field in {"id"}:
                return eid
            if field in {"name"}:
                return self.ent_names[eid]
            meta = self.entity_meta.get(self.ent_names[eid], {})
            if isinstance(meta, dict) and field in meta:
                return meta.get(field)
            return None

        def _match_cond(eid: int, cond: Dict[str, Any]) -> bool:
            if not isinstance(cond, dict):
                return True
            field = cond.get("field")
            op = str(cond.get("op", "exists")).strip().lower()
            val = _get_field(eid, str(field or ""))
            target = cond.get("value")
            if op in {"exists"}:
                return val is not None
            if op in {"missing", "not_exists"}:
                return val is None
            if op in {"=", "=="}:
                return val == target
            if op in {"!=", "<>"}:
                return val != target
            if op in {">", ">=", "<", "<="}:
                if val is None:
                    return False
                try:
                    fv = float(val)
                    ft = float(target)
                except Exception:
                    return False
                if op == ">":
                    return fv > ft
                if op == ">=":
                    return fv >= ft
                if op == "<":
                    return fv < ft
                return fv <= ft
            if op in {"in"}:
                if isinstance(target, list):
                    return val in target
                return False
            if op in {"contains"}:
                if val is None:
                    return False
                return str(target or "").lower() in str(val).lower()
            if op in {"startswith"}:
                if val is None:
                    return False
                return str(val).lower().startswith(str(target or "").lower())
            return True

        if isinstance(where, list) and where:
            ids2: List[int] = []
            for eid in ids:
                ok = True
                for cond in where:
                    if not _match_cond(eid, cond):
                        ok = False
                        break
                if ok:
                    ids2.append(eid)
            ids = ids2

        order_by = str(order_by or "").strip()
        if order_by:
            desc = str(order or "desc").strip().lower() != "asc"

            def _key(eid: int) -> Tuple[int, float, str]:
                v = _get_field(eid, order_by)
                if v is None:
                    return (1, 0.0, self.ent_names[eid])
                try:
                    fv = float(v)
                    return (0, (-fv if desc else fv), self.ent_names[eid])
                except Exception:
                    sv = str(v)
                    return (0, 0.0, ("" if desc else "") + sv)

            ids = sorted(ids, key=_key)

        items: List[Dict[str, Any]] = []
        for eid in ids[offset : offset + limit]:
            it: Dict[str, Any] = {"id": eid, "name": self.ent_names[eid]}
            if inc:
                m = self.entity_meta.get(self.ent_names[eid])
                if isinstance(m, dict) and m:
                    it["meta"] = m
            items.append(it)

        return {"count": len(ids), "items": items, "offset": offset, "limit": limit}

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
        exists = (h_id, r_id, t_id) in self.triple_set
        return {"h": self.ent_names[h_id], "r": self.rel_names[r_id], "t": self.ent_names[t_id], "score": float(sc.item()), "graph_exists": bool(exists)}

    def graph_score(self, *, h: Any, r: Any, t: Any) -> Dict[str, Any]:
        h_id = _resolve_id(h, self.ent2id, "entity")
        r_id = _resolve_id(r, self.rel2id, "relation")
        t_id = _resolve_id(t, self.ent2id, "entity")
        return {"h": self.ent_names[h_id], "r": self.rel_names[r_id], "t": self.ent_names[t_id], "graph_exists": bool((h_id, r_id, t_id) in self.triple_set)}

    def _normalize_mode(self, mode: Optional[str]) -> str:
        m = str(self._default_infer_mode if mode is None else mode).strip().lower()
        if m not in {"infer", "graph", "hybrid"}:
            return "hybrid"
        return m

    def _graph_tail(self, *, h_id: int, r_id: int, candidate_type: str, limit: int) -> List[int]:
        tails = list(self.filter_index.tails_by_hr.get((h_id, r_id), set()))
        if candidate_type:
            allowed = set(self.ids_by_type.get(candidate_type, []))
            if allowed:
                tails = [x for x in tails if x in allowed]
        if limit > 0:
            tails = tails[:limit]
        return tails

    def _graph_head(self, *, t_id: int, r_id: int, candidate_type: str, limit: int) -> List[int]:
        heads = list(self.filter_index.heads_by_rt.get((r_id, t_id), set()))
        if candidate_type:
            allowed = set(self.ids_by_type.get(candidate_type, []))
            if allowed:
                heads = [x for x in heads if x in allowed]
        if limit > 0:
            heads = heads[:limit]
        return heads

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
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        h_id = _resolve_id(h, self.ent2id, "entity")
        r_id = _resolve_id(r, self.rel2id, "relation")
        device = torch.device(self.info.device)

        kk = int(self._default_top_k if k is None else k)
        bb = int(self._default_batch_size if batch_size is None else batch_size)
        ff = bool(self._default_filtered if filtered is None else filtered)
        mm = self._normalize_mode(mode)
        kk = max(1, kk)
        bb = max(1, bb)

        mask = None
        candidate_type = str(candidate_type or "").strip()
        if not candidate_type:
            c = relation_type_constraint(self.rel_names[r_id])
            if c is not None:
                candidate_type = c[1]

        graph_ids: List[int] = []
        if mm in {"graph", "hybrid"}:
            graph_ids = self._graph_tail(h_id=h_id, r_id=r_id, candidate_type=candidate_type, limit=kk if mm == "graph" else kk)
            if mm == "graph":
                return {
                    "mode": "graph",
                    "h": self.ent_names[h_id],
                    "r": self.rel_names[r_id],
                    "topk": [{"t": self.ent_names[i], "score": 0.0, "source": "graph"} for i in graph_ids],
                }
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
        top = _topk(out, k=kk if mm == "infer" else max(0, kk - len(graph_ids)))
        items: List[Dict[str, Any]] = []
        if mm == "hybrid":
            for i in graph_ids:
                items.append({"t": self.ent_names[i], "score": 0.0, "source": "graph"})
        for i, s in top:
            items.append({"t": self.ent_names[i], "score": s, "source": "infer"})
        return {"mode": mm, "h": self.ent_names[h_id], "r": self.rel_names[r_id], "topk": items}

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
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        t_id = _resolve_id(t, self.ent2id, "entity")
        r_id = _resolve_id(r, self.rel2id, "relation")
        device = torch.device(self.info.device)

        kk = int(self._default_top_k if k is None else k)
        bb = int(self._default_batch_size if batch_size is None else batch_size)
        ff = bool(self._default_filtered if filtered is None else filtered)
        mm = self._normalize_mode(mode)
        kk = max(1, kk)
        bb = max(1, bb)

        mask = None
        candidate_type = str(candidate_type or "").strip()
        if not candidate_type:
            c = relation_type_constraint(self.rel_names[r_id])
            if c is not None:
                candidate_type = c[0]

        graph_ids: List[int] = []
        if mm in {"graph", "hybrid"}:
            graph_ids = self._graph_head(t_id=t_id, r_id=r_id, candidate_type=candidate_type, limit=kk if mm == "graph" else kk)
            if mm == "graph":
                return {
                    "mode": "graph",
                    "t": self.ent_names[t_id],
                    "r": self.rel_names[r_id],
                    "topk": [{"h": self.ent_names[i], "score": 0.0, "source": "graph"} for i in graph_ids],
                }
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
        top = _topk(out, k=kk if mm == "infer" else max(0, kk - len(graph_ids)))
        items: List[Dict[str, Any]] = []
        if mm == "hybrid":
            for i in graph_ids:
                items.append({"h": self.ent_names[i], "score": 0.0, "source": "graph"})
        for i, s in top:
            items.append({"h": self.ent_names[i], "score": s, "source": "infer"})
        return {"mode": mm, "t": self.ent_names[t_id], "r": self.rel_names[r_id], "topk": items}

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
