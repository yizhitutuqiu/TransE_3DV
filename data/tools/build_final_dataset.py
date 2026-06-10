from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
            n += 1
    return n


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


def _progress(it: Iterable[Any], *, total: Optional[int], desc: str) -> Iterable[Any]:
    if _tqdm is None:
        return it
    return _tqdm(it, total=total, desc=desc)


def _stable_hash(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def _canonical_method_from_ent_id(ent_id: str) -> str:
    if ent_id.startswith("Method:"):
        return ent_id[len("Method:") :]
    return ent_id


def _should_drop_method(name: str) -> bool:
    t = name.strip()
    if not t:
        return True
    if "http" in t or "www." in t:
        return True
    if len(t) < 3 or len(t) > 80:
        return True
    if re.fullmatch(r"[-_./:()]+", t):
        return True
    stop = {
        "method",
        "model",
        "approach",
        "framework",
        "network",
        "architecture",
        "pipeline",
        "baseline",
        "system",
        "module",
    }
    if t.lower() in stop:
        return True
    return False


def _type_of(ent_id: str) -> Optional[str]:
    if ":" not in ent_id:
        return None
    return ent_id.split(":", 1)[0]


def _valid_type_constraint(r: str, h: str, t: str) -> bool:
    ht = _type_of(h)
    tt = _type_of(t)
    if ht is None or tt is None:
        return False
    if r == "paper_proposes_method":
        return ht == "Paper" and tt == "Method"
    if r == "repo_implements_method":
        return ht == "Repo" and tt == "Method"
    if r == "method_uses_dataset":
        return ht == "Method" and tt == "Dataset"
    if r == "paper_has_repo":
        return ht == "Paper" and tt == "Repo"
    if r == "method_targets_task":
        return ht == "Method" and tt == "Task"
    return False


@dataclass(frozen=True)
class Triple:
    h: str
    r: str
    t: str
    doc_id: str
    confidence: float
    source: str


def _read_triples(path: Path) -> List[Triple]:
    total = _count_lines(path)
    out: List[Triple] = []
    for it in _progress(_iter_jsonl(path), total=total, desc="final:load_triples"):
        h = it.get("h")
        r = it.get("r")
        t = it.get("t")
        doc_id = it.get("doc_id") or ""
        conf = it.get("confidence", 1.0)
        source = it.get("source", "")
        if not isinstance(h, str) or not isinstance(r, str) or not isinstance(t, str) or not isinstance(doc_id, str):
            continue
        if not isinstance(conf, (int, float)):
            conf = 1.0
        out.append(Triple(h=h, r=r, t=t, doc_id=doc_id, confidence=float(conf), source=str(source)))
    return out


def _build_train_triples(
    triples: Sequence[Triple],
    *,
    min_confidence: float,
    allowed_relations: Set[str],
    min_method_doc_freq: int,
) -> Tuple[List[Triple], Dict[str, Any]]:
    filtered: List[Triple] = []
    dropped = Counter()

    method_doc_freq: Dict[str, Set[str]] = defaultdict(set)
    for tr in triples:
        if _type_of(tr.h) == "Method":
            method_doc_freq[tr.h].add(tr.doc_id)
        if _type_of(tr.t) == "Method":
            method_doc_freq[tr.t].add(tr.doc_id)
    method_df = {m: len(ds) for m, ds in method_doc_freq.items()}

    for tr in triples:
        if tr.r not in allowed_relations:
            dropped["relation_not_allowed"] += 1
            continue
        if tr.confidence < min_confidence:
            dropped["low_confidence"] += 1
            continue
        if not _valid_type_constraint(tr.r, tr.h, tr.t):
            dropped["type_mismatch"] += 1
            continue

        if tr.h.startswith("Method:"):
            name = _canonical_method_from_ent_id(tr.h)
            if _should_drop_method(name):
                dropped["bad_method"] += 1
                continue
            if method_df.get(tr.h, 0) < min_method_doc_freq:
                dropped["low_method_df"] += 1
                continue

        if tr.t.startswith("Method:"):
            name = _canonical_method_from_ent_id(tr.t)
            if _should_drop_method(name):
                dropped["bad_method"] += 1
                continue
            if method_df.get(tr.t, 0) < min_method_doc_freq:
                dropped["low_method_df"] += 1
                continue

        filtered.append(tr)

    uniq: Dict[Tuple[str, str, str, str], Triple] = {}
    for tr in filtered:
        k = (tr.h, tr.r, tr.t, tr.doc_id)
        if k not in uniq:
            uniq[k] = tr
    filtered = list(uniq.values())

    by_rel = Counter([t.r for t in filtered])
    meta = {"dropped": dict(dropped), "kept": len(filtered), "by_relation": dict(by_rel)}
    return filtered, meta


def _split_by_doc_id(triples: Sequence[Triple], *, train_ratio: float, seed: int) -> Tuple[List[Triple], List[Triple], Dict[str, Any]]:
    doc_ids = sorted({t.doc_id for t in triples if t.doc_id})
    rng = random.Random(seed)
    rng.shuffle(doc_ids)
    cut = int(round(len(doc_ids) * train_ratio))
    train_docs = set(doc_ids[:cut])
    test_docs = set(doc_ids[cut:])
    train = [t for t in triples if t.doc_id in train_docs]
    test = [t for t in triples if t.doc_id in test_docs]
    meta = {"doc_total": len(doc_ids), "train_docs": len(train_docs), "test_docs": len(test_docs)}
    return train, test, meta


def _split_by_triple(triples: Sequence[Triple], *, train_ratio: float, seed: int) -> Tuple[List[Triple], List[Triple], Dict[str, Any]]:
    rng = random.Random(seed)
    idx = list(range(len(triples)))
    rng.shuffle(idx)
    cut = int(round(len(idx) * train_ratio))
    train_idx = set(idx[:cut])
    train = [triples[i] for i in range(len(triples)) if i in train_idx]
    test = [triples[i] for i in range(len(triples)) if i not in train_idx]
    meta = {"triple_total": len(triples), "train_triples": len(train), "test_triples": len(test)}
    return train, test, meta


def _entity_set(triples: Sequence[Triple]) -> Set[str]:
    out: Set[str] = set()
    for t in triples:
        out.add(t.h)
        out.add(t.t)
    return out


def _oov_fix(train: List[Triple], test: List[Triple]) -> Tuple[List[Triple], List[Triple], Dict[str, Any]]:
    train_ents = _entity_set(train)
    moved: List[Triple] = []
    kept_test: List[Triple] = []
    for tr in test:
        if tr.h in train_ents and tr.t in train_ents:
            kept_test.append(tr)
            continue
        moved.append(tr)
        train.append(tr)
        train_ents.add(tr.h)
        train_ents.add(tr.t)
    meta = {"moved_test_to_train": len(moved), "test_kept": len(kept_test)}
    return train, kept_test, meta


def _build_id_maps(train: Sequence[Triple], test: Sequence[Triple]) -> Tuple[Dict[str, int], Dict[str, int]]:
    ents = sorted(_entity_set(list(train) + list(test)))
    rels = sorted({t.r for t in list(train) + list(test)})
    ent2id = {e: i for i, e in enumerate(ents)}
    rel2id = {r: i for i, r in enumerate(rels)}
    return ent2id, rel2id


def _write_mapping_txt(path: Path, mapping: Dict[str, int]) -> None:
    lines = [str(len(mapping))]
    for k, v in sorted(mapping.items(), key=lambda x: x[1]):
        lines.append(f"{k}\t{v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_transe_triples(path: Path, triples: Sequence[Triple], ent2id: Dict[str, int], rel2id: Dict[str, int]) -> None:
    lines = [str(len(triples))]
    for tr in triples:
        h = ent2id[tr.h]
        t = ent2id[tr.t]
        r = rel2id[tr.r]
        lines.append(f"{h}\t{t}\t{r}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--triples_path",
        type=str,
        default=str((_data_dir() / "preprocessed" / "kg" / "triples_llm" / "triples.jsonl").resolve()),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str((_data_dir() / "preprocessed" / "final").resolve()),
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--split_unit", type=str, default="doc", choices=["doc", "triple"])
    ap.add_argument("--min_confidence", type=float, default=0.65)
    ap.add_argument("--min_method_doc_freq", type=int, default=2)
    ap.add_argument(
        "--relations",
        type=str,
        default="paper_proposes_method,repo_implements_method,method_uses_dataset,paper_cites_paper",
    )
    ap.add_argument("--drop_paper_has_repo", action="store_true")
    args = ap.parse_args()

    triples_path = Path(args.triples_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for p in [
            out_dir / "triples_train.jsonl",
            out_dir / "triples_train2id.txt",
            out_dir / "triples_test.jsonl",
            out_dir / "triples_test2id.txt",
            out_dir / "entity2id.txt",
            out_dir / "relation2id.txt",
            out_dir / "metadata.json",
        ]:
            if p.exists():
                p.unlink()

    raw = _read_triples(triples_path)
    allowed = {r.strip() for r in str(args.relations).split(",") if r.strip()}
    if args.drop_paper_has_repo and "paper_has_repo" in allowed:
        allowed.remove("paper_has_repo")

    train_triples, meta_filter = _build_train_triples(
        raw,
        min_confidence=float(args.min_confidence),
        allowed_relations=allowed,
        min_method_doc_freq=int(args.min_method_doc_freq),
    )

    if args.split_unit == "triple":
        train, test, meta_split = _split_by_triple(train_triples, train_ratio=float(args.train_ratio), seed=int(args.seed))
    else:
        train, test, meta_split = _split_by_doc_id(train_triples, train_ratio=float(args.train_ratio), seed=int(args.seed))
    train, test, meta_oov = _oov_fix(train, test)

    ent2id, rel2id = _build_id_maps(train, test)

    train_jsonl = [{"h": t.h, "r": t.r, "t": t.t, "doc_id": t.doc_id, "confidence": t.confidence, "source": t.source} for t in train]
    test_jsonl = [{"h": t.h, "r": t.r, "t": t.t, "doc_id": t.doc_id, "confidence": t.confidence, "source": t.source} for t in test]
    _write_jsonl(out_dir / "triples_train.jsonl", train_jsonl)
    _write_jsonl(out_dir / "triples_test.jsonl", test_jsonl)

    _write_mapping_txt(out_dir / "entity2id.txt", ent2id)
    _write_mapping_txt(out_dir / "relation2id.txt", rel2id)
    _write_transe_triples(out_dir / "train2id.txt", train, ent2id, rel2id)
    _write_transe_triples(out_dir / "test2id.txt", test, ent2id, rel2id)

    def summarize(trs: Sequence[Triple]) -> Dict[str, Any]:
        by_rel = Counter([t.r for t in trs])
        ents = _entity_set(trs)
        types = Counter(_type_of(e) for e in ents)
        return {"triple_count": len(trs), "by_relation": dict(by_rel), "entity_count": len(ents), "entity_types": dict(types)}

    meta = {
        "created_at": _utc_now_iso(),
        "triples_path": str(triples_path),
        "out_dir": str(out_dir),
        "params": {
            "seed": int(args.seed),
            "train_ratio": float(args.train_ratio),
            "split_unit": str(args.split_unit),
            "min_confidence": float(args.min_confidence),
            "min_method_doc_freq": int(args.min_method_doc_freq),
            "relations": sorted(list(allowed)),
        },
        "raw": {"triple_count": len(raw), "by_relation": dict(Counter([t.r for t in raw]))},
        "filter": meta_filter,
        "split": meta_split,
        "oov_fix": meta_oov,
        "train": summarize(train),
        "test": summarize(test),
        "mappings": {"entity_count": len(ent2id), "relation_count": len(rel2id)},
    }
    _write_json(out_dir / "metadata.json", meta)
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
