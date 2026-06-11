from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from zstp_final.utils.data import build_filter_index, load_id_map, load_triples_hrt
from zstp_final.utils.transe import TransE


@dataclass
class RankStats:
    count: int
    mean_rank: float
    mrr: float
    hits1: float
    hits3: float
    hits10: float


def _summarize_ranks(ranks: List[int]) -> RankStats:
    if not ranks:
        return RankStats(count=0, mean_rank=0.0, mrr=0.0, hits1=0.0, hits3=0.0, hits10=0.0)
    mr = float(sum(ranks) / len(ranks))
    mrr = float(sum(1.0 / r for r in ranks) / len(ranks))
    hits1 = float(sum(1 for r in ranks if r <= 1) / len(ranks))
    hits3 = float(sum(1 for r in ranks if r <= 3) / len(ranks))
    hits10 = float(sum(1 for r in ranks if r <= 10) / len(ranks))
    return RankStats(count=len(ranks), mean_rank=mr, mrr=mrr, hits1=hits1, hits3=hits3, hits10=hits10)


def _eval_ranks_filtered(
    model: TransE,
    triples: Sequence[Tuple[int, int, int]],
    *,
    filter_index,
    num_entities: int,
    device: torch.device,
    batch_size: int,
) -> Tuple[List[int], List[int], Dict[int, List[int]], Dict[int, List[int]]]:
    model.eval()
    ent_ids = torch.arange(num_entities, device=device, dtype=torch.long)

    head_ranks: List[int] = []
    tail_ranks: List[int] = []
    head_ranks_by_r: Dict[int, List[int]] = defaultdict(list)
    tail_ranks_by_r: Dict[int, List[int]] = defaultdict(list)

    with torch.no_grad():
        for h, t, r in triples:
            h0 = torch.tensor([h], device=device, dtype=torch.long)
            t0 = torch.tensor([t], device=device, dtype=torch.long)
            r0 = torch.tensor([r], device=device, dtype=torch.long)

            all_tails = filter_index.tails_by_hr.get((h, r), set())
            tail_scores: List[torch.Tensor] = []
            for start in range(0, num_entities, batch_size):
                cand = ent_ids[start : start + batch_size]
                sc = model.score(h0.expand_as(cand), r0.expand_as(cand), cand)
                tail_scores.append(sc)
            tail_scores_t = torch.cat(tail_scores, dim=0)
            if all_tails:
                mask = torch.zeros(num_entities, device=device, dtype=torch.bool)
                for tt in all_tails:
                    if tt != t:
                        mask[tt] = True
                tail_scores_t = tail_scores_t.masked_fill(mask, float("inf"))
            tail_rank = int((tail_scores_t < tail_scores_t[t]).sum().item()) + 1
            tail_ranks.append(tail_rank)
            tail_ranks_by_r[r].append(tail_rank)

            all_heads = filter_index.heads_by_rt.get((r, t), set())
            head_scores: List[torch.Tensor] = []
            for start in range(0, num_entities, batch_size):
                cand = ent_ids[start : start + batch_size]
                sc = model.score(cand, r0.expand_as(cand), t0.expand_as(cand))
                head_scores.append(sc)
            head_scores_t = torch.cat(head_scores, dim=0)
            if all_heads:
                mask = torch.zeros(num_entities, device=device, dtype=torch.bool)
                for hh in all_heads:
                    if hh != h:
                        mask[hh] = True
                head_scores_t = head_scores_t.masked_fill(mask, float("inf"))
            head_rank = int((head_scores_t < head_scores_t[h]).sum().item()) + 1
            head_ranks.append(head_rank)
            head_ranks_by_r[r].append(head_rank)

    return head_ranks, tail_ranks, head_ranks_by_r, tail_ranks_by_r


def _random_baseline(num_entities: int) -> Dict[str, float]:
    n = float(num_entities)
    hits1 = 1.0 / n
    hits3 = min(3.0 / n, 1.0)
    hits10 = min(10.0 / n, 1.0)
    mrr = float(sum(1.0 / i for i in range(1, num_entities + 1)) / n)
    mr = (n + 1.0) / 2.0
    return {"mrr": mrr, "mean_rank": mr, "hits1": hits1, "hits3": hits3, "hits10": hits10}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--embedding_dim", type=int, default=0)
    ap.add_argument("--p_norm", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    ent2id, ent_names = load_id_map(str(data_dir / "entity2id.txt"))
    rel2id, rel_names = load_id_map(str(data_dir / "relation2id.txt"))
    train_triples = load_triples_hrt(str(data_dir / "train2id.txt"))
    test_triples = load_triples_hrt(str(data_dir / "test2id.txt"))

    device = torch.device(args.device)

    ckpt = torch.load(str(Path(args.ckpt).resolve()), map_location=device)
    st = ckpt.get("model_state", {}) or {}
    ent_w = st.get("ent.weight", None)
    rel_w = st.get("rel.weight", None)

    inferred_dim = int(ent_w.shape[1]) if hasattr(ent_w, "shape") and len(ent_w.shape) == 2 else 0
    inferred_rels = int(rel_w.shape[0]) if hasattr(rel_w, "shape") and len(rel_w.shape) == 2 else 0
    ckpt_args = ckpt.get("args", {}) or {}
    inferred_dim = inferred_dim or int(ckpt_args.get("embedding_dim") or 0)
    inferred_p = int(ckpt_args.get("p_norm") or 0)

    embedding_dim = inferred_dim or int(args.embedding_dim) or 100
    p_norm = inferred_p or int(args.p_norm) or 1
    num_relations = inferred_rels or len(rel2id)

    model = TransE(num_entities=len(ent2id), num_relations=num_relations, embedding_dim=embedding_dim, p_norm=p_norm).to(device)
    model.load_state_dict(ckpt["model_state"])

    filter_index = build_filter_index(list(train_triples) + list(test_triples))
    head_ranks, tail_ranks, head_by_r, tail_by_r = _eval_ranks_filtered(
        model,
        test_triples,
        filter_index=filter_index,
        num_entities=len(ent2id),
        device=device,
        batch_size=int(args.batch_size),
    )

    all_ranks = head_ranks + tail_ranks
    by_rel: Dict[str, Dict[str, Any]] = {}
    for r_id in range(len(rel2id)):
        name = rel_names[r_id]
        rks = (head_by_r.get(r_id, []) + tail_by_r.get(r_id, []))
        by_rel[name] = asdict(_summarize_ranks(rks))

    hist = Counter()
    for rk in all_ranks:
        if rk <= 1:
            hist["1"] += 1
        elif rk <= 3:
            hist["2-3"] += 1
        elif rk <= 10:
            hist["4-10"] += 1
        elif rk <= 50:
            hist["11-50"] += 1
        elif rk <= 100:
            hist["51-100"] += 1
        else:
            hist[">100"] += 1

    report = {
        "num_entities": len(ent2id),
        "num_relations": len(rel2id),
        "train_triples": len(train_triples),
        "test_triples": len(test_triples),
        "test_queries": len(all_ranks),
        "overall": asdict(_summarize_ranks(all_ranks)),
        "tail_only": asdict(_summarize_ranks(tail_ranks)),
        "head_only": asdict(_summarize_ranks(head_ranks)),
        "random_baseline": _random_baseline(len(ent2id)),
        "rank_histogram": dict(hist),
        "by_relation": by_rel,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
