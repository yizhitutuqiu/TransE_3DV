from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from zstp_final.utils.data import FilterIndex
from zstp_final.utils.transe import TransE


@dataclass
class EvalResult:
    mrr: float
    hits1: float
    hits3: float
    hits10: float
    mean_rank: float


def evaluate_filtered(
    model: TransE,
    triples: Sequence[Tuple[int, int, int]],
    *,
    filter_index: FilterIndex,
    num_entities: int,
    device: torch.device,
    batch_size: int = 256,
) -> EvalResult:
    model.eval()
    ranks: List[int] = []

    ent_ids = torch.arange(num_entities, device=device, dtype=torch.long)

    with torch.no_grad():
        for h, t, r in triples:
            h0 = torch.tensor([h], device=device, dtype=torch.long)
            t0 = torch.tensor([t], device=device, dtype=torch.long)
            r0 = torch.tensor([r], device=device, dtype=torch.long)

            all_tails = filter_index.tails_by_hr.get((h, r), set())
            all_heads = filter_index.heads_by_rt.get((r, t), set())

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
            ranks.append(tail_rank)

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
            ranks.append(head_rank)

    mr = float(sum(ranks) / len(ranks)) if ranks else 0.0
    mrr = float(sum(1.0 / r for r in ranks) / len(ranks)) if ranks else 0.0
    hits1 = float(sum(1 for r in ranks if r <= 1) / len(ranks)) if ranks else 0.0
    hits3 = float(sum(1 for r in ranks if r <= 3) / len(ranks)) if ranks else 0.0
    hits10 = float(sum(1 for r in ranks if r <= 10) / len(ranks)) if ranks else 0.0
    return EvalResult(mrr=mrr, hits1=hits1, hits3=hits3, hits10=hits10, mean_rank=mr)

