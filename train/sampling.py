from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch


def corrupt_batch(
    pos_hrt: torch.Tensor,
    *,
    num_entities: int,
    rng: random.Random,
    type_aware: bool = True,
    head_candidates: Optional[Dict[int, List[int]]] = None,
    tail_candidates: Optional[Dict[int, List[int]]] = None,
) -> torch.Tensor:
    h = pos_hrt[:, 0].clone()
    t = pos_hrt[:, 1].clone()
    r = pos_hrt[:, 2].clone()
    bsz = pos_hrt.size(0)
    for i in range(bsz):
        rr = int(r[i].item())
        if rng.random() < 0.5:
            if type_aware and head_candidates is not None:
                cand = head_candidates.get(rr)
            else:
                cand = None
            if cand:
                nh = rng.choice(cand)
            else:
                nh = rng.randrange(num_entities)
            h[i] = nh
        else:
            if type_aware and tail_candidates is not None:
                cand = tail_candidates.get(rr)
            else:
                cand = None
            if cand:
                nt = rng.choice(cand)
            else:
                nt = rng.randrange(num_entities)
            t[i] = nt
    return torch.stack([h, t, r], dim=1)
