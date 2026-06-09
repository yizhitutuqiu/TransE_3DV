from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch


def corrupt_batch(
    pos_hrt: torch.Tensor,
    *,
    num_entities: int,
    rng: random.Random,
) -> torch.Tensor:
    h = pos_hrt[:, 0].clone()
    t = pos_hrt[:, 1].clone()
    r = pos_hrt[:, 2].clone()
    bsz = pos_hrt.size(0)
    for i in range(bsz):
        if rng.random() < 0.5:
            nh = rng.randrange(num_entities)
            h[i] = nh
        else:
            nt = rng.randrange(num_entities)
            t[i] = nt
    return torch.stack([h, t, r], dim=1)

