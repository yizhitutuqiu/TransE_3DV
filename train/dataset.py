from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TripleItem:
    h: int
    t: int
    r: int


class KGTripleDataset(Dataset):
    def __init__(self, triples: List[Tuple[int, int, int]]) -> None:
        self.triples = [TripleItem(h=h, t=t, r=r) for (h, t, r) in triples]

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        it = self.triples[idx]
        return torch.tensor([it.h, it.t, it.r], dtype=torch.long)

