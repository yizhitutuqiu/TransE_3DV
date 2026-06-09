from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class TransE(nn.Module):
    def __init__(
        self,
        *,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        p_norm: int = 1,
    ) -> None:
        super().__init__()
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.embedding_dim = int(embedding_dim)
        self.p_norm = int(p_norm)

        self.ent = nn.Embedding(self.num_entities, self.embedding_dim)
        self.rel = nn.Embedding(self.num_relations, self.embedding_dim)
        nn.init.xavier_uniform_(self.ent.weight.data)
        nn.init.xavier_uniform_(self.rel.weight.data)

    def normalize_entities_(self) -> None:
        with torch.no_grad():
            self.ent.weight.data = torch.nn.functional.normalize(self.ent.weight.data, p=2, dim=-1)

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        he = self.ent(h)
        re = self.rel(r)
        te = self.ent(t)
        return torch.linalg.vector_norm(he + re - te, ord=self.p_norm, dim=-1)

    def forward(self, pos_hrt: torch.Tensor, neg_hrt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ph, pt, pr = pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2]
        nh, nt, nr = neg_hrt[:, 0], neg_hrt[:, 1], neg_hrt[:, 2]
        pos = self.score(ph, pr, pt)
        neg = self.score(nh, nr, nt)
        return pos, neg

