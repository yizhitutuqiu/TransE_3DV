from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from zstp_final.train.dataset import KGTripleDataset
from zstp_final.train.evaluate import evaluate_filtered
from zstp_final.train.sampling import corrupt_batch
from zstp_final.utils.data import build_filter_index, load_id_map, load_triples_hrt
from zstp_final.utils.transe import TransE

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _now() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _save_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="/data/litengmo/ml-test-1/zstp_final/data/preprocessed/final")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--embedding_dim", type=int, default=100)
    ap.add_argument("--p_norm", type=int, default=1)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--eval_batch_size", type=int, default=512)
    ap.add_argument("--normalize_every", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    ent2id, _ = load_id_map(str(data_dir / "entity2id.txt"))
    rel2id, _ = load_id_map(str(data_dir / "relation2id.txt"))
    train_triples = load_triples_hrt(str(data_dir / "train2id.txt"))
    test_triples = load_triples_hrt(str(data_dir / "test2id.txt"))

    num_entities = len(ent2id)
    num_relations = len(rel2id)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    run_name = args.run_name or f"transe_{_now()}"
    ckpt_root = Path("/data/litengmo/ml-test-1/zstp_final/checkpoints").resolve()
    run_dir = ckpt_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    model = TransE(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=int(args.embedding_dim),
        p_norm=int(args.p_norm),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    dataset = KGTripleDataset(train_triples)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, drop_last=False)

    filter_index = build_filter_index(list(train_triples) + list(test_triples))

    best_mrr = -1.0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        it = loader
        if tqdm is not None:
            it = tqdm(it, desc=f"train:epoch{epoch}", leave=False)

        rng = random.Random(args.seed + epoch)
        for pos in it:
            pos = pos.to(device)
            neg = corrupt_batch(pos, num_entities=num_entities, rng=rng).to(device)
            pos_s, neg_s = model(pos, neg)
            loss = F.relu(float(args.margin) + pos_s - neg_s).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1
            if tqdm is not None:
                it.set_postfix(loss=float(loss.item()))

        if int(args.normalize_every) > 0 and epoch % int(args.normalize_every) == 0:
            model.normalize_entities_()

        avg_loss = total_loss / max(1, steps)

        record = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "num_entities": num_entities,
            "num_relations": num_relations,
        }

        if int(args.eval_every) > 0 and epoch % int(args.eval_every) == 0:
            eval_res = evaluate_filtered(
                model,
                test_triples,
                filter_index=filter_index,
                num_entities=num_entities,
                device=device,
                batch_size=int(args.eval_batch_size),
            )
            record.update(
                {
                    "mrr": eval_res.mrr,
                    "hits1": eval_res.hits1,
                    "hits3": eval_res.hits3,
                    "hits10": eval_res.hits10,
                    "mean_rank": eval_res.mean_rank,
                }
            )

            if eval_res.mrr > best_mrr:
                best_mrr = eval_res.mrr
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_mrr": best_mrr,
                        "args": vars(args),
                    },
                    best_path,
                )

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_mrr": best_mrr,
                "args": vars(args),
            },
            last_path,
        )

        _save_jsonl(metrics_path, record)

    print(str(best_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
