from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set


def _derive_pair_names(name: str) -> tuple[str, str]:
    p = Path(name)
    stem = p.stem
    suf = p.suffix or ".png"
    if stem.endswith("_1"):
        return f"{stem}{suf}", f"{stem[:-2]}_2{suf}"
    if stem.endswith("_2"):
        return f"{stem[:-2]}_1{suf}", f"{stem}{suf}"
    return f"{stem}_1{suf}", f"{stem}_2{suf}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefer_paper", type=str, default="1812.03828")
    ap.add_argument("--triples_train", type=str, default="data/preprocessed/final/triples_train.jsonl")
    ap.add_argument("--triples_test", type=str, default="data/preprocessed/final/triples_test.jsonl")
    ap.add_argument("--bg_nodes", type=int, default=25)
    ap.add_argument("--core_degree_cap", type=int, default=5)
    ap.add_argument("--bg_degree_cap", type=int, default=5)
    ap.add_argument("--core_scale", type=float, default=1.25)
    ap.add_argument("--spring_k", type=float, default=0.0)
    ap.add_argument("--spring_iter", type=int, default=260)
    ap.add_argument("--bg_ring_radius", type=float, default=0.0)
    ap.add_argument("--bg_jitter", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="output/vis")
    ap.add_argument("--out_name", type=str, default="compact_4x4_min_bg_pair.png")
    args = ap.parse_args()

    import plot_compact_4x4_min_bg as base

    root = Path(__file__).resolve().parents[1]
    train_path = (root / str(args.triples_train)).resolve() if not Path(str(args.triples_train)).is_absolute() else Path(str(args.triples_train)).resolve()
    test_path = (root / str(args.triples_test)).resolve() if not Path(str(args.triples_test)).is_absolute() else Path(str(args.triples_test)).resolve()
    triples = base._load_triples([train_path, test_path])

    out_dir = (root / str(args.out_dir)).resolve() if not Path(str(args.out_dir)).is_absolute() else Path(str(args.out_dir)).resolve()
    out1_name, out2_name = _derive_pair_names(str(args.out_name))
    out1 = out_dir / out1_name
    out2 = out_dir / out2_name

    res1 = base.render(
        prefer_paper=str(args.prefer_paper),
        triples=triples,
        out_path=out1,
        bg_nodes_target=int(args.bg_nodes),
        core_degree_cap=int(args.core_degree_cap),
        bg_degree_cap=int(args.bg_degree_cap),
        core_scale=float(args.core_scale),
        spring_k=float(args.spring_k),
        spring_iter=int(args.spring_iter),
        bg_ring_radius=float(args.bg_ring_radius),
        bg_jitter=float(args.bg_jitter),
        seed=int(args.seed),
        exclude_core_nodes=set(),
    )

    exclude: Set[str] = set(res1.get("core_nodes") or [])
    res2 = base.render(
        prefer_paper="",
        triples=triples,
        out_path=out2,
        bg_nodes_target=int(args.bg_nodes),
        core_degree_cap=int(args.core_degree_cap),
        bg_degree_cap=int(args.bg_degree_cap),
        core_scale=float(args.core_scale),
        spring_k=float(args.spring_k),
        spring_iter=int(args.spring_iter),
        bg_ring_radius=float(args.bg_ring_radius),
        bg_jitter=float(args.bg_jitter),
        seed=int(args.seed) + 1,
        exclude_core_nodes=exclude,
    )

    print(str(res1["out_path"]))
    print(str(res2["out_path"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

