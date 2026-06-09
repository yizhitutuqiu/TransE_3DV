from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError("config must be a mapping")
    return obj


def _expand(v: Any) -> Any:
    if isinstance(v, str):
        return os.path.expandvars(v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def _bool(v: Any, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _run(
    argv: List[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    dry_run: bool,
) -> None:
    if dry_run:
        print(json.dumps({"cwd": str(cwd), "argv": argv}, ensure_ascii=False))
        return
    subprocess.run(argv, cwd=str(cwd), env=env, check=True)


def _python_cmd(script: Path) -> List[str]:
    return [sys.executable, str(script)]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=str(repo_root / "data" / "tools" / "config" / "pipeline.yaml"),
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = _expand(_load_yaml(Path(args.config).resolve()))
    paths = cfg.get("paths", {}) or {}
    stages = cfg.get("stages", {}) or {}
    env_cfg = cfg.get("env", {}) or {}

    data_dir = Path(paths.get("data_dir", repo_root / "data")).resolve()
    raw_root = Path(paths.get("raw_root", data_dir / "raw")).resolve()
    pre_text = Path(paths.get("pre_text_dir", data_dir / "preprocessed" / "text")).resolve()
    pre_kg = Path(paths.get("pre_kg_dir", data_dir / "preprocessed" / "kg")).resolve()
    final_dir = Path(paths.get("final_dir", data_dir / "preprocessed" / "final")).resolve()

    run_root = Path(paths.get("run_root", data_dir / "preprocessed" / "pipeline_runs")).resolve()
    run_dir = run_root / _ts()
    _ensure_dir(run_dir)

    env = os.environ.copy()
    for k, v in env_cfg.items():
        if v is None:
            continue
        env[str(k)] = str(v)

    scripts_root = data_dir / "tools"
    crawl_py = scripts_root / "crawl_3d_vision_corpus.py"
    preprocess_py = scripts_root / "preprocess_text_to_docs.py"
    extract_entities_py = scripts_root / "extract_entities.py"
    build_registry_py = scripts_root / "build_entity_registry.py"
    build_triples_py = scripts_root / "build_triples.py"
    build_triples_llm_py = scripts_root / "build_triples_llm.py"
    build_final_py = scripts_root / "build_final_dataset.py"

    if _bool(stages.get("crawl", {}).get("enabled"), True):
        c = stages.get("crawl", {}) or {}
        argv = _python_cmd(crawl_py)
        argv += _as_list(c.get("args"))
        argv += ["--mode", str(c.get("mode", "all"))]
        if c.get("max_papers") is not None:
            argv += ["--max_papers", str(c.get("max_papers"))]
        if c.get("max_repos") is not None:
            argv += ["--max_repos", str(c.get("max_repos"))]
        if c.get("github_min_stars") is not None:
            argv += ["--github_min_stars", str(c.get("github_min_stars"))]
        if c.get("year_start") is not None:
            argv += ["--year_start", str(c.get("year_start"))]
        if c.get("year_end") is not None:
            argv += ["--year_end", str(c.get("year_end"))]
        if _bool(c.get("overwrite"), False):
            argv += ["--overwrite"]
        _ensure_dir(raw_root / "paper")
        _ensure_dir(raw_root / "readme")
        _run(argv, cwd=repo_root, env=env, dry_run=bool(args.dry_run))

    if _bool(stages.get("preprocess", {}).get("enabled"), True):
        p = stages.get("preprocess", {}) or {}
        argv = _python_cmd(preprocess_py)
        argv += _as_list(p.get("args"))
        argv += ["--raw_root", str(raw_root)]
        argv += ["--out_dir", str(pre_text)]
        if p.get("max_doc_chars") is not None:
            argv += ["--max_doc_chars", str(p.get("max_doc_chars"))]
        if p.get("min_doc_chars") is not None:
            argv += ["--min_doc_chars", str(p.get("min_doc_chars"))]
        if _bool(p.get("overwrite"), True):
            argv += ["--overwrite"]
        _ensure_dir(pre_text)
        _run(argv, cwd=repo_root, env=env, dry_run=bool(args.dry_run))

    if _bool(stages.get("entities", {}).get("enabled"), True):
        e = stages.get("entities", {}) or {}
        argv = _python_cmd(extract_entities_py)
        argv += _as_list(e.get("args"))
        if _bool(e.get("overwrite"), True):
            argv += ["--overwrite"]
        if _bool(e.get("resume"), False):
            argv += ["--resume"]
        if _bool(e.get("enable_llm"), False):
            argv += ["--enable_llm"]
        _run(argv, cwd=repo_root, env=env, dry_run=bool(args.dry_run))

    if _bool(stages.get("registry", {}).get("enabled"), True):
        r = stages.get("registry", {}) or {}
        argv = _python_cmd(build_registry_py)
        argv += _as_list(r.get("args"))
        if _bool(r.get("overwrite"), True):
            argv += ["--overwrite"]
        _ensure_dir(pre_kg / "entities")
        _run(argv, cwd=repo_root, env=env, dry_run=bool(args.dry_run))

    triples_mode = str((stages.get("triples", {}) or {}).get("mode", "llm")).strip().lower()
    if _bool(stages.get("triples", {}).get("enabled"), True):
        t = stages.get("triples", {}) or {}
        if triples_mode == "llm":
            argv = _python_cmd(build_triples_llm_py)
            argv += _as_list(t.get("args"))
            if _bool(t.get("overwrite"), True):
                argv += ["--overwrite"]
            if _bool(t.get("resume"), False):
                argv += ["--resume"]
            if t.get("limit_docs") is not None:
                argv += ["--limit_docs", str(t.get("limit_docs"))]
            if t.get("max_doc_chars") is not None:
                argv += ["--max_doc_chars", str(t.get("max_doc_chars"))]
            if t.get("max_methods") is not None:
                argv += ["--max_methods", str(t.get("max_methods"))]
            if t.get("max_datasets") is not None:
                argv += ["--max_datasets", str(t.get("max_datasets"))]
            if t.get("batch_size") is not None:
                argv += ["--batch_size", str(t.get("batch_size"))]
            if t.get("request_workers") is not None:
                argv += ["--request_workers", str(t.get("request_workers"))]
            if t.get("sleep_s") is not None:
                argv += ["--sleep_s", str(t.get("sleep_s"))]
            if t.get("min_confidence") is not None:
                argv += ["--min_confidence", str(t.get("min_confidence"))]
            _ensure_dir(pre_kg / "triples_llm")
            _run(argv, cwd=repo_root, env=env, dry_run=bool(args.dry_run))
        else:
            argv = _python_cmd(build_triples_py)
            argv += _as_list(t.get("args"))
            if _bool(t.get("overwrite"), True):
                argv += ["--overwrite"]
            if _bool(t.get("enable_paper_has_repo"), False):
                argv += ["--enable_paper_has_repo"]
            if _bool(t.get("enable_method_uses_dataset"), False):
                argv += ["--enable_method_uses_dataset"]
            _ensure_dir(pre_kg / "triples")
            _run(argv, cwd=repo_root, env=env, dry_run=bool(args.dry_run))

    if _bool(stages.get("final", {}).get("enabled"), True):
        f = stages.get("final", {}) or {}
        argv = _python_cmd(build_final_py)
        argv += _as_list(f.get("args"))
        if _bool(f.get("overwrite"), True):
            argv += ["--overwrite"]
        argv += ["--out_dir", str(final_dir)]
        if f.get("train_ratio") is not None:
            argv += ["--train_ratio", str(f.get("train_ratio"))]
        if f.get("split_unit") is not None:
            argv += ["--split_unit", str(f.get("split_unit"))]
        if f.get("min_confidence") is not None:
            argv += ["--min_confidence", str(f.get("min_confidence"))]
        if f.get("min_method_doc_freq") is not None:
            argv += ["--min_method_doc_freq", str(f.get("min_method_doc_freq"))]

        triples_path = f.get("triples_path")
        if triples_path:
            argv += ["--triples_path", str(triples_path)]
        else:
            if triples_mode == "llm":
                argv += ["--triples_path", str(pre_kg / "triples_llm" / "triples.jsonl")]
            else:
                argv += ["--triples_path", str(pre_kg / "triples" / "triples.jsonl")]
        _ensure_dir(final_dir)
        _run(argv, cwd=repo_root, env=env, dry_run=bool(args.dry_run))

    (run_dir / "config_snapshot.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

