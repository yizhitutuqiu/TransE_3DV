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


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _log(event: str, **fields: Any) -> None:
    payload = {"ts": _iso_now(), "event": event, **fields}
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)


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
    stage: str,
    cwd: Path,
    env: Dict[str, str],
    dry_run: bool,
    heartbeat_s: float,
) -> None:
    if dry_run:
        print(json.dumps({"cwd": str(cwd), "argv": argv}, ensure_ascii=False))
        return
    _log("proc_start", stage=stage, cwd=str(cwd), argv=argv)
    t0 = time.time()
    hb = max(float(heartbeat_s), 0.0)
    p = subprocess.Popen(argv, cwd=str(cwd), env=env)
    while True:
        if hb <= 0:
            rc = p.wait()
            break
        try:
            rc = p.wait(timeout=hb)
            break
        except subprocess.TimeoutExpired:
            _log("heartbeat", stage=stage, pid=p.pid, elapsed_s=round(time.time() - t0, 3))
            continue
    _log("proc_end", stage=stage, pid=p.pid, returncode=rc, elapsed_s=round(time.time() - t0, 3))
    if rc != 0:
        raise subprocess.CalledProcessError(rc, argv)


def _python_cmd(script: Path) -> List[str]:
    return [sys.executable, str(script)]


def _resolve_path(p: Any, *, base: Path) -> Path:
    if p is None:
        return base
    s = str(p).strip()
    if not s:
        return base
    pp = Path(s).expanduser()
    if pp.is_absolute():
        return pp.resolve()
    return (base / pp).resolve()


def _read_text_if_exists(p: Path) -> str:
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def _is_nonempty_file(p: Path) -> bool:
    try:
        return p.exists() and p.is_file() and p.stat().st_size > 0
    except Exception:
        return False


def _validate_outputs(data_dir: Path, stage: str) -> bool:
    if stage == "crawl":
        raw = data_dir / "raw"
        return (raw / "paper").exists() and (raw / "readme").exists()
    if stage == "preprocess":
        return _is_nonempty_file(data_dir / "preprocessed" / "text" / "documents.jsonl")
    if stage == "entities":
        return _is_nonempty_file(data_dir / "preprocessed" / "text" / "entities" / "doc_entities.jsonl")
    if stage == "registry":
        return _is_nonempty_file(data_dir / "preprocessed" / "kg" / "entities" / "entity_registry.jsonl")
    if stage == "triples_llm":
        return _is_nonempty_file(data_dir / "preprocessed" / "kg" / "triples_llm" / "triples.jsonl")
    if stage == "triples_rules":
        return _is_nonempty_file(data_dir / "preprocessed" / "kg" / "triples" / "triples.jsonl")
    if stage == "final":
        final_dir = data_dir / "preprocessed" / "final"
        need = ["entity2id.txt", "relation2id.txt", "train2id.txt", "test2id.txt", "metadata.json"]
        return all(_is_nonempty_file(final_dir / n) for n in need)
    return False


def _stage_marker(run_dir: Path, stage: str) -> Path:
    return run_dir / f"{stage}.done"


def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k).lower()
            if any(x in ks for x in ["token", "api_key", "apikey", "secret", "password"]):
                out[k] = "<redacted>"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    return obj


def _run_stage(
    *,
    stage: str,
    argv: List[str],
    repo_root: Path,
    data_dir: Path,
    env: Dict[str, str],
    dry_run: bool,
    run_dir: Path,
    heartbeat_s: float,
) -> None:
    marker = _stage_marker(run_dir, stage)
    if marker.exists() and _validate_outputs(data_dir, stage):
        _log("stage_skip", stage=stage, run_dir=str(run_dir))
        return
    _log("stage_start", stage=stage, run_dir=str(run_dir))
    _run(argv, stage=stage, cwd=repo_root, env=env, dry_run=dry_run, heartbeat_s=heartbeat_s)
    if not dry_run and not _validate_outputs(data_dir, stage):
        _log("stage_validation_failed", stage=stage, run_dir=str(run_dir))
        raise RuntimeError(f"stage validation failed: {stage}")
    if not dry_run:
        marker.write_text(json.dumps({"stage": stage, "finished_at": time.time()}, ensure_ascii=False) + "\n", encoding="utf-8")
    _log("stage_done", stage=stage, run_dir=str(run_dir))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=str(repo_root / "data" / "tools" / "config" / "pipeline.yaml"),
    )
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    cfg = _expand(_load_yaml(Path(args.config).resolve()))
    paths = cfg.get("paths", {}) or {}
    stages = cfg.get("stages", {}) or {}
    env_cfg = cfg.get("env", {}) or {}
    logging_cfg = cfg.get("logging", {}) or {}
    heartbeat_s = float(logging_cfg.get("heartbeat_s", 30))

    data_dir = _resolve_path(paths.get("data_dir", "data"), base=repo_root)
    raw_root = _resolve_path(paths.get("raw_root", "data/raw"), base=repo_root)
    pre_text = _resolve_path(paths.get("pre_text_dir", "data/preprocessed/text"), base=repo_root)
    pre_kg = _resolve_path(paths.get("pre_kg_dir", "data/preprocessed/kg"), base=repo_root)
    final_dir = _resolve_path(paths.get("final_dir", "data/preprocessed/final"), base=repo_root)

    run_root = _resolve_path(paths.get("run_root", "data/preprocessed/pipeline_runs"), base=repo_root)
    _ensure_dir(run_root)
    state_path = run_root / "last_run.json"
    state = _load_json(state_path)
    no_resume = bool(args.no_resume) or bool(args.refresh)
    resume_ok = (not no_resume) and state.get("status") == "in_progress" and isinstance(state.get("run_dir"), str)
    if resume_ok and Path(str(state["run_dir"])).exists():
        run_dir = Path(str(state["run_dir"])).resolve()
    else:
        run_dir = run_root / _ts()
        _ensure_dir(run_dir)
        _write_json(state_path, {"status": "in_progress", "run_dir": str(run_dir), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    _ensure_dir(run_dir)
    _log(
        "pipeline_start",
        repo_root=str(repo_root),
        config_path=str(Path(args.config).resolve()),
        run_dir=str(run_dir),
        data_dir=str(data_dir),
        raw_root=str(raw_root),
        pre_text_dir=str(pre_text),
        pre_kg_dir=str(pre_kg),
        final_dir=str(final_dir),
        heartbeat_s=heartbeat_s,
        dry_run=bool(args.dry_run),
        refresh=bool(args.refresh),
    )

    env = os.environ.copy()
    for k, v in env_cfg.items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
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
        argv += ["--out_root", str(raw_root)]
        if bool(args.refresh):
            argv += ["--refresh_semantic_scholar"]
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
        _run_stage(stage="crawl", argv=argv, repo_root=repo_root, data_dir=data_dir, env=env, dry_run=bool(args.dry_run), run_dir=run_dir, heartbeat_s=heartbeat_s)

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
        if bool(args.refresh):
            argv += ["--overwrite"]
        else:
            if _bool(p.get("resume"), True):
                argv += ["--resume"]
            if _bool(p.get("overwrite"), True):
                argv += ["--overwrite"]
        _ensure_dir(pre_text)
        _run_stage(stage="preprocess", argv=argv, repo_root=repo_root, data_dir=data_dir, env=env, dry_run=bool(args.dry_run), run_dir=run_dir, heartbeat_s=heartbeat_s)

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
        _run_stage(stage="entities", argv=argv, repo_root=repo_root, data_dir=data_dir, env=env, dry_run=bool(args.dry_run), run_dir=run_dir, heartbeat_s=heartbeat_s)

    if _bool(stages.get("registry", {}).get("enabled"), True):
        r = stages.get("registry", {}) or {}
        argv = _python_cmd(build_registry_py)
        argv += _as_list(r.get("args"))
        if _bool(r.get("overwrite"), True):
            argv += ["--overwrite"]
        _ensure_dir(pre_kg / "entities")
        _run_stage(stage="registry", argv=argv, repo_root=repo_root, data_dir=data_dir, env=env, dry_run=bool(args.dry_run), run_dir=run_dir, heartbeat_s=heartbeat_s)

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
            _run_stage(stage="triples_llm", argv=argv, repo_root=repo_root, data_dir=data_dir, env=env, dry_run=bool(args.dry_run), run_dir=run_dir, heartbeat_s=heartbeat_s)
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
            _run_stage(stage="triples_rules", argv=argv, repo_root=repo_root, data_dir=data_dir, env=env, dry_run=bool(args.dry_run), run_dir=run_dir, heartbeat_s=heartbeat_s)

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
        _run_stage(stage="final", argv=argv, repo_root=repo_root, data_dir=data_dir, env=env, dry_run=bool(args.dry_run), run_dir=run_dir, heartbeat_s=heartbeat_s)

    (run_dir / "config_snapshot.json").write_text(json.dumps(_redact(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.dry_run:
        _write_json(state_path, {"status": "done", "run_dir": str(run_dir), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    _log("pipeline_done", run_dir=str(run_dir))
    print(str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
