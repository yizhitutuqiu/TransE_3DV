from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _write_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
            n += 1
    return n


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


def _progress(it: Iterable[Any], *, total: Optional[int], desc: str) -> Iterable[Any]:
    if _tqdm is None:
        return it
    return _tqdm(it, total=total, desc=desc)


def _normalize_ws(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _clean_paper_text(title: str, summary: str) -> str:
    title = _normalize_ws(title)
    summary = _normalize_ws(summary)
    t = f"{title}\n\n{summary}".strip()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _strip_badge_lines(md: str) -> str:
    md = re.sub(r"<!--[\s\S]*?-->", "\n", md)
    lines = md.splitlines()
    out: List[str] = []
    for ln in lines:
        t = ln.strip()
        if not t:
            out.append("")
            continue
        if re.fullmatch(r"\[!\[.*\]\(.*\)\]\(.*\)", t):
            continue
        if t.startswith("![") and "](" in t and t.endswith(")"):
            continue
        if t.startswith("[comment]: <>"):
            continue
        if t.startswith("<!--") and t.endswith("-->"):
            continue
        out.append(ln)
    return "\n".join(out)


def _remove_fenced_code_blocks(md: str) -> str:
    md = re.sub(r"```[\s\S]*?```", "\n", md)
    md = re.sub(r"~~~[\s\S]*?~~~", "\n", md)
    return md


def _heading_level(line: str) -> Optional[int]:
    m = re.match(r"^\s*(#{1,6})\s+\S+", line)
    if not m:
        return None
    return len(m.group(1))


def _filter_md_sections(md: str, drop_section_keywords: List[str]) -> str:
    lines = md.splitlines()
    out: List[str] = []
    skip = False
    skip_level: Optional[int] = None

    def is_drop_heading(text: str) -> bool:
        t = text.strip().lower()
        if not t.startswith("#"):
            return False
        t = re.sub(r"^#+\s*", "", t)
        for kw in drop_section_keywords:
            if kw in t:
                return True
        return False

    for ln in lines:
        lvl = _heading_level(ln)
        if lvl is not None:
            if skip and skip_level is not None and lvl <= skip_level:
                skip = False
                skip_level = None
            if is_drop_heading(ln):
                skip = True
                skip_level = lvl
                continue
        if skip:
            continue
        out.append(ln)
    return "\n".join(out)


def _md_to_text(md: str) -> str:
    md = re.sub(r"<[^>]+>", " ", md)
    md = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", md)
    md = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", md)
    md = re.sub(r"^\s{0,3}#{1,6}\s*", "", md, flags=re.M)
    md = re.sub(r"^\s{0,3}[-*+]\s+", "", md, flags=re.M)
    md = re.sub(r"^\s{0,3}\d+\.\s+", "", md, flags=re.M)
    md = re.sub(r"`([^`]+)`", r"\1", md)
    md = re.sub(r"[\r\t]+", " ", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return _normalize_ws(md)


def _clean_readme_text(md: str, *, max_chars: int) -> Tuple[str, Dict[str, Any]]:
    raw_len = len(md)
    md = md.replace("\r\n", "\n")
    md = _strip_badge_lines(md)
    md = _remove_fenced_code_blocks(md)
    md = _filter_md_sections(
        md,
        drop_section_keywords=[
            "installation",
            "install",
            "setup",
            "requirements",
            "dependency",
            "getting started",
            "quick start",
            "quickstart",
            "usage",
            "demo",
            "train",
            "training",
            "evaluation",
            "inference",
            "run",
            "license",
            "citation",
            "acknowledg",
            "download",
            "benchmark",
        ],
    )
    text = _md_to_text(md)
    text = text.replace("-->", " ")
    text = re.sub(r"\bhttps?://\S+\b", " ", text)
    text = _normalize_ws(text)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    meta = {"raw_len": raw_len, "clean_len": len(text)}
    return text, meta


def _safe_int(v: Any) -> Optional[int]:
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.isdigit():
        return int(v)
    return None


def _extract_year_from_iso(s: Any) -> Optional[int]:
    if not isinstance(s, str):
        return None
    m = re.match(r"^(\d{4})-\d{2}-\d{2}", s)
    if not m:
        return None
    return int(m.group(1))


def _build_docs_from_papers(
    *,
    raw_root: Path,
    max_doc_chars: int,
    min_doc_chars: int,
) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    paper_root = raw_root / "paper"
    cat_dirs = [p for p in sorted(paper_root.iterdir())] if paper_root.exists() else []
    for cat_dir in _progress(cat_dirs, total=len(cat_dirs), desc="papers:categories"):
        if not cat_dir.is_dir():
            continue
        items_path = cat_dir / "items.jsonl"
        if not items_path.exists():
            continue
        total = _count_lines(items_path)
        for it in _progress(_iter_jsonl(items_path), total=total, desc=f"papers:{cat_dir.name}"):
            arxiv_id = it.get("arxiv_id")
            if not isinstance(arxiv_id, str) or not arxiv_id:
                continue
            title = it.get("title") if isinstance(it.get("title"), str) else ""
            summary = it.get("summary") if isinstance(it.get("summary"), str) else ""
            text = _clean_paper_text(title, summary)
            if len(text) > max_doc_chars:
                text = text[:max_doc_chars].rsplit(" ", 1)[0]
            if len(text) < min_doc_chars:
                continue

            category = it.get("category") if isinstance(it.get("category"), str) else cat_dir.name
            category_zh = it.get("category_zh") if isinstance(it.get("category_zh"), str) else ""
            year = _extract_year_from_iso(it.get("published")) or _safe_int(it.get("year"))

            doc = {
                "doc_id": f"arxiv:{arxiv_id}",
                "doc_type": "paper",
                "source": "arxiv",
                "category": category,
                "category_zh": category_zh,
                "title": _normalize_ws(title),
                "text": text,
                "text_sha256": _sha256_text(text),
                "text_len": len(text),
                "created_at": _utc_now_iso(),
                "metadata": {
                    "arxiv_id": arxiv_id,
                    "arxiv_id_raw": it.get("arxiv_id_raw"),
                    "authors": it.get("authors"),
                    "published": it.get("published"),
                    "updated": it.get("updated"),
                    "year": year,
                    "primary_category": it.get("primary_category"),
                    "categories": it.get("categories"),
                    "links": it.get("links"),
                    "query": it.get("query"),
                    "retrieved_at": it.get("retrieved_at"),
                },
            }
            docs.append(doc)
    return docs


def _build_docs_from_readmes(
    *,
    raw_root: Path,
    max_doc_chars: int,
    min_doc_chars: int,
) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    readme_root = raw_root / "readme"
    cat_dirs = [p for p in sorted(readme_root.iterdir())] if readme_root.exists() else []
    for cat_dir in _progress(cat_dirs, total=len(cat_dirs), desc="readmes:categories"):
        if not cat_dir.is_dir():
            continue
        items_path = cat_dir / "items.jsonl"
        if not items_path.exists():
            continue
        total = _count_lines(items_path)
        for it in _progress(_iter_jsonl(items_path), total=total, desc=f"readmes:{cat_dir.name}"):
            full_name = it.get("repo_full_name")
            if not isinstance(full_name, str) or not full_name:
                continue
            md = it.get("readme_text") if isinstance(it.get("readme_text"), str) else ""
            if not md:
                continue
            text, clean_meta = _clean_readme_text(md, max_chars=max_doc_chars)
            if not text:
                continue
            if len(text) < min_doc_chars:
                continue

            category = it.get("category") if isinstance(it.get("category"), str) else cat_dir.name
            category_zh = it.get("category_zh") if isinstance(it.get("category_zh"), str) else ""
            year = _extract_year_from_iso(it.get("repo_created_at"))

            doc = {
                "doc_id": f"github:{full_name}",
                "doc_type": "readme",
                "source": "github",
                "category": category,
                "category_zh": category_zh,
                "title": it.get("repo_description") or full_name,
                "text": text,
                "text_sha256": _sha256_text(text),
                "text_len": len(text),
                "created_at": _utc_now_iso(),
                "metadata": {
                    "repo_full_name": full_name,
                    "repo_html_url": it.get("repo_html_url"),
                    "repo_description": it.get("repo_description"),
                    "repo_language": it.get("repo_language"),
                    "repo_stargazers_count": it.get("repo_stargazers_count"),
                    "repo_forks_count": it.get("repo_forks_count"),
                    "repo_open_issues_count": it.get("repo_open_issues_count"),
                    "repo_default_branch": it.get("repo_default_branch"),
                    "repo_topics": it.get("repo_topics"),
                    "repo_license": it.get("repo_license"),
                    "repo_created_at": it.get("repo_created_at"),
                    "repo_updated_at": it.get("repo_updated_at"),
                    "year": year,
                    "query": it.get("query"),
                    "retrieved_at": it.get("retrieved_at"),
                    "readme_sha256": it.get("readme_sha256"),
                    "readme_cleaning": clean_meta,
                },
            }
            docs.append(doc)
    return docs


def build_ontology_schema() -> Dict[str, Any]:
    return {
        "version": "0.1",
        "created_at": _utc_now_iso(),
        "domain": "3d_vision_paper_code_ecosystem",
        "entity_types": [
            {"type": "Paper", "id_field": "arxiv_id", "examples": ["arxiv:2409.06662"]},
            {"type": "Repo", "id_field": "repo_full_name", "examples": ["github:princeton-vl/DPVO"]},
            {"type": "Method", "id_field": "canonical_name", "examples": ["NeRF", "DPVO", "3D Gaussian Splatting"]},
            {"type": "Task", "id_field": "canonical_name", "examples": ["Pose Estimation", "SLAM", "3D Reconstruction"]},
            {"type": "Dataset", "id_field": "canonical_name", "examples": ["KITTI", "ScanNet", "TUM RGB-D"]},
            {"type": "Venue", "id_field": "canonical_name", "examples": ["CVPR", "ICCV", "ECCV", "NeurIPS", "SIGGRAPH"]},
            {"type": "Metric", "id_field": "canonical_name", "examples": ["PSNR", "SSIM", "ATE", "mAP", "Chamfer Distance"]},
        ],
        "relation_types": [
            {"type": "paper_has_repo", "head": "Paper", "tail": "Repo"},
            {"type": "paper_proposes_method", "head": "Paper", "tail": "Method"},
            {"type": "repo_implements_method", "head": "Repo", "tail": "Method"},
            {"type": "method_targets_task", "head": "Method", "tail": "Task"},
            {"type": "method_uses_dataset", "head": "Method", "tail": "Dataset"},
            {"type": "paper_published_in", "head": "Paper", "tail": "Venue"},
            {"type": "method_reports_metric", "head": "Method", "tail": "Metric"},
        ],
        "document_unit_schema": {
            "doc_id": "str (unique; arxiv:<id> or github:<full_name>)",
            "doc_type": "paper | readme",
            "source": "arxiv | github",
            "category": "pose_estimation | 3d_generation | 4d_reconstruction",
            "category_zh": "str",
            "title": "str",
            "text": "str (cleaned)",
            "text_sha256": "str",
            "text_len": "int",
            "created_at": "iso8601",
            "metadata": "object",
        },
    }


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw_root",
        type=str,
        default=str((_data_dir() / "raw").resolve()),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str((_data_dir() / "preprocessed" / "text").resolve()),
    )
    ap.add_argument("--max_doc_chars", type=int, default=12000)
    ap.add_argument("--min_doc_chars", type=int, default=80)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ontology = build_ontology_schema()
    (out_dir / "ontology_schema.json").write_text(json.dumps(ontology, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    docs: List[Dict[str, Any]] = []
    docs.extend(
        _build_docs_from_papers(
            raw_root=raw_root,
            max_doc_chars=max(1000, args.max_doc_chars),
            min_doc_chars=max(1, args.min_doc_chars),
        )
    )
    docs.extend(
        _build_docs_from_readmes(
            raw_root=raw_root,
            max_doc_chars=max(1000, args.max_doc_chars),
            min_doc_chars=max(1, args.min_doc_chars),
        )
    )

    docs_path = out_dir / "documents.jsonl"
    existing = 0
    seen_doc_ids = set()
    if docs_path.exists() and (args.resume and not args.overwrite):
        for it in _iter_jsonl(docs_path):
            doc_id = it.get("doc_id") if isinstance(it, dict) else None
            if isinstance(doc_id, str) and doc_id:
                seen_doc_ids.add(doc_id)
            existing += 1

    if args.overwrite or not docs_path.exists() or not args.resume:
        n = _write_jsonl(docs_path, docs)
        new_n = n
        total_n = n
    else:
        new_docs = []
        for d in docs:
            doc_id = d.get("doc_id")
            if isinstance(doc_id, str) and doc_id in seen_doc_ids:
                continue
            new_docs.append(d)
        if new_docs:
            with docs_path.open("a", encoding="utf-8") as f:
                for it in new_docs:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
        new_n = len(new_docs)
        total_n = existing + new_n
        n = new_n

    by_type = Counter(d["doc_type"] for d in docs)
    by_cat = Counter(d["category"] for d in docs)
    lens = [int(d.get("text_len") or 0) for d in docs]
    lens.sort()

    def pct(p: float) -> int:
        if not lens:
            return 0
        idx = int(round((len(lens) - 1) * p))
        return lens[max(0, min(idx, len(lens) - 1))]

    stats = {
        "created_at": _utc_now_iso(),
        "raw_root": str(raw_root),
        "out_dir": str(out_dir),
        "documents_path": str(docs_path),
        "document_count_total": total_n,
        "document_count_new": new_n,
        "by_doc_type": dict(by_type),
        "by_category": dict(by_cat),
        "text_len": {
            "min": lens[0] if lens else 0,
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p99": pct(0.99),
            "max": lens[-1] if lens else 0,
        },
        "write_mode": "overwrite" if (args.overwrite or not args.resume) else "append_new",
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
