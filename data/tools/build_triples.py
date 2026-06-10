from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1]


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


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    s = s.replace("\u00a0", " ").replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _canonical_key(s: str) -> str:
    s = _normalize_ws(s)
    if not s:
        return s
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._+\-:/()]*", s):
        s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


_RE_PAREN_ACRONYM = re.compile(r"^(?P<base>.+?)\s*\((?P<abbr>[A-Za-z0-9-]{2,10})\)\s*$")


def _strip_paren_acronym(s: str) -> str:
    m = _RE_PAREN_ACRONYM.match(s.strip())
    if not m:
        return s.strip()
    base = m.group("base").strip()
    return base if base else s.strip()


def _entity_lookup(
    registry_path: Path,
) -> Tuple[Dict[Tuple[str, str], str], Dict[str, str], Dict[str, str]]:
    by_type_can: Dict[Tuple[str, str], str] = {}
    paper_by_arxiv: Dict[str, str] = {}
    repo_by_fullname_lower: Dict[str, str] = {}
    for it in _iter_jsonl(registry_path):
        typ = it.get("type")
        canonical = it.get("canonical")
        ent_id = it.get("entity_id")
        if not isinstance(typ, str) or not isinstance(canonical, str) or not isinstance(ent_id, str):
            continue
        by_type_can[(typ, canonical)] = ent_id
        by_type_can[(typ, _canonical_key(canonical))] = ent_id

        display_name = it.get("display_name")
        if isinstance(display_name, str) and display_name:
            by_type_can[(typ, _canonical_key(display_name))] = ent_id
            if typ == "Method":
                by_type_can[(typ, _canonical_key(_strip_paren_acronym(display_name)))] = ent_id

        aliases = it.get("aliases")
        if isinstance(aliases, list):
            for a in aliases:
                if not isinstance(a, str) or not a.strip():
                    continue
                by_type_can[(typ, _canonical_key(a))] = ent_id
                if typ == "Method":
                    by_type_can[(typ, _canonical_key(_strip_paren_acronym(a)))] = ent_id

        if typ == "Method":
            by_type_can[(typ, _canonical_key(_strip_paren_acronym(canonical)))] = ent_id
        if typ == "Paper":
            paper_by_arxiv[canonical] = ent_id
        if typ == "Repo":
            repo_by_fullname_lower[canonical.lower()] = ent_id
    return by_type_can, paper_by_arxiv, repo_by_fullname_lower


_RE_ARXIV_NEW = re.compile(r"\b(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?\b", flags=re.I)
_RE_GITHUB_REPO = re.compile(r"https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", flags=re.I)


def _extract_arxiv_ids(text: str) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for m in _RE_ARXIV_NEW.finditer(text):
        aid = m.group(1)
        if aid and aid not in seen:
            seen.add(aid)
            out.append(aid)
    return out


def _extract_github_repos(text: str) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for m in _RE_GITHUB_REPO.finditer(text):
        repo = m.group(1)
        if not repo:
            continue
        repo = repo.rstrip("/").rstrip(".git")
        if repo not in seen:
            seen.add(repo)
            out.append(repo)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--documents_path",
        type=str,
        default=str((_data_dir() / "preprocessed" / "text" / "documents.jsonl").resolve()),
    )
    ap.add_argument(
        "--doc_entities_path",
        type=str,
        default=str((_data_dir() / "preprocessed" / "text" / "entities" / "doc_entities.jsonl").resolve()),
    )
    ap.add_argument(
        "--entity_registry_path",
        type=str,
        default=str((_data_dir() / "preprocessed" / "kg" / "entities" / "entity_registry.jsonl").resolve()),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str((_data_dir() / "preprocessed" / "kg" / "triples").resolve()),
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--enable_paper_has_repo", action="store_true")
    ap.add_argument("--enable_method_uses_dataset", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    documents_path = Path(args.documents_path).resolve()
    doc_entities_path = Path(args.doc_entities_path).resolve()
    registry_path = Path(args.entity_registry_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "triples.jsonl"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} exists, pass --overwrite")

    by_type_can, paper_by_arxiv, repo_by_fullname_lower = _entity_lookup(registry_path)

    doc_text: Dict[str, str] = {}
    doc_refs_arxiv: Dict[str, List[str]] = {}
    total_docs = _count_lines(documents_path)
    for d in _progress(_iter_jsonl(documents_path), total=total_docs, desc="triples:load_docs"):
        doc_id = d.get("doc_id")
        text = d.get("text")
        if isinstance(doc_id, str) and isinstance(text, str):
            doc_text[doc_id] = text
        if isinstance(doc_id, str):
            md = d.get("metadata")
            if isinstance(md, dict):
                refs = md.get("references_arxiv")
                if isinstance(refs, list):
                    out: List[str] = []
                    for r in refs:
                        if isinstance(r, str) and r.strip():
                            out.append(r.strip())
                    if out:
                        doc_refs_arxiv[doc_id] = out

    triples: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    rel_counter = Counter()

    total = _count_lines(doc_entities_path)
    for d in _progress(_iter_jsonl(doc_entities_path), total=total, desc="triples:build"):
        doc_id = d.get("doc_id")
        doc_type = d.get("doc_type")
        category = d.get("category")
        ents = d.get("entities")
        if not isinstance(doc_id, str) or not doc_id or not isinstance(ents, list):
            continue

        paper_ent: Optional[str] = None
        repo_ent: Optional[str] = None
        methods: Set[str] = set()
        datasets: Set[str] = set()

        for e in ents:
            if not isinstance(e, dict):
                continue
            typ = e.get("type")
            canonical = e.get("canonical")
            if not isinstance(typ, str) or not isinstance(canonical, str) or not typ or not canonical:
                continue
            ent_id = by_type_can.get((typ, canonical)) or by_type_can.get((typ, _canonical_key(canonical)))
            if ent_id is None and typ == "Method":
                ent_id = by_type_can.get((typ, _canonical_key(_strip_paren_acronym(canonical))))
            if ent_id is None:
                continue
            if typ == "Paper":
                paper_ent = ent_id
            elif typ == "Repo":
                repo_ent = ent_id
            elif typ == "Method":
                methods.add(ent_id)
            elif typ == "Dataset":
                datasets.add(ent_id)

        task_ent: Optional[str] = None
        if isinstance(category, str) and category:
            task_ent = by_type_can.get(("Task", _canonical_key(category))) or by_type_can.get(("Task", category))

        def add(h: str, r: str, t: str, source: str) -> None:
            k = (h, r, t)
            if k in seen:
                return
            seen.add(k)
            triples.append({"h": h, "r": r, "t": t, "doc_id": doc_id, "source": source})
            rel_counter[r] += 1

        if paper_ent:
            for m in methods:
                add(paper_ent, "paper_proposes_method", m, "doc_methods")
        if repo_ent:
            for m in methods:
                add(repo_ent, "repo_implements_method", m, "doc_methods")

        if task_ent:
            for m in methods:
                add(m, "method_targets_task", task_ent, "doc_category")

        if args.enable_method_uses_dataset and datasets:
            for m in methods:
                for ds in datasets:
                    add(m, "method_uses_dataset", ds, "doc_cooccur")

        if args.enable_paper_has_repo:
            text = doc_text.get(doc_id, "")
            if isinstance(text, str) and text:
                if repo_ent and doc_type == "readme":
                    for aid in _extract_arxiv_ids(text):
                        pid = paper_by_arxiv.get(aid)
                        if pid:
                            add(pid, "paper_has_repo", repo_ent, "readme_arxiv_match")
                if paper_ent and doc_type == "paper":
                    for repo in _extract_github_repos(text):
                        rid = repo_by_fullname_lower.get(repo.lower())
                        if rid:
                            add(paper_ent, "paper_has_repo", rid, "paper_github_url")

        if paper_ent and doc_type == "paper":
            refs = doc_refs_arxiv.get(doc_id) or []
            for aid in refs:
                pid = paper_by_arxiv.get(aid)
                if pid:
                    add(paper_ent, "paper_cites_paper", pid, "semantic_scholar")

        if args.limit and len(triples) >= args.limit:
            break

    _write_jsonl(out_path, triples)
    rels = sorted(rel_counter.keys())
    (out_dir / "relations.txt").write_text("\n".join(rels) + "\n", encoding="utf-8")

    stats = {
        "created_at": _utc_now_iso(),
        "documents_path": str(documents_path),
        "doc_entities_path": str(doc_entities_path),
        "entity_registry_path": str(registry_path),
        "out_dir": str(out_dir),
        "triples_path": str(out_path),
        "triple_count": len(triples),
        "by_relation": dict(rel_counter),
        "enable_paper_has_repo": bool(args.enable_paper_has_repo),
        "enable_method_uses_dataset": bool(args.enable_method_uses_dataset),
    }
    _write_json(out_dir / "stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
