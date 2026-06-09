from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import requests

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1]


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


def _append_jsonl_line(path: Path, obj: Dict[str, Any], *, lock: Optional[Lock] = None) -> None:
    s = json.dumps(obj, ensure_ascii=False) + "\n"
    if lock is None:
        with path.open("a", encoding="utf-8") as f:
            f.write(s)
        return
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(s)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


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


def _extract_first_json(text: str) -> Optional[Any]:
    t = text.strip()
    if not t:
        return None
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        frag = t[start : end + 1]
        try:
            return json.loads(frag)
        except json.JSONDecodeError:
            pass
    start = t.find("[")
    end = t.rfind("]")
    if start != -1 and end != -1 and end > start:
        frag = t[start : end + 1]
        try:
            return json.loads(frag)
        except json.JSONDecodeError:
            pass
    return None


@dataclass
class DoubaoConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    connect_timeout_s: float
    read_timeout_s: float
    api_mode: str


def _doubao_from_env(*, model_fallback: str, base_url_fallback: str, api_mode_fallback: str) -> DoubaoConfig:
    api_key = os.environ.get("DOUBAO_API_KEY") or os.environ.get("ARK_API_KEY") or ""
    base_url = os.environ.get("DOUBAO_BASE_URL") or os.environ.get("ARK_BASE_URL") or base_url_fallback
    model = os.environ.get("DOUBAO_MODEL") or os.environ.get("ARK_MODEL") or model_fallback
    api_mode = os.environ.get("DOUBAO_API_MODE") or os.environ.get("ARK_API_MODE") or api_mode_fallback
    temperature = float(os.environ.get("DOUBAO_TEMPERATURE") or "0")
    max_tokens = int(os.environ.get("DOUBAO_MAX_TOKENS") or "2048")
    connect_timeout_s = float(os.environ.get("DOUBAO_CONNECT_TIMEOUT_S") or "10")
    read_timeout_s = float(os.environ.get("DOUBAO_READ_TIMEOUT_S") or "180")
    return DoubaoConfig(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        connect_timeout_s=connect_timeout_s,
        read_timeout_s=read_timeout_s,
        api_mode=api_mode,
    )


def _extract_text_from_ark_payload(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    output = data.get("output")
    if isinstance(output, list) and output:
        texts: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    t = c.get("text")
                    if isinstance(t, str) and t:
                        texts.append(t)
            elif isinstance(content, str) and content:
                texts.append(content)
        if texts:
            return "\n".join(texts).strip()
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content:
                return content
    content = data.get("content")
    if isinstance(content, str) and content:
        return content
    return None


def _doubao_responses_api(session: requests.Session, cfg: DoubaoConfig, *, system: str, user: str) -> str:
    url = f"{cfg.base_url}/responses"
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    full_user = (system + "\n\n" + user).strip()
    payload = {"model": cfg.model, "input": [{"role": "user", "content": [{"type": "input_text", "text": full_user}]}]}
    resp = session.post(
        url,
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=(cfg.connect_timeout_s, cfg.read_timeout_s),
    )
    resp.raise_for_status()
    data = resp.json()
    text = _extract_text_from_ark_payload(data)
    if isinstance(text, str) and text:
        return text
    return json.dumps(data, ensure_ascii=False)


def _doubao_chat_completions_api(session: requests.Session, cfg: DoubaoConfig, *, system: str, user: str) -> str:
    url = f"{cfg.base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    payload = {
        "model": cfg.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    resp = session.post(
        url,
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=(cfg.connect_timeout_s, cfg.read_timeout_s),
    )
    resp.raise_for_status()
    data = resp.json()
    text = _extract_text_from_ark_payload(data)
    if isinstance(text, str) and text:
        return text
    return json.dumps(data, ensure_ascii=False)


def _doubao_infer(session: requests.Session, cfg: DoubaoConfig, *, system: str, user: str) -> str:
    if cfg.api_mode == "chat_completions":
        return _doubao_chat_completions_api(session, cfg, system=system, user=user)
    try:
        return _doubao_responses_api(session, cfg, system=system, user=user)
    except requests.HTTPError as e:
        resp = getattr(e, "response", None)
        if resp is not None and resp.status_code in (404, 405):
            return _doubao_chat_completions_api(session, cfg, system=system, user=user)
        raise


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


def _load_documents(documents_path: Path, *, max_chars: int) -> Dict[str, Dict[str, Any]]:
    doc_map: Dict[str, Dict[str, Any]] = {}
    total = _count_lines(documents_path)
    it = _iter_jsonl(documents_path)
    if _tqdm is not None:
        it = _tqdm(it, total=total, desc="llm_triples:load_docs")
    for d in it:
        doc_id = d.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            continue
        text = d.get("text") if isinstance(d.get("text"), str) else ""
        title = d.get("title") if isinstance(d.get("title"), str) else ""
        full = _normalize_ws(f"{title}\n\n{text}")
        if len(full) > max_chars:
            full = full[:max_chars].rsplit(" ", 1)[0]
        doc_map[doc_id] = {
            "doc_id": doc_id,
            "doc_type": d.get("doc_type"),
            "category": d.get("category"),
            "category_zh": d.get("category_zh"),
            "title": title,
            "text": full,
        }
    return doc_map


def _load_registry(registry_path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, str]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    paper_by_arxiv: Dict[str, str] = {}
    repo_by_fullname_lower: Dict[str, str] = {}
    for it in _iter_jsonl(registry_path):
        ent_id = it.get("entity_id")
        typ = it.get("type")
        canonical = it.get("canonical")
        if not isinstance(ent_id, str) or not isinstance(typ, str) or not isinstance(canonical, str):
            continue
        by_id[ent_id] = it
        if typ == "Paper":
            paper_by_arxiv[canonical] = ent_id
        if typ == "Repo":
            repo_by_fullname_lower[canonical.lower()] = ent_id
    return by_id, paper_by_arxiv, repo_by_fullname_lower


_RE_PAREN_ACRONYM = re.compile(r"^(?P<base>.+?)\s*\((?P<abbr>[A-Za-z0-9-]{2,10})\)\s*$")


def _strip_paren_acronym(s: str) -> str:
    m = _RE_PAREN_ACRONYM.match(s.strip())
    if not m:
        return s.strip()
    base = m.group("base").strip()
    return base if base else s.strip()


def _build_type_can_to_id(by_id: Dict[str, Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
    out: Dict[Tuple[str, str], str] = {}
    for ent_id, it in by_id.items():
        typ = it.get("type")
        canonical = it.get("canonical")
        if not isinstance(typ, str) or not isinstance(canonical, str):
            continue
        out[(typ, canonical)] = ent_id
        out[(typ, _canonical_key(canonical))] = ent_id
        display_name = it.get("display_name")
        if isinstance(display_name, str) and display_name:
            out[(typ, _canonical_key(display_name))] = ent_id
            if typ == "Method":
                out[(typ, _canonical_key(_strip_paren_acronym(display_name)))] = ent_id
        aliases = it.get("aliases")
        if isinstance(aliases, list):
            for a in aliases:
                if isinstance(a, str) and a.strip():
                    out[(typ, _canonical_key(a))] = ent_id
                    if typ == "Method":
                        out[(typ, _canonical_key(_strip_paren_acronym(a)))] = ent_id
        if typ == "Method":
            out[(typ, _canonical_key(_strip_paren_acronym(canonical)))] = ent_id
    return out


def _load_processed_doc_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    out: Set[str] = set()
    for it in _iter_jsonl(path):
        doc_id = it.get("doc_id")
        if isinstance(doc_id, str) and doc_id:
            out.add(doc_id)
    return out


def _chunked(seq: Sequence[Any], n: int) -> List[Sequence[Any]]:
    if n <= 0:
        return [seq]
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def _entity_ids_by_type(entities: List[Dict[str, Any]], *, by_id: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    seen: Set[str] = set()
    for e in entities:
        if not isinstance(e, dict):
            continue
        typ = e.get("type")
        ent_id = e.get("entity_id")
        if isinstance(typ, str) and isinstance(ent_id, str) and ent_id in by_id and ent_id not in seen:
            out[typ].append(ent_id)
            seen.add(ent_id)
    return out


def _build_candidate_entities(
    doc_id: str,
    doc_type: str,
    doc_text: str,
    *,
    entities_by_type: Dict[str, List[str]],
    by_id: Dict[str, Dict[str, Any]],
    paper_by_arxiv: Dict[str, str],
    repo_by_fullname_lower: Dict[str, str],
    max_methods: int,
    max_datasets: int,
) -> Dict[str, Any]:
    paper_id = entities_by_type.get("Paper", [None])[0]
    repo_id = entities_by_type.get("Repo", [None])[0]
    methods = entities_by_type.get("Method", [])[:max_methods]
    datasets = entities_by_type.get("Dataset", [])[:max_datasets]

    arxiv_mentions = _extract_arxiv_ids(doc_text)
    arxiv_candidates = []
    for aid in arxiv_mentions[:10]:
        pid = paper_by_arxiv.get(aid)
        if pid:
            arxiv_candidates.append({"arxiv_id": aid, "entity_id": pid})

    repo_mentions = _extract_github_repos(doc_text)
    repo_candidates = []
    for r in repo_mentions[:10]:
        rid = repo_by_fullname_lower.get(r.lower())
        if rid:
            repo_candidates.append({"repo": r, "entity_id": rid})

    def view(ent_id: str) -> Dict[str, Any]:
        it = by_id.get(ent_id, {})
        name = it.get("display_name") if isinstance(it.get("display_name"), str) else it.get("canonical")
        return {"id": ent_id, "name": name}

    return {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "paper_id": paper_id,
        "repo_id": repo_id,
        "methods": [view(x) for x in methods],
        "datasets": [view(x) for x in datasets],
        "paper_candidates": arxiv_candidates,
        "repo_candidates": repo_candidates,
    }


def _build_re_prompt_batch(items: Sequence[Dict[str, Any]]) -> Tuple[str, str, List[str]]:
    system = "You extract high-precision KG triples from 3D vision text. Output must be strict JSON."
    doc_ids: List[str] = []
    parts: List[str] = []
    for it in items:
        doc_id = it["doc"]["doc_id"]
        doc_ids.append(doc_id)
        doc = it["doc"]
        cands = it["cands"]
        parts.append(
            "\n".join(
                [
                    f"doc_id: {doc_id}",
                    f"doc_type: {doc.get('doc_type')}",
                    f"category: {doc.get('category')}",
                    "text:",
                    doc.get("text", ""),
                    "candidates:",
                    json.dumps(cands, ensure_ascii=False),
                ]
            )
        )

    user = (
        "For each item, generate KG triples using ONLY the candidate ids.\n"
        "Allowed relations:\n"
        "- paper_proposes_method (Paper -> Method): only if the paper introduces/proposes the method.\n"
        "- repo_implements_method (Repo -> Method): only if the repository implements the method.\n"
        "- method_uses_dataset (Method -> Dataset): only if explicitly used for training/evaluation/benchmark.\n"
        "- paper_has_repo (Paper -> Repo): only if explicitly linked/mentioned.\n"
        "Rules:\n"
        "- Use only candidate ids from candidates.methods/datasets/paper_id/repo_id/paper_candidates/repo_candidates.\n"
        "- If uncertain, return empty array.\n"
        "Return format:\n"
        "{ \"<doc_id>\": [ {\"h\":\"<entity_id>\",\"r\":\"<relation>\",\"t\":\"<entity_id>\",\"confidence\":0-1} ] }\n"
        "\n\n---\n\n"
        + "\n\n---\n\n".join(parts)
    )
    return system, user, doc_ids


def _validate_triples(
    doc_id: str,
    obj: Any,
    *,
    known_ids: Set[str],
) -> List[Dict[str, Any]]:
    if not isinstance(obj, list):
        return []
    out: List[Dict[str, Any]] = []
    for t in obj:
        if not isinstance(t, dict):
            continue
        h = t.get("h")
        r = t.get("r")
        ta = t.get("t")
        conf = t.get("confidence")
        if not isinstance(h, str) or not isinstance(r, str) or not isinstance(ta, str):
            continue
        if h not in known_ids or ta not in known_ids:
            continue
        if r not in {"paper_proposes_method", "repo_implements_method", "method_uses_dataset", "paper_has_repo"}:
            continue
        if not isinstance(conf, (int, float)):
            conf = 0.6
        conf = float(conf)
        if conf < 0:
            conf = 0.0
        if conf > 1:
            conf = 1.0
        out.append({"h": h, "r": r, "t": ta, "doc_id": doc_id, "confidence": conf, "source": "llm"})
    return out


def _request_with_retries(
    session: requests.Session,
    cfg: DoubaoConfig,
    *,
    system: str,
    user: str,
    max_retries: int,
    backoff_s: float,
) -> str:
    last: Optional[Exception] = None
    for i in range(max_retries + 1):
        try:
            return _doubao_infer(session, cfg, system=system, user=user)
        except Exception as e:
            last = e
            if i >= max_retries:
                break
            time.sleep(backoff_s * (2**i))
    raise last if last is not None else RuntimeError("unknown")


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
        default=str((_data_dir() / "preprocessed" / "kg" / "triples_llm").resolve()),
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_docs", type=int, default=0)
    ap.add_argument("--max_doc_chars", type=int, default=3500)
    ap.add_argument("--max_methods", type=int, default=40)
    ap.add_argument("--max_datasets", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--request_workers", type=int, default=6)
    ap.add_argument("--sleep_s", type=float, default=0.2)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--backoff_s", type=float, default=1.0)
    ap.add_argument("--min_confidence", type=float, default=0.65)
    ap.add_argument("--doubao_model", type=str, default="doubao-seed-2-0-pro-260215")
    ap.add_argument("--doubao_base_url", type=str, default="https://ark.cn-beijing.volces.com/api/v3")
    ap.add_argument("--doubao_api_mode", type=str, default="responses", choices=["responses", "chat_completions"])
    args = ap.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_triples_path = out_dir / "doc_triples.jsonl"
    triples_path = out_dir / "triples.jsonl"
    cache_path = out_dir / "llm_cache.jsonl"
    stats_path = out_dir / "stats.json"

    if args.overwrite:
        for p in [doc_triples_path, triples_path, cache_path, stats_path]:
            if p.exists():
                p.unlink()

    processed = _load_processed_doc_ids(doc_triples_path) if args.resume else set()

    cfg = _doubao_from_env(
        model_fallback=args.doubao_model,
        base_url_fallback=args.doubao_base_url,
        api_mode_fallback=args.doubao_api_mode,
    )
    if not cfg.api_key:
        raise SystemExit("Missing ARK_API_KEY (or DOUBAO_API_KEY) in environment.")

    documents_path = Path(args.documents_path).resolve()
    doc_entities_path = Path(args.doc_entities_path).resolve()
    registry_path = Path(args.entity_registry_path).resolve()

    doc_map = _load_documents(documents_path, max_chars=max(500, int(args.max_doc_chars)))
    by_id, paper_by_arxiv, repo_by_fullname_lower = _load_registry(registry_path)
    type_can_to_id = _build_type_can_to_id(by_id)
    known_ids = set(by_id.keys())

    todo_docs: List[str] = []
    doc_entities_map: Dict[str, List[Dict[str, Any]]] = {}
    total = _count_lines(doc_entities_path)
    it = _iter_jsonl(doc_entities_path)
    if _tqdm is not None:
        it = _tqdm(it, total=total, desc="llm_triples:scan_doc_entities")
    for d in it:
        doc_id = d.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            continue
        ents = d.get("entities")
        if isinstance(ents, list):
            doc_entities_map[doc_id] = ents
        if doc_id in processed:
            continue
        if doc_id not in doc_map:
            continue
        todo_docs.append(doc_id)
        if args.limit_docs and len(todo_docs) >= args.limit_docs:
            break

    batches = _chunked(todo_docs, max(1, int(args.batch_size)))
    session = requests.Session()
    lock = Lock()

    rel_counter = Counter()
    triple_seen: Set[Tuple[str, str, str]] = set()

    def run_batch(doc_ids: Sequence[str]) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []
        for doc_id in doc_ids:
            doc = doc_map[doc_id]
            ents = doc_entities_map.get(doc_id)
            if not isinstance(ents, list):
                continue
            ent_by_type = defaultdict(list)
            for e in ents:
                if not isinstance(e, dict):
                    continue
                typ = e.get("type")
                canonical = e.get("canonical")
                if not isinstance(typ, str) or not isinstance(canonical, str):
                    continue
                ent_id = type_can_to_id.get((typ, canonical)) or type_can_to_id.get((typ, _canonical_key(canonical)))
                if ent_id is None and typ == "Method":
                    ent_id = type_can_to_id.get((typ, _canonical_key(_strip_paren_acronym(canonical))))
                if ent_id is not None and ent_id in by_id:
                    ent_by_type[typ].append(ent_id)

            cands = _build_candidate_entities(
                doc_id=doc_id,
                doc_type=str(doc.get("doc_type") or ""),
                doc_text=str(doc.get("text") or ""),
                entities_by_type=ent_by_type,
                by_id=by_id,
                paper_by_arxiv=paper_by_arxiv,
                repo_by_fullname_lower=repo_by_fullname_lower,
                max_methods=int(args.max_methods),
                max_datasets=int(args.max_datasets),
            )
            items.append({"doc": doc, "cands": cands})

        system, user, used_doc_ids = _build_re_prompt_batch(items)
        prompt_sha = _sha256_text(system + "\n" + user)
        raw = _request_with_retries(
            session,
            cfg,
            system=system,
            user=user,
            max_retries=int(args.max_retries),
            backoff_s=float(args.backoff_s),
        )
        parsed = _extract_first_json(raw)
        ok = isinstance(parsed, dict)
        cache_item = {
            "doc_ids": used_doc_ids,
            "model": cfg.model,
            "base_url": cfg.base_url,
            "api_mode": cfg.api_mode,
            "prompt_sha256": prompt_sha,
            "ok": ok,
            "raw": raw,
            "created_at": _utc_now_iso(),
        }
        _append_jsonl_line(cache_path, cache_item, lock=lock)
        if not ok:
            return {"doc_triples": {}, "error": "parse_failed"}
        return {"doc_triples": parsed, "error": ""}

    pbar = _tqdm(total=len(todo_docs), desc="llm_triples:requests") if _tqdm is not None else None

    def handle_result(res: Dict[str, Any]) -> int:
        doc_triples = res.get("doc_triples")
        if not isinstance(doc_triples, dict):
            return 0
        wrote = 0
        for doc_id, triples_obj in doc_triples.items():
            if not isinstance(doc_id, str) or not doc_id:
                continue
            triples = _validate_triples(doc_id, triples_obj, known_ids=known_ids)
            triples = [t for t in triples if float(t.get("confidence", 0)) >= float(args.min_confidence)]
            _append_jsonl_line(doc_triples_path, {"doc_id": doc_id, "triples": triples, "created_at": _utc_now_iso()}, lock=lock)
            for t in triples:
                k = (t["h"], t["r"], t["t"])
                if k in triple_seen:
                    continue
                triple_seen.add(k)
                _append_jsonl_line(triples_path, t, lock=lock)
                rel_counter[t["r"]] += 1
                wrote += 1
        return wrote

    total_written = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.request_workers))) as ex:
        futs = {ex.submit(run_batch, b): b for b in batches}
        for fut in as_completed(futs):
            b = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                _append_jsonl_line(
                    cache_path,
                    {"doc_ids": list(b), "ok": False, "error": f"{type(e).__name__}:{str(e)[:200]}", "created_at": _utc_now_iso()},
                    lock=lock,
                )
                if pbar is not None:
                    pbar.update(len(b))
                continue

            wrote = handle_result(res)
            total_written += wrote
            if pbar is not None:
                pbar.update(len(b))
                pbar.set_postfix(triples=total_written)
            time.sleep(max(0.0, float(args.sleep_s)))

    if pbar is not None:
        pbar.close()

    stats = {
        "created_at": _utc_now_iso(),
        "documents_path": str(documents_path),
        "doc_entities_path": str(doc_entities_path),
        "entity_registry_path": str(registry_path),
        "out_dir": str(out_dir),
        "doc_triples_path": str(doc_triples_path),
        "triples_path": str(triples_path),
        "cache_path": str(cache_path),
        "processed_docs": len(todo_docs),
        "triple_count": int(sum(rel_counter.values())),
        "by_relation": dict(rel_counter),
        "min_confidence": float(args.min_confidence),
        "batch_size": int(args.batch_size),
        "request_workers": int(args.request_workers),
        "model": cfg.model,
        "base_url": cfg.base_url,
        "api_mode": cfg.api_mode,
    }
    _write_json(stats_path, stats)
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
