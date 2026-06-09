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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import requests

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


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


def _append_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("a", encoding="utf-8") as f:
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


def _progress(it: Iterable[Any], *, total: Optional[int], desc: str, initial: int = 0) -> Iterable[Any]:
    if _tqdm is None:
        return it
    return _tqdm(it, total=total, desc=desc, initial=initial)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


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


def _load_dict_terms(path: Path) -> List[str]:
    if not path.exists():
        return []
    out: List[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        t = ln.strip()
        if not t or t.startswith("#"):
            continue
        out.append(t)
    out.sort(key=len, reverse=True)
    return out


def _compile_term_patterns(
    terms: Sequence[str],
    *,
    case_sensitive_all_caps_leq: int = 0,
) -> List[Tuple[str, re.Pattern[str]]]:
    pats: List[Tuple[str, re.Pattern[str]]] = []
    for t in terms:
        esc = re.escape(t)
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._+\-:/()]*", t):
            flags = re.I
            is_all_caps_short = (
                bool(case_sensitive_all_caps_leq)
                and len(t) <= case_sensitive_all_caps_leq
                and t.isalpha()
                and t.upper() == t
            )
            if is_all_caps_short:
                flags = 0
            boundary = r"[A-Za-z0-9]"
            if is_all_caps_short:
                boundary = r"[A-Za-z0-9-]"
            pat = re.compile(rf"(?<!{boundary}){esc}(?!{boundary})", flags=flags)
        else:
            pat = re.compile(esc)
        pats.append((t, pat))
    return pats


def _find_terms(text: str, patterns: Sequence[Tuple[str, re.Pattern[str]]], *, max_hits: int) -> List[str]:
    hits: List[str] = []
    seen: Set[str] = set()
    for term, pat in patterns:
        if len(hits) >= max_hits:
            break
        if pat.search(text) and term not in seen:
            hits.append(term)
            seen.add(term)
    return hits


def _extract_structured_entities(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    doc_type = doc.get("doc_type")
    meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}

    if doc_type == "paper":
        arxiv_id = meta.get("arxiv_id")
        if isinstance(arxiv_id, str) and arxiv_id:
            out.append(
                {
                    "type": "Paper",
                    "mention": arxiv_id,
                    "canonical": arxiv_id,
                    "source": "structured",
                }
            )
    elif doc_type == "readme":
        repo_full_name = meta.get("repo_full_name")
        if isinstance(repo_full_name, str) and repo_full_name:
            out.append(
                {
                    "type": "Repo",
                    "mention": repo_full_name,
                    "canonical": repo_full_name,
                    "source": "structured",
                }
            )

    category = doc.get("category")
    category_zh = doc.get("category_zh")
    if isinstance(category, str) and category:
        out.append(
            {
                "type": "Task",
                "mention": category_zh if isinstance(category_zh, str) and category_zh else category,
                "canonical": category,
                "source": "structured",
            }
        )
    return out


def _extract_dict_entities(
    doc: Dict[str, Any],
    *,
    dataset_pats: Sequence[Tuple[str, re.Pattern[str]]],
    venue_pats: Sequence[Tuple[str, re.Pattern[str]]],
    metric_pats: Sequence[Tuple[str, re.Pattern[str]]],
    task_pats: Sequence[Tuple[str, re.Pattern[str]]],
    max_hits_each: int,
) -> List[Dict[str, Any]]:
    text = doc.get("text")
    if not isinstance(text, str) or not text:
        return []
    text = _normalize_ws(text)

    out: List[Dict[str, Any]] = []
    for typ, pats in [
        ("Dataset", dataset_pats),
        ("Venue", venue_pats),
        ("Metric", metric_pats),
        ("Task", task_pats),
    ]:
        for m in _find_terms(text, pats, max_hits=max_hits_each):
            canonical = _canonical_key(m)
            if typ == "Metric" and m.isalpha() and m.upper() == m and len(m) <= 4:
                canonical = m
            out.append({"type": typ, "mention": m, "canonical": canonical, "source": "dict"})
    return out


def _extract_first_json(text: str) -> Optional[Any]:
    t = text.strip()
    if not t:
        return None
    start = t.find("[")
    end = t.rfind("]")
    if start != -1 and end != -1 and end > start:
        frag = t[start : end + 1]
        try:
            return json.loads(frag)
        except json.JSONDecodeError:
            pass
    start = t.find("{")
    end = t.rfind("}")
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


def _doubao_from_env(
    *,
    model_fallback: str,
    base_url_fallback: str,
) -> DoubaoConfig:
    api_key = os.environ.get("DOUBAO_API_KEY") or os.environ.get("ARK_API_KEY") or ""
    base_url = os.environ.get("DOUBAO_BASE_URL") or os.environ.get("ARK_BASE_URL") or base_url_fallback
    model = os.environ.get("DOUBAO_MODEL") or os.environ.get("ARK_MODEL") or model_fallback
    temperature = float(os.environ.get("DOUBAO_TEMPERATURE") or "0")
    max_tokens = int(os.environ.get("DOUBAO_MAX_TOKENS") or "512")
    connect_timeout_s = float(os.environ.get("DOUBAO_CONNECT_TIMEOUT_S") or "10")
    read_timeout_s = float(os.environ.get("DOUBAO_READ_TIMEOUT_S") or "120")
    api_mode = os.environ.get("DOUBAO_API_MODE") or os.environ.get("ARK_API_MODE") or "responses"
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


def _doubao_responses_api(
    session: requests.Session,
    cfg: DoubaoConfig,
    *,
    system: str,
    user: str,
) -> str:
    url = f"{cfg.base_url}/responses"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
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


def _doubao_chat_completions_api(
    session: requests.Session,
    cfg: DoubaoConfig,
    *,
    system: str,
    user: str,
) -> str:
    url = f"{cfg.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
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


def _doubao_infer(
    session: requests.Session,
    cfg: DoubaoConfig,
    *,
    system: str,
    user: str,
) -> str:
    if cfg.api_mode == "chat_completions":
        return _doubao_chat_completions_api(session, cfg, system=system, user=user)
    try:
        return _doubao_responses_api(session, cfg, system=system, user=user)
    except requests.HTTPError as e:
        resp = getattr(e, "response", None)
        if resp is not None and resp.status_code in (404, 405):
            return _doubao_chat_completions_api(session, cfg, system=system, user=user)
        raise


def _build_method_prompt(doc: Dict[str, Any], *, max_chars: int) -> Tuple[str, str]:
    title = doc.get("title") if isinstance(doc.get("title"), str) else ""
    text = doc.get("text") if isinstance(doc.get("text"), str) else ""
    text = _normalize_ws(f"{title}\n\n{text}")
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    system = "You extract entity names from 3D vision text. Output must be strict JSON."
    user = (
        "Extract method/model/framework names that are relevant to 3D vision from the text.\n"
        "Return a JSON array of strings. Each string is a concise name, no explanations.\n"
        "Text:\n"
        f"{text}"
    )
    return system, user


def _load_processed_doc_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    out: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = obj.get("doc_id")
            if isinstance(doc_id, str) and doc_id:
                out.add(doc_id)
    return out


def _load_llm_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    cache: Dict[str, Dict[str, Any]] = {}
    for it in _iter_jsonl(path):
        doc_id = it.get("doc_id")
        if isinstance(doc_id, str) and doc_id:
            cache[doc_id] = it
    return cache


def _append_llm_cache(path: Path, item: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--docs_path",
        type=str,
        default="/data/litengmo/ml-test-1/zstp_final/data/preprocessed/text/documents.jsonl",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/data/litengmo/ml-test-1/zstp_final/data/preprocessed/text/entities",
    )
    ap.add_argument(
        "--dict_dir",
        type=str,
        default="/data/litengmo/ml-test-1/zstp_final/data/preprocessed/text/dicts",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_doc_chars", type=int, default=12000)
    ap.add_argument("--min_doc_chars", type=int, default=80)
    ap.add_argument("--max_dict_hits_each", type=int, default=30)
    ap.add_argument("--enable_llm", action="store_true")
    ap.add_argument("--llm_max_chars", type=int, default=3500)
    ap.add_argument("--llm_sleep_s", type=float, default=0.2)
    ap.add_argument("--llm_cache_path", type=str, default="")
    ap.add_argument("--doubao_model", type=str, default="doubao-seed-2-0-pro-260215")
    ap.add_argument("--doubao_base_url", type=str, default="https://ark.cn-beijing.volces.com/api/v3")
    ap.add_argument("--doubao_api_mode", type=str, default="responses", choices=["responses", "chat_completions"])
    args = ap.parse_args()

    random.seed(args.seed)

    docs_path = Path(args.docs_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dict_dir = Path(args.dict_dir).resolve()
    dataset_terms = _load_dict_terms(dict_dir / "datasets.txt")
    venue_terms = _load_dict_terms(dict_dir / "venues.txt")
    metric_terms = _load_dict_terms(dict_dir / "metrics.txt")
    task_terms = _load_dict_terms(dict_dir / "tasks.txt")

    dataset_pats = _compile_term_patterns(dataset_terms)
    venue_pats = _compile_term_patterns(venue_terms)
    metric_pats = _compile_term_patterns(metric_terms, case_sensitive_all_caps_leq=3)
    task_pats = _compile_term_patterns(task_terms)

    doc_entities_path = out_dir / "doc_entities.jsonl"
    entity_registry_path = out_dir / "entity_registry_seed.jsonl"
    stats_path = out_dir / "stats.json"

    if args.overwrite:
        for p in [doc_entities_path, entity_registry_path, stats_path]:
            if p.exists():
                p.unlink()

    processed = _load_processed_doc_ids(doc_entities_path) if args.resume else set()

    llm_cache_path = Path(args.llm_cache_path).resolve() if args.llm_cache_path else (out_dir / "llm_cache.jsonl")
    llm_cache = _load_llm_cache(llm_cache_path)

    llm_cfg = _doubao_from_env(model_fallback=args.doubao_model, base_url_fallback=args.doubao_base_url)
    llm_cfg.api_mode = os.environ.get("DOUBAO_API_MODE") or os.environ.get("ARK_API_MODE") or args.doubao_api_mode
    if args.enable_llm and not llm_cfg.api_key:
        raise SystemExit("Missing DOUBAO_API_KEY (or ARK_API_KEY) in environment.")

    session = requests.Session()

    uniq: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if entity_registry_path.exists() and args.resume:
        for it in _iter_jsonl(entity_registry_path):
            typ = it.get("type")
            canonical = it.get("canonical")
            if isinstance(typ, str) and isinstance(canonical, str) and typ and canonical:
                uniq[(typ, canonical)] = it

    type_counter = Counter()
    cat_counter = Counter()
    n_docs = 0

    def add_entity(e: Dict[str, Any]) -> None:
        typ = e.get("type")
        canonical = e.get("canonical")
        if not isinstance(typ, str) or not isinstance(canonical, str) or not typ or not canonical:
            return
        k = (typ, canonical)
        if k not in uniq:
            uniq[k] = {"type": typ, "canonical": canonical, "created_at": _utc_now_iso()}

    total_docs = _count_lines(docs_path)
    pbar = _tqdm(total=total_docs, desc="entities:docs") if _tqdm is not None else None
    for doc in _iter_jsonl(docs_path):
        if pbar is not None:
            pbar.update(1)
        doc_id = doc.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            continue
        if doc_id in processed:
            continue

        text = doc.get("text")
        if not isinstance(text, str) or len(text) < args.min_doc_chars:
            continue
        if len(text) > args.max_doc_chars:
            doc["text"] = text[: args.max_doc_chars].rsplit(" ", 1)[0]

        entities: List[Dict[str, Any]] = []
        entities.extend(_extract_structured_entities(doc))
        entities.extend(
            _extract_dict_entities(
                doc,
                dataset_pats=dataset_pats,
                venue_pats=venue_pats,
                metric_pats=metric_pats,
                task_pats=task_pats,
                max_hits_each=args.max_dict_hits_each,
            )
        )

        if args.enable_llm:
            cached = llm_cache.get(doc_id)
            if cached and cached.get("ok") is True:
                parsed = cached.get("parsed")
            else:
                system, user = _build_method_prompt(doc, max_chars=args.llm_max_chars)
                prompt_sha = _sha256_text(system + "\n" + user)
                raw = ""
                ok = False
                parsed: Any = None
                err = ""
                try:
                    raw = _doubao_infer(session, llm_cfg, system=system, user=user)
                    parsed = _extract_first_json(raw)
                    ok = isinstance(parsed, list)
                    if not ok:
                        err = "parse_failed"
                except Exception as e:
                    err = f"{type(e).__name__}:{str(e)[:200]}"
                item = {
                    "doc_id": doc_id,
                    "model": llm_cfg.model,
                    "base_url": llm_cfg.base_url,
                    "prompt_sha256": prompt_sha,
                    "ok": ok,
                    "error": err,
                    "raw": raw,
                    "parsed": parsed if ok else None,
                    "created_at": _utc_now_iso(),
                }
                _append_llm_cache(llm_cache_path, item)
                llm_cache[doc_id] = item
                time.sleep(max(0.0, float(args.llm_sleep_s)))

            if isinstance(parsed, list):
                for name in parsed[:80]:
                    if not isinstance(name, str):
                        continue
                    name = _normalize_ws(name)
                    if not name or len(name) > 80:
                        continue
                    entities.append({"type": "Method", "mention": name, "canonical": _canonical_key(name), "source": "llm"})

        seen = set()
        deduped: List[Dict[str, Any]] = []
        for e in entities:
            typ = e.get("type")
            canonical = e.get("canonical")
            if not isinstance(typ, str) or not isinstance(canonical, str):
                continue
            k = (typ, canonical)
            if k in seen:
                continue
            seen.add(k)
            deduped.append(e)
            add_entity(e)
            type_counter[typ] += 1

        cat = doc.get("category")
        if isinstance(cat, str) and cat:
            cat_counter[cat] += 1

        out_item = {
            "doc_id": doc_id,
            "doc_type": doc.get("doc_type"),
            "source": doc.get("source"),
            "category": doc.get("category"),
            "category_zh": doc.get("category_zh"),
            "title": doc.get("title"),
            "text_sha256": doc.get("text_sha256"),
            "entities": deduped,
            "created_at": _utc_now_iso(),
        }
        _append_jsonl(doc_entities_path, [out_item])
        n_docs += 1
        if pbar is not None and (n_docs % 50 == 0):
            pbar.set_postfix(processed=n_docs, registry=len(uniq))
        if args.limit and n_docs >= args.limit:
            break
    if pbar is not None:
        pbar.set_postfix(processed=n_docs, registry=len(uniq))
        pbar.close()

    if args.overwrite or not entity_registry_path.exists():
        uniq_items = list(uniq.values())
        uniq_items.sort(key=lambda x: (x.get("type", ""), x.get("canonical", "")))
        with entity_registry_path.open("w", encoding="utf-8") as f:
            for it in uniq_items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    else:
        new_items = []
        for k, it in uniq.items():
            new_items.append(it)
        new_items.sort(key=lambda x: (x.get("type", ""), x.get("canonical", "")))
        with entity_registry_path.open("w", encoding="utf-8") as f:
            for it in new_items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    stats = {
        "created_at": _utc_now_iso(),
        "docs_path": str(docs_path),
        "out_dir": str(out_dir),
        "doc_entities_path": str(doc_entities_path),
        "entity_registry_seed_path": str(entity_registry_path),
        "doc_count_processed": n_docs,
        "entity_registry_seed_count": len(uniq),
        "entity_mentions_by_type": dict(type_counter),
        "docs_by_category": dict(cat_counter),
        "llm_enabled": bool(args.enable_llm),
        "llm_cache_path": str(llm_cache_path),
    }
    _write_json(stats_path, stats)
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
