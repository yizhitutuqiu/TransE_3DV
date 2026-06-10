"""
Collect 3D-vision text corpus for downstream IE/KG training.

Sources:
- arXiv API (Atom): title/abstract/authors/links/categories
- GitHub REST API: repository metadata + raw README

Default output:
  <repo>/data/raw/
    paper/{pose_estimation,3d_generation,4d_reconstruction}/items.jsonl
    readme/{pose_estimation,3d_generation,4d_reconstruction}/items.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import random
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


ARXIV_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _progress(total: int, *, desc: str):
    if _tqdm is None:
        return None
    return _tqdm(total=total, desc=desc)


def _append_jsonl_line(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def _read_jsonl_keys(path: Path, key_field: str) -> Set[str]:
    if not path.exists():
        return set()
    keys: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = obj.get(key_field)
            if isinstance(v, str) and v:
                keys.add(v)
    return keys


def _append_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
            n += 1
    return n


def semantic_scholar_headers() -> Dict[str, str]:
    h: Dict[str, str] = {}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if api_key:
        h["x-api-key"] = api_key
    return h


def _normalize_arxiv_id_text(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    return re.sub(r"v\d+$", "", s)


def semantic_scholar_fetch_paper_by_arxiv(
    session: requests.Session,
    *,
    arxiv_id: str,
    timeout_s: int,
    connect_timeout_s: int,
    sleep_s: float,
) -> Dict[str, Any]:
    if sleep_s > 0:
        time.sleep(float(sleep_s))
    aid = _normalize_arxiv_id_text(arxiv_id)
    url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{urllib.parse.quote(aid)}"
    params = {
        "fields": "citationCount,referenceCount,references.externalIds,references.paperId",
    }
    resp = _request_with_backoff(
        session,
        "GET",
        url,
        headers=semantic_scholar_headers(),
        params=params,
        timeout_s=timeout_s,
        connect_timeout_s=connect_timeout_s,
        max_retries=6,
        base_sleep_s=1.5,
    )
    if resp.status_code == 404:
        return {}
    resp.raise_for_status()
    parsed = resp.json()
    obj = parsed if isinstance(parsed, dict) else {}
    out: Dict[str, Any] = {}
    if isinstance(obj.get("citationCount"), int):
        out["citation_count"] = int(obj["citationCount"])
    if isinstance(obj.get("referenceCount"), int):
        out["reference_count"] = int(obj["referenceCount"])
    refs = obj.get("references")
    arxiv_refs: List[str] = []
    if isinstance(refs, list):
        for r in refs:
            if not isinstance(r, dict):
                continue
            ext = r.get("externalIds")
            if not isinstance(ext, dict):
                continue
            arx = ext.get("ArXiv") or ext.get("arXiv") or ext.get("arxiv")
            if isinstance(arx, str) and arx.strip():
                arxiv_refs.append(_normalize_arxiv_id_text(arx))
    if arxiv_refs:
        uniq: List[str] = []
        seen: Set[str] = set()
        for x in arxiv_refs:
            if x and x not in seen:
                seen.add(x)
                uniq.append(x)
        out["references_arxiv"] = uniq
    return out


def semantic_scholar_refresh_items_jsonl(
    session: requests.Session,
    *,
    items_path: Path,
    timeout_s: int,
    connect_timeout_s: int,
    sleep_s: float,
) -> Dict[str, int]:
    if not items_path.exists():
        return {"scanned": 0, "updated": 0}
    tmp = items_path.with_suffix(items_path.suffix + ".tmp")
    scanned = 0
    updated = 0
    total = 0
    with items_path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            total += 1
    fin = items_path.open("r", encoding="utf-8", errors="ignore")
    try:
        it = fin
        if _tqdm is not None:
            it = _tqdm(it, total=total, desc=f"s2:refresh:{items_path.parent.name}")
        with tmp.open("w", encoding="utf-8") as fout:
            for line in it:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                scanned += 1
                aid = obj.get("arxiv_id")
                need = ("citation_count" not in obj) or ("references_arxiv" not in obj)
                if need and isinstance(aid, str) and aid.strip():
                    extra = semantic_scholar_fetch_paper_by_arxiv(
                        session,
                        arxiv_id=aid,
                        timeout_s=timeout_s,
                        connect_timeout_s=connect_timeout_s,
                        sleep_s=sleep_s,
                    )
                    if extra:
                        obj.update(extra)
                        updated += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        fin.close()
    tmp.replace(items_path)
    return {"scanned": scanned, "updated": updated}


def _request_with_backoff(
    session: requests.Session,
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout_s: int = 30,
    connect_timeout_s: int = 10,
    max_retries: int = 6,
    base_sleep_s: float = 1.5,
) -> requests.Response:
    last_exc: Optional[Exception] = None
    last_status: Optional[int] = None
    last_text: str = ""
    for i in range(max_retries):
        try:
            resp = session.request(
                method,
                url,
                headers=headers,
                params=params,
                timeout=(connect_timeout_s, timeout_s),
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                last_status = int(resp.status_code)
                try:
                    last_text = (resp.text or "")[:300]
                except Exception:
                    last_text = ""
                retry_after = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
                if isinstance(retry_after, str) and retry_after.strip().isdigit():
                    sleep_s = float(retry_after.strip())
                else:
                    sleep_s = base_sleep_s * (2**i)
                time.sleep(min(sleep_s, 30))
                continue
            return resp
        except Exception as e:
            last_exc = e
            sleep_s = base_sleep_s * (2**i)
            time.sleep(min(sleep_s, 30))
            continue
    if last_exc is not None:
        raise last_exc
    if last_status is not None:
        raise RuntimeError(f"request failed after retries: status={last_status} url={url} body={last_text!r}")
    raise RuntimeError(f"request failed after retries: url={url}")


@dataclasses.dataclass(frozen=True)
class CategoryConfig:
    slug: str
    zh_name: str
    arxiv_queries: Sequence[str]
    github_queries: Sequence[str]


def build_category_configs(github_min_stars: int) -> List[CategoryConfig]:
    base_cv_cats = "(cat:cs.CV OR cat:cs.GR OR cat:cs.RO)"

    def q_all(expr: str) -> str:
        return f"({expr}) AND {base_cv_cats}"

    pose_arxiv = [
        q_all('ti:"pose estimation" OR abs:"pose estimation"'),
        q_all('ti:"camera pose" OR abs:"camera pose" OR ti:"relative pose" OR abs:"relative pose"'),
        q_all('ti:"6d pose" OR abs:"6d pose" OR ti:"object pose" OR abs:"object pose"'),
        q_all('ti:"3d human pose" OR abs:"3d human pose" OR ti:"human pose" OR abs:"human pose"'),
    ]

    gen_arxiv = [
        q_all('ti:"NeRF" OR abs:"NeRF" OR ti:"neural radiance field" OR abs:"neural radiance field"'),
        q_all('ti:"3d gaussian splatting" OR abs:"3d gaussian splatting" OR ti:"gaussian splatting" OR abs:"gaussian splatting"'),
        q_all('ti:"text-to-3d" OR abs:"text-to-3d" OR ti:"3d diffusion" OR abs:"3d diffusion" OR ti:"3d generation" OR abs:"3d generation"'),
        q_all('ti:"implicit representation" OR abs:"implicit representation" OR ti:"signed distance function" OR abs:"signed distance function"'),
    ]

    rec_arxiv = [
        q_all('ti:"3d reconstruction" OR abs:"3d reconstruction" OR ti:"scene reconstruction" OR abs:"scene reconstruction"'),
        q_all('ti:"structure from motion" OR abs:"structure from motion" OR ti:"sfm" OR abs:"sfm"'),
        q_all('ti:"multi-view stereo" OR abs:"multi-view stereo" OR ti:"mvs" OR abs:"mvs"'),
        q_all('ti:"slam" OR abs:"slam" OR ti:"visual odometry" OR abs:"visual odometry"'),
        q_all('ti:"point cloud" OR abs:"point cloud" OR ti:"depth estimation" OR abs:"depth estimation"'),
        q_all('ti:"dynamic scene" OR abs:"dynamic scene" OR ti:"4d reconstruction" OR abs:"4d reconstruction"'),
    ]

    s = f"stars:>={github_min_stars}"

    pose_gh = [
        f'"pose estimation" {s} (topic:computer-vision OR topic:pose-estimation)',
        f'"6d pose" {s} (topic:computer-vision OR topic:robotics)',
        f'"camera pose" {s} topic:computer-vision',
        f'"3d human pose" {s} (topic:computer-vision OR topic:human-pose-estimation)',
    ]

    gen_gh = [
        f'NeRF {s} (topic:nerf OR topic:computer-vision)',
        f'"gaussian splatting" {s} (topic:computer-vision OR topic:graphics)',
        f'"text-to-3d" {s} (topic:diffusion OR topic:computer-vision)',
        f'"3d diffusion" {s} (topic:diffusion OR topic:computer-vision)',
    ]

    rec_gh = [
        f'"3d reconstruction" {s} (topic:slam OR topic:computer-vision OR topic:photogrammetry)',
        f'SfM {s} (topic:slam OR topic:photogrammetry OR topic:computer-vision)',
        f'MVS {s} (topic:mvs OR topic:computer-vision OR topic:photogrammetry)',
        f'SLAM {s} (topic:slam OR topic:robotics)',
    ]

    return [
        CategoryConfig("pose_estimation", "姿态估计", pose_arxiv, pose_gh),
        CategoryConfig("3d_generation", "3D生成", gen_arxiv, gen_gh),
        CategoryConfig("4d_reconstruction", "4D重建", rec_arxiv, rec_gh),
    ]


def _normalize_arxiv_id_from_entry_id(entry_id_text: str) -> Tuple[str, str]:
    m = re.search(r"arxiv\.org/abs/([^/]+)$", entry_id_text.strip())
    raw = m.group(1) if m else entry_id_text.strip()
    base = re.sub(r"v\d+$", "", raw)
    return raw, base


def parse_arxiv_atom(xml_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    items: List[Dict[str, Any]] = []
    for e in root.findall("a:entry", namespaces=ARXIV_NS):
        entry_id_text = (e.findtext("a:id", default="", namespaces=ARXIV_NS) or "").strip()
        arxiv_id_raw, arxiv_id = _normalize_arxiv_id_from_entry_id(entry_id_text)

        title = (e.findtext("a:title", default="", namespaces=ARXIV_NS) or "").strip()
        summary = (e.findtext("a:summary", default="", namespaces=ARXIV_NS) or "").strip()
        summary = re.sub(r"\s+", " ", summary)

        published = (e.findtext("a:published", default="", namespaces=ARXIV_NS) or "").strip()
        updated = (e.findtext("a:updated", default="", namespaces=ARXIV_NS) or "").strip()

        authors = []
        for a in e.findall("a:author", namespaces=ARXIV_NS):
            nm = (a.findtext("a:name", default="", namespaces=ARXIV_NS) or "").strip()
            if nm:
                authors.append(nm)

        categories = []
        for c in e.findall("a:category", namespaces=ARXIV_NS):
            term = c.attrib.get("term", "").strip()
            if term:
                categories.append(term)

        primary_category = ""
        pc = e.find("arxiv:primary_category", namespaces=ARXIV_NS)
        if pc is not None:
            primary_category = (pc.attrib.get("term") or "").strip()

        links: Dict[str, str] = {}
        for l in e.findall("a:link", namespaces=ARXIV_NS):
            href = (l.attrib.get("href") or "").strip()
            rel = (l.attrib.get("rel") or "").strip()
            typ = (l.attrib.get("type") or "").strip()
            if not href:
                continue
            if href.endswith(".pdf") or "/pdf/" in href:
                links.setdefault("pdf", href)
            if rel == "alternate":
                links.setdefault("abs", href)
            if typ == "application/rss+xml":
                links.setdefault("rss", href)

        items.append(
            {
                "arxiv_id": arxiv_id,
                "arxiv_id_raw": arxiv_id_raw,
                "title": title,
                "summary": summary,
                "authors": authors,
                "published": published,
                "updated": updated,
                "primary_category": primary_category,
                "categories": categories,
                "links": links,
            }
        )
    return items


def fetch_arxiv(
    session: requests.Session,
    *,
    query: str,
    max_results: int,
    start: int,
    user_agent: str,
    base_urls: Sequence[str],
    timeout_s: int,
    connect_timeout_s: int,
) -> List[Dict[str, Any]]:
    params = {
        "search_query": query,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    last_exc: Optional[Exception] = None
    for url in base_urls:
        try:
            resp = _request_with_backoff(
                session,
                "GET",
                url,
                params=params,
                headers={"User-Agent": user_agent},
                timeout_s=timeout_s,
                connect_timeout_s=connect_timeout_s,
            )
            resp.raise_for_status()
            return parse_arxiv_atom(resp.text)
        except Exception as e:
            last_exc = e
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("arxiv request failed but no exception captured")


def _arxiv_year_range_filter(year: int) -> str:
    return f"submittedDate:[{year}01010000 TO {year}12312359]"


def crawl_arxiv_for_category(
    session: requests.Session,
    *,
    cfg: CategoryConfig,
    out_dir: Path,
    max_items: int,
    per_page: int,
    sleep_s: float,
    year_start: Optional[int],
    year_end: Optional[int],
    rng: random.Random,
    arxiv_pool_mult: int,
    request_timeout_s: int,
    connect_timeout_s: int,
    arxiv_base_urls: Sequence[str],
    dry_run: bool,
    enable_semantic_scholar: bool,
    pbar: Any,
) -> int:
    out_path = out_dir / "items.jsonl"
    _safe_mkdir(out_dir)
    existing = _read_jsonl_keys(out_path, "arxiv_id")

    user_agent = "zstp_final-corpus-crawler/1.0 (mailto:local)"
    appended = 0

    if max_items <= 0:
        return 0

    if dry_run:
        yr = ""
        if year_start is not None and year_end is not None:
            yr = f" years={year_start}-{year_end}"
        print(f"[arxiv][{cfg.slug}] queries={len(cfg.arxiv_queries)} out={out_path}{yr}")
        return 0

    if year_start is not None and year_end is not None:
        if year_start > year_end:
            raise ValueError("--year_start must be <= --year_end")
        years = list(range(year_start, year_end + 1))
        per_year = max(1, max_items // max(len(years), 1))
        remainder = max(0, max_items - per_year * len(years))

        for yi, year in enumerate(years):
            year_budget = per_year + (1 if yi < remainder else 0)
            if year_budget <= 0:
                continue

            pool_target = max(1, year_budget * max(1, arxiv_pool_mult))
            candidates: List[Dict[str, Any]] = []
            cand_ids: Set[str] = set()

            for q in cfg.arxiv_queries:
                if len(candidates) >= pool_target:
                    break
                qy = f"({q}) AND {_arxiv_year_range_filter(year)}"
                start = 0
                while len(candidates) < pool_target:
                    batch_size = min(per_page, max(1, pool_target - len(candidates)))
                    items = fetch_arxiv(
                        session,
                        query=qy,
                        max_results=batch_size,
                        start=start,
                        user_agent=user_agent,
                        base_urls=arxiv_base_urls,
                        timeout_s=request_timeout_s,
                        connect_timeout_s=connect_timeout_s,
                    )
                    if not items:
                        break
                    for it in items:
                        aid = it.get("arxiv_id")
                        if not isinstance(aid, str) or not aid:
                            continue
                        if aid in existing or aid in cand_ids:
                            continue
                        it["_query"] = qy
                        it["_year"] = year
                        candidates.append(it)
                        cand_ids.add(aid)
                        if len(candidates) >= pool_target:
                            break
                    start += batch_size
                    time.sleep(max(sleep_s, 0.0))

            if not candidates:
                continue

            k = min(year_budget, len(candidates))
            picked = rng.sample(candidates, k=k) if k < len(candidates) else candidates
            ts = _utc_now_iso()
            for it in picked:
                aid = it["arxiv_id"]
                if aid in existing:
                    continue
                it.update(
                    {
                        "source": "arxiv",
                        "category": cfg.slug,
                        "category_zh": cfg.zh_name,
                        "query": it.pop("_query", ""),
                        "retrieved_at": ts,
                        "year": it.pop("_year", year),
                    }
                )
                if enable_semantic_scholar:
                    extra = semantic_scholar_fetch_paper_by_arxiv(
                        session,
                        arxiv_id=aid,
                        timeout_s=request_timeout_s,
                        connect_timeout_s=connect_timeout_s,
                        sleep_s=1.0,
                    )
                    if extra:
                        it.update(extra)
                _append_jsonl_line(out_path, it)
                appended += 1
                if pbar is not None:
                    pbar.update(1)
                existing.add(aid)
    else:
        remaining = max_items
        for q in cfg.arxiv_queries:
            if remaining <= 0:
                break
            start = 0
            while remaining > 0:
                batch_size = min(per_page, remaining)
                items = fetch_arxiv(
                    session,
                    query=q,
                    max_results=batch_size,
                    start=start,
                    user_agent=user_agent,
                    base_urls=arxiv_base_urls,
                    timeout_s=request_timeout_s,
                    connect_timeout_s=connect_timeout_s,
                )
                if not items:
                    break
                ts = _utc_now_iso()
                for it in items:
                    if it["arxiv_id"] in existing:
                        continue
                    it.update(
                        {
                            "source": "arxiv",
                            "category": cfg.slug,
                            "category_zh": cfg.zh_name,
                            "query": q,
                            "retrieved_at": ts,
                        }
                    )
                    if enable_semantic_scholar:
                        extra = semantic_scholar_fetch_paper_by_arxiv(
                            session,
                            arxiv_id=str(it.get("arxiv_id") or ""),
                            timeout_s=request_timeout_s,
                            connect_timeout_s=connect_timeout_s,
                            sleep_s=1.0,
                        )
                        if extra:
                            it.update(extra)
                    _append_jsonl_line(out_path, it)
                    appended += 1
                    if pbar is not None:
                        pbar.update(1)
                    existing.add(it["arxiv_id"])
                    remaining -= 1
                    if remaining <= 0:
                        break
                start += batch_size
                time.sleep(max(sleep_s, 0.0))

    return appended


def github_headers() -> Dict[str, str]:
    h = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def github_raw_readme_headers() -> Dict[str, str]:
    h = {"Accept": "application/vnd.github.raw"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def github_search_repos(
    session: requests.Session,
    *,
    query: str,
    per_page: int,
    page: int,
) -> Dict[str, Any]:
    url = "https://api.github.com/search/repositories"
    params = {"q": query, "per_page": str(per_page), "page": str(page)}
    resp = _request_with_backoff(session, "GET", url, params=params, headers=github_headers())
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        reset = resp.headers.get("X-RateLimit-Reset")
        if reset and reset.isdigit():
            sleep_s = max(0, int(reset) - int(time.time()) + 2)
            time.sleep(min(sleep_s, 120))
            resp = _request_with_backoff(session, "GET", url, params=params, headers=github_headers())
    resp.raise_for_status()
    return resp.json()


def github_fetch_readme(
    session: requests.Session,
    *,
    full_name: str,
) -> Optional[str]:
    url = f"https://api.github.com/repos/{full_name}/readme"
    resp = _request_with_backoff(session, "GET", url, headers=github_raw_readme_headers())
    if resp.status_code == 404:
        return None
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        reset = resp.headers.get("X-RateLimit-Reset")
        if reset and reset.isdigit():
            sleep_s = max(0, int(reset) - int(time.time()) + 2)
            time.sleep(min(sleep_s, 120))
            resp = _request_with_backoff(session, "GET", url, headers=github_raw_readme_headers())
    resp.raise_for_status()
    return resp.text


def crawl_github_for_category(
    session: requests.Session,
    *,
    cfg: CategoryConfig,
    out_dir: Path,
    max_items: int,
    per_page: int,
    sleep_s: float,
    max_readme_chars: int,
    dry_run: bool,
    pbar: Any,
) -> int:
    out_path = out_dir / "items.jsonl"
    _safe_mkdir(out_dir)
    existing = _read_jsonl_keys(out_path, "repo_full_name")

    if dry_run:
        print(f"[github][{cfg.slug}] queries={len(cfg.github_queries)} out={out_path}")
        return 0

    if max_items <= 0:
        return 0

    remaining = max_items
    appended = 0

    for q in cfg.github_queries:
        if remaining <= 0:
            break
        page = 1
        while remaining > 0:
            data = github_search_repos(session, query=q, per_page=per_page, page=page)
            repos = data.get("items") or []
            if not repos:
                break
            ts = _utc_now_iso()
            for r in repos:
                full_name = (r.get("full_name") or "").strip()
                if not full_name or full_name in existing:
                    continue
                readme_text = github_fetch_readme(session, full_name=full_name)
                if not readme_text:
                    continue
                if len(readme_text) > max_readme_chars:
                    readme_text = readme_text[:max_readme_chars]
                item = {
                    "source": "github",
                    "category": cfg.slug,
                    "category_zh": cfg.zh_name,
                    "query": q,
                    "retrieved_at": ts,
                    "repo_full_name": full_name,
                    "repo_html_url": r.get("html_url"),
                    "repo_description": r.get("description"),
                    "repo_language": r.get("language"),
                    "repo_stargazers_count": r.get("stargazers_count"),
                    "repo_forks_count": r.get("forks_count"),
                    "repo_open_issues_count": r.get("open_issues_count"),
                    "repo_default_branch": r.get("default_branch"),
                    "repo_topics": r.get("topics") or [],
                    "repo_license": (r.get("license") or {}).get("spdx_id") if isinstance(r.get("license"), dict) else None,
                    "repo_created_at": r.get("created_at"),
                    "repo_updated_at": r.get("updated_at"),
                    "readme_text": readme_text,
                    "readme_sha256": _sha256_text(readme_text),
                }
                _append_jsonl_line(out_path, item)
                appended += 1
                if pbar is not None:
                    pbar.update(1)
                existing.add(full_name)
                remaining -= 1
                if remaining <= 0:
                    break
                time.sleep(max(sleep_s, 0.0))

            total_count = int(data.get("total_count") or 0)
            if page * per_page >= min(total_count, 1000):
                break
            page += 1

    return appended


def main() -> int:
    default_out_root = str((Path(__file__).resolve().parents[1] / "raw"))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_root",
        type=str,
        default=default_out_root,
        help="Output root containing paper/ and readme/",
    )
    ap.add_argument("--mode", choices=["all", "paper", "readme"], default="all")
    ap.add_argument("--category", choices=["all", "pose_estimation", "3d_generation", "4d_reconstruction"], default="all")
    ap.add_argument("--max_papers", type=int, default=400)
    ap.add_argument("--max_repos", type=int, default=200)
    ap.add_argument("--arxiv_per_page", type=int, default=100)
    ap.add_argument("--github_per_page", type=int, default=50)
    ap.add_argument("--github_min_stars", type=int, default=50)
    ap.add_argument("--sleep_s", type=float, default=0.8)
    ap.add_argument("--max_readme_chars", type=int, default=200_000)
    ap.add_argument("--request_timeout_s", type=int, default=120)
    ap.add_argument("--connect_timeout_s", type=int, default=10)
    ap.add_argument("--disable_semantic_scholar", action="store_true")
    ap.add_argument("--refresh_semantic_scholar", action="store_true")
    ap.add_argument(
        "--arxiv_base_urls",
        type=str,
        default="https://export.arxiv.org/api/query,http://export.arxiv.org/api/query,https://arxiv.org/api/query,http://arxiv.org/api/query",
    )
    ap.add_argument("--year_start", type=int, default=None, help="arXiv sampling start year (inclusive), e.g. 2018")
    ap.add_argument("--year_end", type=int, default=None, help="arXiv sampling end year (inclusive), e.g. 2026")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arxiv_pool_mult", type=int, default=4, help="candidate pool size multiplier per year for random sampling")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing items.jsonl for selected mode/category")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    paper_root = out_root / "paper"
    readme_root = out_root / "readme"
    _safe_mkdir(paper_root)
    _safe_mkdir(readme_root)
    enable_s2 = not bool(args.disable_semantic_scholar)
    if enable_s2:
        print(json.dumps({"event": "semantic_scholar_enabled"}, ensure_ascii=False))

    cfgs = build_category_configs(args.github_min_stars)
    if args.category != "all":
        cfgs = [c for c in cfgs if c.slug == args.category]

    session = requests.Session()
    rng = random.Random(args.seed)
    arxiv_base_urls = [u.strip() for u in args.arxiv_base_urls.split(",") if u.strip()]

    if args.overwrite:
        for c in cfgs:
            if args.mode in ("all", "paper"):
                p = paper_root / c.slug / "items.jsonl"
                if p.exists():
                    p.unlink()
            if args.mode in ("all", "readme"):
                p = readme_root / c.slug / "items.jsonl"
                if p.exists():
                    p.unlink()

    total_papers = 0
    total_repos = 0

    if args.mode in ("all", "paper") and int(args.max_papers) > 0:
        pbar = _progress(max(0, int(args.max_papers)), desc="crawl:papers:new")
        for c in cfgs:
            n = crawl_arxiv_for_category(
                session,
                cfg=c,
                out_dir=paper_root / c.slug,
                max_items=max(0, int(args.max_papers) - total_papers),
                per_page=max(1, args.arxiv_per_page),
                sleep_s=args.sleep_s,
                year_start=args.year_start,
                year_end=args.year_end,
                rng=rng,
                arxiv_pool_mult=max(1, args.arxiv_pool_mult),
                request_timeout_s=max(1, int(args.request_timeout_s)),
                connect_timeout_s=max(1, int(args.connect_timeout_s)),
                arxiv_base_urls=arxiv_base_urls,
                dry_run=args.dry_run,
                enable_semantic_scholar=enable_s2,
                pbar=pbar,
            )
            total_papers += n
            if total_papers >= max(0, int(args.max_papers)):
                break
        if pbar is not None:
            pbar.close()
        if enable_s2 and bool(args.refresh_semantic_scholar) and not args.dry_run:
            stats = {"event": "semantic_scholar_refresh", "by_category": {}}
            for c in cfgs:
                items_path = (paper_root / c.slug / "items.jsonl").resolve()
                r = semantic_scholar_refresh_items_jsonl(
                    session,
                    items_path=items_path,
                    timeout_s=max(1, int(args.request_timeout_s)),
                    connect_timeout_s=max(1, int(args.connect_timeout_s)),
                    sleep_s=1.0,
                )
                stats["by_category"][c.slug] = r
            print(json.dumps(stats, ensure_ascii=False))
    elif enable_s2 and bool(args.refresh_semantic_scholar) and not args.dry_run and args.mode in ("all", "paper"):
        stats = {"event": "semantic_scholar_refresh", "by_category": {}}
        for c in cfgs:
            items_path = (paper_root / c.slug / "items.jsonl").resolve()
            r = semantic_scholar_refresh_items_jsonl(
                session,
                items_path=items_path,
                timeout_s=max(1, int(args.request_timeout_s)),
                connect_timeout_s=max(1, int(args.connect_timeout_s)),
                sleep_s=1.0,
            )
            stats["by_category"][c.slug] = r
        print(json.dumps(stats, ensure_ascii=False))

    if args.mode in ("all", "readme") and int(args.max_repos) > 0:
        pbar = _progress(max(0, int(args.max_repos)), desc="crawl:readmes:new")
        for c in cfgs:
            n = crawl_github_for_category(
                session,
                cfg=c,
                out_dir=readme_root / c.slug,
                max_items=max(0, int(args.max_repos) - total_repos),
                per_page=max(1, min(args.github_per_page, 100)),
                sleep_s=args.sleep_s,
                max_readme_chars=max(1, args.max_readme_chars),
                dry_run=args.dry_run,
                pbar=pbar,
            )
            total_repos += n
            if total_repos >= max(0, int(args.max_repos)):
                break
        if pbar is not None:
            pbar.close()

    print(json.dumps({"papers_appended": total_papers, "readmes_appended": total_repos}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
