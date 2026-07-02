from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _to_int(x: str) -> Optional[int]:
    s = str(x or "").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _to_float(x: str) -> Optional[float]:
    s = str(x or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _cypher_str(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _cypher_val(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return _cypher_str(str(v))


def _read_csv(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {k: (v if v is not None else "") for k, v in row.items()}


def _chunk(xs: List[Any], n: int) -> Iterable[List[Any]]:
    if n <= 0:
        n = 200
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _emit_unwind_rows(rows: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    parts.append("UNWIND [")
    for i, r in enumerate(rows):
        kvs = ",".join(f"{k}:{_cypher_val(v)}" for k, v in r.items())
        comma = "," if i < len(rows) - 1 else ""
        parts.append(f"  {{{kvs}}}{comma}")
    parts.append("] AS row")
    return "\n".join(parts)


def _build_node_block(rows: List[Dict[str, Any]]) -> str:
    q = _emit_unwind_rows(rows)
    q += "\nMERGE (n:Entity {id: row.id})\n"
    q += "SET n.name = row.label,\n"
    q += "    n.type = row.type,\n"
    q += "    n.citation_count = row.citation_count,\n"
    q += "    n.repo_stars = row.repo_stars,\n"
    q += "    n.year = row.year\n"
    q += "FOREACH (_ IN CASE WHEN row.type = 'Paper' THEN [1] ELSE [] END | SET n:Paper)\n"
    q += "FOREACH (_ IN CASE WHEN row.type = 'Method' THEN [1] ELSE [] END | SET n:Method)\n"
    q += "FOREACH (_ IN CASE WHEN row.type = 'Repo' THEN [1] ELSE [] END | SET n:Repo)\n"
    q += "FOREACH (_ IN CASE WHEN row.type = 'Dataset' THEN [1] ELSE [] END | SET n:Dataset)\n"
    q += ";\n"
    return q


def _build_edge_block(rel_type: str, rows: List[Dict[str, Any]]) -> str:
    q = _emit_unwind_rows(rows)
    q += "\nMATCH (h:Entity {id: row.h})\n"
    q += "MATCH (t:Entity {id: row.t})\n"
    q += f"MERGE (h)-[r:{rel_type}]->(t)\n"
    q += "SET r.weight = row.weight\n"
    q += ";\n"
    return q


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes_csv", type=str, default="data/vis/gephi/nodes.csv")
    ap.add_argument("--edges_csv", type=str, default="data/vis/gephi/edges.csv")
    ap.add_argument("--out_cypher", type=str, default="data/vis/neo4j/aura_gephi_import.cypher")
    ap.add_argument("--node_chunk", type=int, default=200)
    ap.add_argument("--edge_chunk", type=int, default=300)
    args = ap.parse_args()

    nodes_csv = (repo_root / str(args.nodes_csv)).resolve() if not Path(str(args.nodes_csv)).is_absolute() else Path(str(args.nodes_csv)).resolve()
    edges_csv = (repo_root / str(args.edges_csv)).resolve() if not Path(str(args.edges_csv)).is_absolute() else Path(str(args.edges_csv)).resolve()
    out_cypher = (repo_root / str(args.out_cypher)).resolve() if not Path(str(args.out_cypher)).is_absolute() else Path(str(args.out_cypher)).resolve()
    out_cypher.parent.mkdir(parents=True, exist_ok=True)

    nodes: List[Dict[str, Any]] = []
    for row in _read_csv(nodes_csv):
        nodes.append(
            {
                "id": row.get("Id", "").strip(),
                "label": row.get("Label", "").strip(),
                "type": row.get("Type", "").strip(),
                "citation_count": _to_int(row.get("citation_count", "")),
                "repo_stars": _to_int(row.get("repo_stars", "")),
                "year": _to_int(row.get("year", "")),
            }
        )
    nodes = [n for n in nodes if n.get("id")]

    edges_by_rel: Dict[str, List[Dict[str, Any]]] = {}
    for row in _read_csv(edges_csv):
        rel = row.get("Label", "").strip()
        if not rel:
            continue
        edges_by_rel.setdefault(rel, []).append(
            {
                "h": row.get("Source", "").strip(),
                "t": row.get("Target", "").strip(),
                "weight": _to_float(row.get("Weight", "")),
            }
        )

    lines: List[str] = []
    lines.append("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;")
    lines.append("")

    for chunk in _chunk(nodes, int(args.node_chunk)):
        lines.append(_build_node_block(chunk).rstrip())
        lines.append("")

    for rel in sorted(edges_by_rel.keys()):
        rel_rows = [e for e in edges_by_rel[rel] if e.get("h") and e.get("t")]
        for chunk in _chunk(rel_rows, int(args.edge_chunk)):
            lines.append(_build_edge_block(rel, chunk).rstrip())
            lines.append("")

    out_cypher.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(str(out_cypher))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
