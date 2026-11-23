"""把 parsed/<doc_id>.json 的块切成两种粒度。

semantic：按章节和块边界合并，文本块最多 800 字，表格/图单独成块。
hierarchical：父块最多 1600 字，子块最多 400 字；检索用子块，返回父块。

表格不切开，图块不切开；超长时在报告中记为 oversize。
"""

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
PARSED_DIR = ROOT / "parsed"
CHUNKS_DIR = ROOT / "chunks"

SEM_MAX = 800
HIER_PARENT_MAX = 1600
HIER_CHILD_MAX = 400

SENT_RE = re.compile(r"(?<=[。；！？])")


# ── 文本切分 ──────────────────────────────────────────────────────────────────


def split_text(text: str, limit: int) -> list[str]:
    """按行和句号切分，再把碎片拼回不超过 limit 的段。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    pieces: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) <= limit:
            pieces.append(line)
            continue
        for sentence in SENT_RE.split(line):
            sentence = sentence.strip()
            while len(sentence) > limit:
                pieces.append(sentence[:limit])
                sentence = sentence[limit:]
            if sentence:
                pieces.append(sentence)

    segments: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    def flush():
        nonlocal buffer, buffer_len
        if buffer:
            segments.append("\n".join(buffer))
            buffer = []
            buffer_len = 0

    for piece in pieces:
        extra = len(piece) + (1 if buffer else 0)
        if buffer and buffer_len + extra > limit:
            flush()
        buffer.append(piece)
        buffer_len += extra
        if buffer_len >= limit:
            flush()
    flush()
    return segments


# ── 公共结构 ──────────────────────────────────────────────────────────────────


def load_blocks(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["meta"], data["blocks"]


def make_chunk(
    chunk_id: str,
    doc_id: str,
    strategy: str,
    pieces: list[tuple[dict, str]],
    section_path: list[str],
    parent_id: str = "",
) -> dict:
    content = "\n\n".join(text for _, text in pieces if text.strip())
    block_ids = []
    for block, _ in pieces:
        if block["block_id"] not in block_ids:
            block_ids.append(block["block_id"])
    pages = [block["page_num"] for block, _ in pieces]
    chunk = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "strategy": strategy,
        "content": content,
        "block_ids": block_ids,
        "page_start": min(pages),
        "page_end": max(pages),
        "section_path": list(section_path),
        "block_types": sorted({block["block_type"] for block, _ in pieces}),
    }
    if parent_id:
        chunk["parent_id"] = parent_id
    return chunk


def block_piece(block: dict) -> tuple[dict, str]:
    return block, block["content"]


# ── semantic ──────────────────────────────────────────────────────────────────


def build_semantic(meta: dict, blocks: list[dict]) -> list[dict]:
    doc_id = meta["doc_id"]
    chunks: list[dict] = []
    buffer: list[tuple[dict, str]] = []
    buffer_len = 0
    buffer_section: list[str] = []
    counter = 0

    def flush():
        nonlocal buffer, buffer_len, buffer_section
        if not buffer:
            return
        nonlocal counter
        counter += 1
        chunks.append(make_chunk(
            f"{doc_id}:sem:{counter:05d}", doc_id, "semantic",
            buffer, buffer_section,
        ))
        buffer = []
        buffer_len = 0
        buffer_section = []

    for block in blocks:
        btype = block["block_type"]
        section = block["section_path"]

        if btype == "title":
            flush()
            continue

        if btype in ("table", "figure"):
            flush()
            counter += 1
            chunks.append(make_chunk(
                f"{doc_id}:sem:{counter:05d}", doc_id, "semantic",
                [block_piece(block)], section,
            ))
            continue

        if buffer and section != buffer_section:
            flush()
        if not buffer:
            buffer_section = list(section)

        for segment in split_text(block["content"], SEM_MAX):
            if buffer and buffer_len + len(segment) > SEM_MAX:
                flush()
                buffer_section = list(section)
            buffer.append((block, segment))
            buffer_len += len(segment)
            if buffer_len >= SEM_MAX:
                flush()

    flush()
    return chunks


# ── hierarchical ──────────────────────────────────────────────────────────────


def build_parents(meta: dict, blocks: list[dict]) -> list[dict]:
    doc_id = meta["doc_id"]
    parents: list[dict] = []
    current: list[tuple[dict, str]] = []
    current_len = 0
    current_section: list[str] = []
    counter = 0

    def flush():
        nonlocal current, current_len
        if not current:
            return
        nonlocal counter
        counter += 1
        parents.append({
            "parent_id": f"{doc_id}:hier:p{counter:04d}",
            "doc_id": doc_id,
            "strategy": "hierarchical_parent",
            "content": "\n\n".join(text for _, text in current if text.strip()),
            "block_ids": list(dict.fromkeys(b["block_id"] for b, _ in current)),
            "page_start": min(b["page_num"] for b, _ in current),
            "page_end": max(b["page_num"] for b, _ in current),
            "section_path": list(current_section),
            "block_types": sorted({b["block_type"] for b, _ in current}),
            "_pieces": list(current),
        })
        current = []
        current_len = 0

    for block in blocks:
        btype = block["block_type"]
        section = block["section_path"]

        if btype == "title":
            flush()
            current_section = list(section)
            current = [block_piece(block)]
            current_len = len(block["content"])
            continue

        if current and section != current_section:
            flush()
        if not current:
            current_section = list(section)

        text = block["content"]
        if len(text) > HIER_PARENT_MAX and btype == "text":
            flush()
            for segment in split_text(text, HIER_PARENT_MAX):
                current_section = list(section)
                current = [(block, segment)]
                current_len = len(segment)
                flush()
            continue

        if current and current_len + len(text) > HIER_PARENT_MAX:
            flush()
            current_section = list(section)
        current.append(block_piece(block))
        current_len += len(text)

    flush()
    return parents


def build_children(meta: dict, parents: list[dict]) -> list[dict]:
    doc_id = meta["doc_id"]
    children: list[dict] = []
    counter = 0

    for parent in parents:
        pieces = [
            (block, text) for block, text in parent["_pieces"]
            if block["block_type"] != "title"
        ]
        if not pieces:
            continue

        buffer: list[tuple[dict, str]] = []
        buffer_len = 0

        def flush():
            nonlocal buffer, buffer_len, counter
            if not buffer:
                return
            counter += 1
            children.append(make_chunk(
                f"{doc_id}:hier:c{counter:05d}", doc_id, "hierarchical",
                buffer, parent["section_path"], parent["parent_id"],
            ))
            buffer = []
            buffer_len = 0

        for block, text in pieces:
            if block["block_type"] in ("table", "figure"):
                flush()
                counter += 1
                children.append(make_chunk(
                    f"{doc_id}:hier:c{counter:05d}", doc_id, "hierarchical",
                    [(block, text)], parent["section_path"], parent["parent_id"],
                ))
                continue

            for segment in split_text(text, HIER_CHILD_MAX):
                if buffer and buffer_len + len(segment) > HIER_CHILD_MAX:
                    flush()
                buffer.append((block, segment))
                buffer_len += len(segment)
                if buffer_len >= HIER_CHILD_MAX:
                    flush()
        flush()

    return children


# ── 统计 ──────────────────────────────────────────────────────────────────────


def size_stats(chunks: list[dict]) -> dict:
    sizes = sorted(len(c["content"]) for c in chunks)
    if not sizes:
        return {"count": 0}
    return {
        "count": len(sizes),
        "chars": sum(sizes),
        "mean": round(statistics.mean(sizes), 1),
        "median": statistics.median(sizes),
        "p90": sizes[int(len(sizes) * 0.9)],
        "max": max(sizes),
    }


def coverage(chunks: list[dict], blocks: list[dict]) -> dict:
    expected = {b["block_id"] for b in blocks if b["block_type"] != "title"}
    seen = Counter()
    for chunk in chunks:
        for block_id in chunk["block_ids"]:
            seen[block_id] += 1
    covered = set(seen) & expected
    return {
        "expected": len(expected),
        "covered": len(covered),
        "missing": sorted(expected - covered),
        "split_blocks": sum(1 for block_id in expected if seen[block_id] > 1),
        "extra_ids": sorted(set(seen) - expected),
    }


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 主流程 ────────────────────────────────────────────────────────────────────


def main():
    CHUNKS_DIR.mkdir(exist_ok=True)
    parsed_files = sorted(
        p for p in PARSED_DIR.glob("*.json") if not p.name.startswith("_")
    )
    if not parsed_files:
        print("没有找到 parsed/*.json，请先运行 parse_pdf.py")
        return

    all_semantic: list[dict] = []
    all_parents: list[dict] = []
    all_children: list[dict] = []
    per_doc: list[dict] = []

    for path in parsed_files:
        meta, blocks = load_blocks(path)
        doc_id = meta["doc_id"]

        semantic = build_semantic(meta, blocks)
        parents = build_parents(meta, blocks)
        parents = [
            p for p in parents
            if any(b["block_type"] != "title" for b, _ in p["_pieces"])
        ]
        children = build_children(meta, parents)

        all_semantic.extend(semantic)
        all_parents.extend(parents)
        all_children.extend(children)

        sem_cov = coverage(semantic, blocks)
        child_cov = coverage(children, blocks)
        table_ids = {b["block_id"] for b in blocks if b["block_type"] == "table"}
        figure_ids = {b["block_id"] for b in blocks if b["block_type"] == "figure"}
        sem_table_chunks = sum(1 for c in semantic if "table" in c["block_types"])
        sem_figure_chunks = sum(1 for c in semantic if "figure" in c["block_types"])
        child_table_chunks = sum(1 for c in children if "table" in c["block_types"])
        child_figure_chunks = sum(1 for c in children if "figure" in c["block_types"])

        per_doc.append({
            "doc_id": doc_id,
            "blocks": len(blocks),
            "non_title_blocks": sem_cov["expected"],
            "semantic": {
                **size_stats(semantic),
                "coverage": sem_cov,
                "oversize": sum(1 for c in semantic if len(c["content"]) > SEM_MAX),
                "table_chunks": sem_table_chunks,
                "figure_chunks": sem_figure_chunks,
                "table_integrity": sem_table_chunks == len(table_ids),
                "figure_integrity": sem_figure_chunks == len(figure_ids),
            },
            "hierarchical": {
                "parents": len(parents),
                "children": len(children),
                **{f"child_{k}": v for k, v in size_stats(children).items() if k != "count"},
                "coverage": child_cov,
                "oversize_children": sum(1 for c in children if len(c["content"]) > HIER_CHILD_MAX),
                "table_chunks": child_table_chunks,
                "figure_chunks": child_figure_chunks,
                "table_integrity": child_table_chunks == len(table_ids),
                "figure_integrity": child_figure_chunks == len(figure_ids),
            },
        })
        print(
            f"[{doc_id}] semantic {len(semantic)} 块；"
            f"hierarchical {len(parents)} 父 / {len(children)} 子；"
            f"覆盖 {sem_cov['covered']}/{sem_cov['expected']}"
        )

    parent_records = [
        {k: v for k, v in parent.items() if k != "_pieces"}
        for parent in all_parents
    ]
    write_json(CHUNKS_DIR / "semantic.json", all_semantic)
    write_json(CHUNKS_DIR / "hierarchical_parents.json", parent_records)
    write_json(CHUNKS_DIR / "hierarchical_children.json", all_children)

    report = {
        "settings": {
            "semantic_max": SEM_MAX,
            "hier_parent_max": HIER_PARENT_MAX,
            "hier_child_max": HIER_CHILD_MAX,
        },
        "totals": {
            "semantic_chunks": len(all_semantic),
            "hierarchical_parents": len(all_parents),
            "hierarchical_children": len(all_children),
        },
        "semantic": size_stats(all_semantic),
        "hierarchical_children": size_stats(all_children),
        "hierarchical_parents": size_stats(all_parents),
        "documents": per_doc,
    }
    write_json(CHUNKS_DIR / "_chunk_report.json", report)

    print(json.dumps(report["totals"], ensure_ascii=False, indent=2))
    print(f"输出目录: {CHUNKS_DIR}")


if __name__ == "__main__":
    main()
