"""把 docs/ 下的北斗 ICD 文档解析成 block 列表。

输出：
    parsed/<文件名>.json          每份文档一个
    parsed/_parse_report.json     跨文档统计

结构：
    meta        文档级信息：页数、正文基准字号、页眉页脚、PUA 统计
    blocks[]    一维 block 列表，按阅读顺序排列
      - 公共字段   block_id、block_type、page_num、section_path、content 等
      - table      仅 table 块有：caption、rows、n_rows、n_cols 等
      - figure     仅 figure 块有：caption、labels 等

用法：
    python parse_pdf.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pdfplumber
import pymupdf

try:
    import pytesseract
    from PIL import Image

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "docs"
PARSED_DIR = ROOT / "parsed"

# 标题字号必须比正文基准大出这个量。B1C 正文 14.0、标题 15.0/16.0；
# B1I 正文 14.0、标题 15.0；原始白皮书正文 12.0、标题 14.1。
TITLE_SIZE_DELTA = 0.8
# 无编号但字号明显大于正文的行，按 1 级标题处理。
TITLE_UNNUMBERED_DELTA = 3.0

# 编号标题：1、4.1、5.2.4.8。编号后面必须跟空白或汉字，
# 避免把 "1575.42 MHz"、"2006 年" 这类正文误判成标题。
SECTION_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})(?=\s|[一-鿿])")
# 中文编号标题：一、 / （一） / 第一章。发展报告和白皮书用这一套。
CN_TOP_RE = re.compile(r"^[一二三四五六七八九十]{1,3}、")
CN_SUB_RE = re.compile(r"^[（(][一二三四五六七八九十]{1,3}[）)]")
CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百]+[章节]")
CAPTION_RE = re.compile(r"^(表|图)\s*\d+([\-－—.．]\d+)*")
# 题注是名词短语。正文里的“表4-2 给出了……”“表6-3 中，……”虽然也以
# 表/图编号开头，但带句读或谓语，属于引语段，不能挂到图或表上。
CAPTION_SENTENCE_RE = re.compile(r"[，。；：]|给出|示出|如下|参见|其中")
PAGE_NUM_RE = re.compile(r"^\d{1,4}$")
PAGE_NUM_TEXT_RE = re.compile(r"^(第\s*\d+\s*页|[-—]\s*\d+\s*[-—])$")
# 白皮书页码排成 “２·”，全角数字加间隔点。
PAGE_NUM_ANY_RE = re.compile(r"^(?:\d{1,4}|[０-９]{1,4})\s*[·•]?$")
ROMAN_PAGE_RE = re.compile(r"^[IVXLCDM]{1,7}$")
FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

# PDF 文本层里的私有区字符全部来自 Symbol / MT Extra / Wingdings 字体。
# Symbol 的 0x20–0x7E 里既有 ASCII 也有希腊字母（0x57 是 Ω，不是 W），
# 所以必须按 span 的字体分别映射，不能统一减 F000。
SYMBOL_MAP = {
    0xF020: " ", 0xF028: "(", 0xF029: ")", 0xF02B: "+", 0xF02D: "-",
    0xF03C: "<", 0xF03D: "=", 0xF03E: ">",
    0xF044: "Δ", 0xF046: "Φ", 0xF057: "Ω", 0xF05B: "[", 0xF05D: "]",
    0xF061: "α", 0xF062: "β", 0xF064: "δ", 0xF065: "ε", 0xF066: "φ",
    0xF067: "γ", 0xF069: "ι", 0xF06A: "ϕ", 0xF06C: "λ", 0xF06D: "μ",
    0xF06E: "ν", 0xF070: "π", 0xF071: "θ", 0xF072: "ρ", 0xF073: "σ",
    0xF074: "τ", 0xF077: "ω", 0xF079: "ψ", 0xF07A: "ζ", 0xF07B: "{",
    0xF07D: "}",
    0xF0A2: "′", 0xF0A3: "≤", 0xF0A5: "∞", 0xF0AE: "→", 0xF0B0: "°",
    0xF0B1: "±", 0xF0B3: "≥", 0xF0B4: "×", 0xF0B7: "•", 0xF0B9: "≠",
    0xF0BC: "…", 0xF0C5: "⊕", 0xF0CE: "∈", 0xF0D3: "©", 0xF0D7: "⋅",
    0xF0D9: "∧", 0xF0E5: "∑", 0xF0E6: "⎛", 0xF0E7: "⎜", 0xF0E8: "⎝",
    0xF0E9: "⎡", 0xF0EA: "⎢", 0xF0EB: "⎣", 0xF0EC: "⎧", 0xF0ED: "⎨",
    0xF0EE: "⎩", 0xF0EF: "⎪", 0xF0F6: "⎰", 0xF0F7: "⎱", 0xF0F8: "⎲",
    0xF0F9: "⎳", 0xF0FA: "|", 0xF0FB: "~",
}
MT_EXTRA_MAP = {0xF026: "̇", 0xF03D: "="}
WINGDINGS_MAP = {0xF020: " ", 0xF06C: "•", 0xF0D8: "➢"}

# 没有字体信息的文本（pdfplumber 表格单元格）只能按通用规则映射：
# ASCII 区减 F000，F0A0 以上按 Symbol 符号映射。
PUA_MAP = {0xF000 + code: chr(code) for code in range(0x20, 0x7F)}
PUA_MAP.update({k: v for k, v in SYMBOL_MAP.items() if k >= 0xF0A0})
# 白皮书用 E-BZ-PK74840e 排小数点和全角数字，“０E010２” 即 “0.2”。
PUA_MAP[0xE010] = "."

CJK_CHARS = "一-鿿　-〿！-｠"
CJK_SPACE_RE = re.compile(rf"(?<=[{CJK_CHARS}])\s+(?=[{CJK_CHARS}])")


# ── 文本清洗 ──────────────────────────────────────────────────────────────────


def is_pua_cp(cp: int) -> bool:
    return 0xE000 <= cp <= 0xF8FF or 0xF0000 <= cp <= 0xFFFFD


def pua_table(font: str) -> dict:
    f = font.lower()
    if f.startswith("symbol"):
        return SYMBOL_MAP
    if "mtextra" in f or "mt extra" in f:
        return MT_EXTRA_MAP
    if "wingding" in f:
        return WINGDINGS_MAP
    return PUA_MAP


def map_pua_text(text: str, font: str = "") -> str:
    table = pua_table(font)
    return "".join(table.get(ord(ch), ch) for ch in text)


def normalize_text(text: str, font: str = "") -> str:
    """映射私有区字符，压缩空白，删除汉字之间的空格。"""
    s = map_pua_text(text, font)
    # 映射后仍留在文本里的私有区字符没有可用含义；替换符是提取失败的
    # 占位，两者都从正文删除。
    s = "".join(ch for ch in s if ch != "�" and not is_pua_cp(ord(ch)))
    s = s.replace(" ", " ").replace("　", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = CJK_SPACE_RE.sub("", s)
    return s.strip()


def clean_cell(value) -> str:
    if value is None:
        return ""
    s = str(value).replace("\r", " ").replace("\n", " ")
    return normalize_text(s)


def is_bold_font(font: str) -> bool:
    f = font.lower()
    return any(k in f for k in ("bold", "black", "heavy", "simhei", "黑体"))


# ── 几何工具 ──────────────────────────────────────────────────────────────────


def bbox_contains(bbox, x: float, y: float, margin: float = 0.0) -> bool:
    x0, y0, x1, y1 = bbox
    return x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin


def bbox_area(bbox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def intersect_area(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def union_bbox(boxes):
    boxes = [b for b in boxes if b]
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def line_center(bbox):
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def h_overlap(a, b, margin: float = 10.0) -> bool:
    return a[0] <= b[2] + margin and a[2] >= b[0] - margin


def vertical_gap(line_bbox, visual_bbox) -> float:
    if line_bbox[3] <= visual_bbox[1]:
        return visual_bbox[1] - line_bbox[3]
    if line_bbox[1] >= visual_bbox[3]:
        return line_bbox[1] - visual_bbox[3]
    return 0.0


def cluster_values(values, tol: float = 1.5) -> list[float]:
    vals = sorted(values)
    clusters: list[list[float]] = []
    for v in vals:
        if clusters and v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


# ── 页眉页脚 ──────────────────────────────────────────────────────────────────

# 页边区域按页高取，不用固定磅值：B1I 有 27 页是 842×595 的横版，
# 固定 7%/9% 会把横版页的页眉页脚漏在区域外。
def top_limit(height: float) -> float:
    # BDSBAS／PPP／发展报告的书眉下沿到 75.8pt，0.08H=67.4 会漏掉，
    # 取 0.10H；横版页 595pt 时仍保留 70pt，覆盖 42.7–55.6 的书眉。
    return max(70.0, 0.10 * height)


def bottom_start(height: float) -> float:
    return height - max(75.0, 0.13 * height)


def line_in_table(bbox, tables) -> bool:
    cx, cy = line_center(bbox)
    return any(bbox_contains(t["bbox"], cx, cy) for t in tables)


def detect_header_footer(pages):
    heads = Counter()
    foots = Counter()
    has_page_num = False
    for page in pages:
        h = page["height"]
        for line in page["lines"]:
            if line_in_table(line["bbox"], page["tables"]):
                continue
            text = line["text"]
            if not text:
                continue
            if line["bbox"][3] <= top_limit(h):
                heads[text] += 1
            elif line["bbox"][1] >= bottom_start(h):
                if (PAGE_NUM_ANY_RE.fullmatch(text) or ROMAN_PAGE_RE.fullmatch(text)
                        or PAGE_NUM_TEXT_RE.fullmatch(text)):
                    has_page_num = True
                else:
                    foots[text] += 1

    threshold = max(2, int(len(pages) * 0.30))
    headers = [t for t, c in heads.items() if c >= threshold]
    footers = [t for t, c in foots.items() if c >= threshold]
    return headers, footers, has_page_num


def is_header_footer(line, page, headers, footers, has_page_num) -> bool:
    h = page["height"]
    y0, y1 = line["bbox"][1], line["bbox"][3]
    text = line["text"]
    if y1 <= top_limit(h) and text in headers:
        return True
    if y0 >= bottom_start(h):
        if text in footers:
            return True
        if has_page_num and (PAGE_NUM_ANY_RE.fullmatch(text) or ROMAN_PAGE_RE.fullmatch(text)
                             or PAGE_NUM_TEXT_RE.fullmatch(text)):
            return True
    return False


def find_printed_page(page, has_page_num):
    if not has_page_num:
        return None
    best = None
    for line in page["lines"]:
        if line_in_table(line["bbox"], page["tables"]):
            continue
        if line["bbox"][1] < bottom_start(page["height"]):
            continue
        if PAGE_NUM_ANY_RE.fullmatch(line["text"]):
            if best is None or line["bbox"][1] > best[0]:
                best = (line["bbox"][1], line["text"].translate(FULLWIDTH_DIGITS).strip(" ·•"))
        elif ROMAN_PAGE_RE.fullmatch(line["text"]):
            if best is None or line["bbox"][1] > best[0]:
                best = (line["bbox"][1], line["text"])
        elif PAGE_NUM_TEXT_RE.fullmatch(line["text"]):
            digits = re.sub(r"\D", "", line["text"].translate(FULLWIDTH_DIGITS))
            if best is None or line["bbox"][1] > best[0]:
                best = (line["bbox"][1], digits)
    return best[1] if best else None


# ── 表格 ──────────────────────────────────────────────────────────────────────


def fill_merged_cells(data, cell_rects, row_bboxes):
    """把跨行/跨列格里的文字复制到被覆盖的位置。

    pdfplumber 的 row.cells 与数据列一一对齐：被合并覆盖的位置是 None。
    空单元格本身也会返回 None，但它有单元格矩形，所以能区分：
    矩形为 None 的位置才是被合并覆盖的位置。
    """
    n_rows = len(data)
    n_cols = max((len(r) for r in data), default=0)
    rows = [list(r) + [None] * (n_cols - len(r)) for r in data]

    rects = []
    for i in range(n_rows):
        row = cell_rects[i] if i < len(cell_rects) else []
        rects.append(list(row) + [None] * (n_cols - len(row)))

    # 行边界取所有单元格 y0 的聚类，不能用 row.bbox：
    # 一行里只要有一个跨多行的合并格，row.bbox 就会横跨整张表。
    ys = [r[1] for row in rects for r in row if r]
    tops = cluster_values(ys, tol=1.0)
    if len(tops) != n_rows:
        tops = []
        for row in rects:
            row_ys = [r[1] for r in row if r]
            tops.append(min(row_ys) if row_ys else None)
        known = [t for t in tops if t is not None]
        step = 12.0
        if len(known) >= 2:
            diffs = sorted(b - a for a, b in zip(known, known[1:]) if b > a)
            step = diffs[len(diffs) // 2] if diffs else 12.0
        previous = None
        for i, t in enumerate(tops):
            if t is None or (previous is not None and t <= previous + 0.5):
                t = previous + step if previous is not None else t
                tops[i] = t
            previous = t

    bottom = max((r[3] for row in rects for r in row if r), default=0.0)
    bounds = []
    for i in range(n_rows):
        y0 = tops[i] if i < len(tops) else None
        y1 = tops[i + 1] if i + 1 < len(tops) else bottom
        bounds.append((y0, y1) if y0 is not None and y1 is not None and y1 > y0 else None)

    xs = []
    for row in rects:
        for r in row:
            if r:
                xs.extend([r[0], r[2]])
    edges = cluster_values(xs)
    if len(edges) != n_cols + 1:
        # 从单元格最多的一行取列边界
        best = max(rects, key=lambda row: sum(1 for r in row if r), default=[])
        xs = []
        for r in best:
            if r:
                xs.extend([r[0], r[2]])
        edges = cluster_values(xs)
    if len(edges) != n_cols + 1:
        return rows, 0

    filled = 0
    table_w = edges[-1] - edges[0]
    for r in range(n_rows):
        if bounds[r] is None:
            continue
        for c in range(n_cols):
            if rows[r][c] is not None or rects[r][c] is not None:
                continue
            cx = (edges[c] + edges[c + 1]) / 2
            cy = (bounds[r][0] + bounds[r][1]) / 2
            cover = None
            for r2 in range(n_rows):
                for c2 in range(n_cols):
                    rect = rects[r2][c2]
                    if rect and bbox_contains(rect, cx, cy, margin=0.5):
                        cover = (r2, c2)
                        break
                if cover:
                    break
            if cover:
                # 通栏合并格（如表下注释）只在首列保留一份，
                # 复制到其余列会让同一句注释在表里出现七八遍。
                cover_rect = rects[cover[0]][cover[1]]
                if (cover_rect and table_w > 0
                        and (cover_rect[2] - cover_rect[0]) >= 0.85 * table_w):
                    continue
                value = rows[cover[0]][cover[1]]
                if value is not None and str(value).strip():
                    rows[r][c] = value
                    filled += 1
    return rows, filled


def table_to_markdown(rows) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [list(r) + [""] * (width - len(r)) for r in rows]
    lines = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def md_cell(text: str) -> str:
    return text.replace("|", "\\|")


# ── 图形区域 ──────────────────────────────────────────────────────────────────


def scan_figures(fitz_page):
    """把矢量绘图和位图聚成图形区域。

    表格边框也是矢量线，所以表格周围也会形成区域；后面用表格 bbox
    判断区域里有没有表，有表就把区域内的文字当作表格标签，不另产 figure。
    """
    page_w = fitz_page.rect.width
    page_h = fitz_page.rect.height
    items = []

    for item in fitz_page.get_drawings():
        r = item["rect"]
        w, h = r.width, r.height
        if w < 1.0 and h < 1.0:
            continue
        # 整页背景、通栏横线、极小装饰线不参与聚类
        if w > 0.90 * page_w and h < 5:
            continue
        if w * h > 0.90 * page_w * page_h:
            continue
        if w * h < 20:
            continue
        items.append(((r.x0, r.y0, r.x1, r.y1), "vec"))

    for info in fitz_page.get_image_info():
        r = pymupdf.Rect(info["bbox"])
        if r.width < 5 or r.height < 5:
            continue
        items.append(((r.x0, r.y0, r.x1, r.y1), "img"))

    clusters = []
    for bbox, kind in sorted(items, key=lambda x: (x[0][1], x[0][0])):
        placed = False
        for c in clusters:
            cb = c["bbox"]
            if (
                bbox[1] <= cb[3] + 12
                and bbox[3] >= cb[1] - 12
                and bbox[0] <= cb[2] + 12
                and bbox[2] >= cb[0] - 12
            ):
                cb[0] = min(cb[0], bbox[0])
                cb[1] = min(cb[1], bbox[1])
                cb[2] = max(cb[2], bbox[2])
                cb[3] = max(cb[3], bbox[3])
                c["n_vec" if kind == "vec" else "n_img"] += 1
                placed = True
                break
        if not placed:
            clusters.append({
                "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                "n_vec": int(kind == "vec"),
                "n_img": int(kind == "img"),
            })

    # 贪心插入会把本应合并的区域拆成两块，最后再合并相互接触的区域。
    # B1I 第 17 页的或门就在主框图右侧 0.7pt 处，第一轮没有并进去。
    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                a, b = clusters[i], clusters[j]
                ab, bb = a["bbox"], b["bbox"]
                if (
                    bb[1] <= ab[3] + 12
                    and bb[3] >= ab[1] - 12
                    and bb[0] <= ab[2] + 12
                    and bb[2] >= ab[0] - 12
                ):
                    ab[0] = min(ab[0], bb[0])
                    ab[1] = min(ab[1], bb[1])
                    ab[2] = max(ab[2], bb[2])
                    ab[3] = max(ab[3], bb[3])
                    a["n_vec"] += b["n_vec"]
                    a["n_img"] += b["n_img"]
                    clusters.pop(j)
                    changed = True
                    break
            if changed:
                break

    out = []
    for c in clusters:
        cb = c["bbox"]
        w, h = cb[2] - cb[0], cb[3] - cb[1]
        if w < 30 or h < 15:
            continue
        if w * h > 0.80 * page_w * page_h:
            continue
        # 单个位图算图；纯矢量区域至少要有 3 条线，避免把装饰框当图
        if c["n_img"] == 0 and c["n_vec"] < 3:
            continue
        c["has_vector"] = c["n_vec"] > 0
        c["has_image"] = c["n_img"] > 0
        out.append(c)
    return out


# ── 扫描阶段 ──────────────────────────────────────────────────────────────────


def ocr_page(fitz_page, dpi: int = 200) -> str:
    if not OCR_AVAILABLE:
        return "[扫描页，OCR 不可用，内容跳过]"
    try:
        pix = fitz_page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img, lang="chi_sim+eng")
    except Exception as exc:  # noqa: BLE001
        return f"[扫描页，OCR 失败：{exc}]"


def scan_document(pdf_path: Path):
    pages = []

    with pdfplumber.open(pdf_path) as plumber, pymupdf.open(pdf_path) as doc:
        for index in range(len(doc)):
            fp = doc[index]
            pp = plumber.pages[index]
            width, height = fp.rect.width, fp.rect.height

            tables = []
            try:
                found = pp.find_tables()
            except Exception as exc:  # noqa: BLE001
                print(f"  [警告] 第 {index + 1} 页表格检测失败：{exc}")
                found = []

            for table in found:
                try:
                    data = table.extract()
                except Exception:
                    continue
                if not data:
                    continue
                n_rows = len(data)
                n_cols = max((len(r) for r in data), default=0)
                if n_rows == 0 or n_cols == 0:
                    continue
                # 单列且不超过 3 行：图内标题框之类的伪表
                if n_cols == 1 and n_rows <= 3:
                    continue
                if bbox_area(table.bbox) < 300:
                    continue
                if not any(clean_cell(c) for row in data for c in row):
                    continue

                cell_rects = []
                row_bboxes = []
                for row in table.rows:
                    row_bboxes.append(tuple(row.bbox))
                    cell_rects.append(
                        [None if c is None else tuple(c) for c in row.cells]
                    )

                tables.append({
                    "bbox": tuple(table.bbox),
                    "data": data,
                    "cell_rects": cell_rects,
                    "row_bboxes": row_bboxes,
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                })

            lines = []
            for bi, block in enumerate(
                fp.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]
            ):
                if block.get("type") != 0:
                    continue
                for li, line in enumerate(block.get("lines", [])):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    raw = "".join(map_pua_text(s["text"], s["font"]) for s in spans)
                    if not raw.strip():
                        continue
                    lines.append({
                        "text": normalize_text(raw),
                        "size": max(s["size"] for s in spans),
                        "bold": any(is_bold_font(s["font"]) for s in spans),
                        "bbox": tuple(line["bbox"]),
                        "block": bi,
                        "line": li,
                    })

            # BDSBAS 第 29、37 页有重复内容流，同一行文字会被取到两遍。
            # 文本和 bbox 都相同的行只保留一条，否则正文和图表标签都会翻倍。
            seen = set()
            unique_lines = []
            for line in lines:
                key = (line["text"], *(round(v, 1) for v in line["bbox"]))
                if key in seen:
                    continue
                seen.add(key)
                unique_lines.append(line)
            lines = unique_lines

            figures = scan_figures(fp)

            full_text = fp.get_text("text")
            ocr_text = ""
            if len(full_text.strip()) < 50 and (
                fp.get_image_info() or figures
            ):
                ocr_text = ocr_page(fp)

            pages.append({
                "page": index + 1,
                "width": width,
                "height": height,
                "lines": lines,
                "tables": tables,
                "figures": figures,
                "ocr_text": ocr_text,
            })

    return pages


def compute_body_font_size(pages) -> float:
    counts = Counter()
    for page in pages:
        h = page["height"]
        for line in page["lines"]:
            if line["bbox"][3] <= top_limit(h) or line["bbox"][1] >= bottom_start(h):
                continue
            if line_in_table(line["bbox"], page["tables"]):
                continue
            if any(bbox_contains(f["bbox"], *line_center(line["bbox"])) for f in page["figures"]):
                continue
            if line["size"] < 5:
                continue
            counts[round(line["size"], 1)] += len(line["text"])
    if not counts:
        return 12.0
    return counts.most_common(1)[0][0]


# ── 块构建 ────────────────────────────────────────────────────────────────────


def _join_fragments(left: str, right: str, gap: float, size: float) -> str:
    if not left:
        return right
    if not right:
        return left
    if gap <= 0.25 * size:
        return left + right
    if _is_cjk(left[-1]) and _is_cjk(right[0]):
        return left + right
    return left + " " + right


def _is_cjk(ch: str) -> bool:
    return "一" <= ch <= "鿿"


def merge_heading_fragments(lines, body_size: float, tables):
    """合并同一基线上被拆开的标题碎片。

    PyMuPDF 会把 “1 范围”“4.1 空间段” 拆成编号行和文字行，
    也会把 “引 言” 拆成两个单字行。只合并字号达到标题量级的行，
    正文行和图内小字号标签不参与。
    """
    def heading_like(line):
        if line["size"] >= body_size + 0.5:
            return True
        return (
            bool(re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){0,3}", line["text"]))
            and line["size"] >= body_size - 0.5
        )

    order = sorted(
        range(len(lines)), key=lambda i: (lines[i]["bbox"][1], lines[i]["bbox"][0])
    )
    groups = []
    for i in order:
        b = lines[i]["bbox"]
        cy = (b[1] + b[3]) / 2
        for group in groups:
            gb = group["bbox"]
            gcy = (gb[1] + gb[3]) / 2
            tol = max(2.0, 0.25 * min(b[3] - b[1], gb[3] - gb[1]))
            if abs(cy - gcy) <= tol:
                group["items"].append(i)
                group["bbox"] = (
                    min(gb[0], b[0]), min(gb[1], b[1]),
                    max(gb[2], b[2]), max(gb[3], b[3]),
                )
                break
        else:
            groups.append({"items": [i], "bbox": b})

    merged = []
    for group in groups:
        items = sorted(group["items"], key=lambda i: lines[i]["bbox"][0])
        out = []
        for i in items:
            line = lines[i]
            if (
                out
                and heading_like(out[-1])
                and heading_like(line)
                and not line_in_table(out[-1]["bbox"], tables)
                and not line_in_table(line["bbox"], tables)
            ):
                prev = out[-1]
                gap = line["bbox"][0] - prev["bbox"][2]
                size = max(prev["size"], line["size"])
                if gap <= 1.6 * size and abs(prev["size"] - line["size"]) <= 1.5:
                    out[-1] = {
                        "text": _join_fragments(prev["text"], line["text"], gap, size),
                        "size": size,
                        "bold": prev["bold"] or line["bold"],
                        "bbox": union_bbox([prev["bbox"], line["bbox"]]),
                        "block": prev["block"],
                        "line": prev["line"],
                    }
                    continue
            out.append(line)
        merged.extend(out)

    merged.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
    return merged


def title_depth(text: str, size: float, bold: bool, body_size: float):
    """判断是否标题，返回层级；不是标题返回 None。"""
    if not text or len(text) > 60:
        return None
    # 公式符号行（∑、=∑、+∑）字号大，会被误判成标题；标题至少要有一个
    # 汉字，或两个以上连续的英文字母/数字。
    if not re.search(r"[一-鿿]", text) and not re.search(r"[A-Za-z0-9]{2,}", text):
        return None
    if CAPTION_RE.match(text):
        return None
    if text.endswith(("。", "；", "，", "：", "、", "？", "！")):
        return None

    # 中文编号：一、为 1 级，（一）为 2 级。发展报告的标题只比正文大
    # 1~2 号，白皮书的标题字体比正文更醒目，用 0.5 的差值都能覆盖。
    if len(text) <= 30 and size > body_size + 0.5:
        if CN_SUB_RE.match(text):
            return 2
        if CN_TOP_RE.match(text) or CHAPTER_RE.match(text):
            return 1

    match = SECTION_RE.match(text)
    if match:
        if size > body_size + TITLE_SIZE_DELTA or (bold and size >= body_size - 0.2):
            depth = len(match.group(1).split("."))
            return min(depth, 4)
        return None

    if re.match(r"^附件\s*[A-Z]", text) or text in (
        "参考文献", "缩略语", "术语和定义", "引言", "前言",
    ):
        return 1

    if size > body_size + TITLE_UNNUMBERED_DELTA and len(text) <= 30:
        return 1
    return None


def build_document_blocks(pages, doc_id, headers, footers, has_page_num, body_size):
    blocks = []
    stack: list[str] = []
    stats = Counter()
    merged_total = 0

    for page in pages:
        page_no = page["page"]
        printed = find_printed_page(page, has_page_num)
        tables = page["tables"]
        figures = page["figures"]
        lines = merge_heading_fragments(page["lines"], body_size, tables)

        # 哪些绘图区域包住了表格
        table_cluster = {}
        cluster_tables = defaultdict(list)
        for ti, table in enumerate(tables):
            for ci, figure in enumerate(figures):
                if intersect_area(figure["bbox"], table["bbox"]) > 0.5 * bbox_area(table["bbox"]):
                    table_cluster[ti] = ci
                    cluster_tables[ci].append(ti)
                    break

        cap_idx = [
            i for i, line in enumerate(lines)
            if CAPTION_RE.match(line["text"])
            and len(line["text"]) <= 60
            and not CAPTION_SENTENCE_RE.search(line["text"])
            and not line["text"].rstrip().endswith(("、", "，", "；", "："))
            and not line_in_table(line["bbox"], tables)
        ]
        used: set[int] = set()

        def attach_caption(visual_bbox):
            best = None
            for i in cap_idx:
                if i in used or not h_overlap(lines[i]["bbox"], visual_bbox):
                    continue
                gap = vertical_gap(lines[i]["bbox"], visual_bbox)
                if gap > 50:
                    continue
                if best is None or gap < best[0]:
                    best = (gap, i)
            if best:
                used.add(best[1])
                return best[1]
            return None

        table_caption = {
            ti: attach_caption(table["bbox"]) for ti, table in enumerate(tables)
        }

        # 视觉元素：表格、不含表格的绘图区域、没挂上图题的图
        visuals = []
        table_visual = {}
        for ti, table in enumerate(tables):
            table_visual[ti] = len(visuals)
            visuals.append({
                "kind": "table",
                "ti": ti,
                "bbox": table["bbox"],
                "caption_idx": table_caption[ti],
                "cluster": figures[table_cluster[ti]] if ti in table_cluster else None,
            })
        for ci, figure in enumerate(figures):
            if ci in cluster_tables:
                continue
            visuals.append({
                "kind": "figure",
                "ti": None,
                "bbox": figure["bbox"],
                "caption_idx": attach_caption(figure["bbox"]),
                "cluster": figure,
            })
        for i in cap_idx:
            if i in used or not lines[i]["text"].startswith("图"):
                continue
            used.add(i)
            visuals.append({
                "kind": "figure",
                "ti": None,
                "bbox": lines[i]["bbox"],
                "caption_idx": i,
                "cluster": None,
            })

        # 小于正文字号的行是图内标签的候选。B1I 位域图的标签是 6.6–9.9 号，
        # 正文是 14 号，字号阈值能把图注、坐标标注和正文分开。
        candidates = []
        for i, line in enumerate(lines):
            if i in used or line_in_table(line["bbox"], tables):
                continue
            if is_header_footer(line, page, headers, footers, has_page_num):
                continue
            if line["size"] < body_size - 1.5:
                candidates.append(i)

        label_owner: dict[int, int] = {}
        for i in candidates:
            line_bbox = lines[i]["bbox"]
            best = None
            for vi, visual in enumerate(visuals):
                vb = visual["bbox"]
                if visual["kind"] == "table":
                    cx = (line_bbox[0] + line_bbox[2]) / 2
                    if not (vb[0] - 60 <= cx <= vb[2] + 60):
                        continue
                    max_gap = 60
                else:
                    max_gap = 120
                gap = vertical_gap(line_bbox, vb)
                if gap > max_gap:
                    continue
                if best is None or gap < best[0]:
                    best = (gap, vi)
            if best:
                label_owner[i] = best[1]

        # 绘图区域内部的大字号短文字也算图标签（如 “OR”“D0”）。
        # 长行通常是正文，不能因为落在 cluster bbox 里就吞掉。
        for vi, visual in enumerate(visuals):
            cluster = visual["cluster"]
            if cluster is None:
                continue
            for i, line in enumerate(lines):
                if i in used or i in label_owner or line_in_table(line["bbox"], tables):
                    continue
                if len(line["text"]) > 12:
                    continue
                if not bbox_contains(cluster["bbox"], *line_center(line["bbox"])):
                    continue
                if is_header_footer(line, page, headers, footers, has_page_num):
                    continue
                label_owner[i] = vi

        labels_by_visual = defaultdict(list)
        for i, vi in label_owner.items():
            labels_by_visual[vi].append(i)

        def label_texts(vi):
            out = []
            for i in sorted(
                labels_by_visual.get(vi, []),
                key=lambda k: (lines[k]["bbox"][1], lines[k]["bbox"][0]),
            ):
                text = lines[i]["text"]
                if not text or CAPTION_RE.match(text):
                    continue
                if PAGE_NUM_RE.fullmatch(text) and len(text) <= 3:
                    continue
                if text not in out:
                    out.append(text)
            return out

        elements = []

        # 表格块
        for ti, table in enumerate(tables):
            rows, filled = fill_merged_cells(
                table["data"], table["cell_rects"], table["row_bboxes"]
            )
            merged_total += filled
            rows = [[clean_cell(v) for v in row] for row in rows]

            n_rows = len(rows)
            n_cols = max((len(r) for r in rows), default=0)
            layout = "bitfield" if n_rows == 1 and n_cols >= 8 else "grid"

            cap_i = table_caption.get(ti)
            caption = lines[cap_i]["text"] if cap_i is not None else ""
            labels = label_texts(table_visual[ti])
            if caption:
                labels = [x for x in labels if x != caption]

            md = table_to_markdown([[md_cell(c) for c in row] for row in rows])
            parts = []
            if caption:
                parts.append(caption)
            parts.append(md)
            if labels:
                parts.append("表格标签：" + "；".join(labels))

            elements.append({
                "kind": "table",
                "y0": table["bbox"][1],
                "x0": table["bbox"][0],
                "bbox": table["bbox"],
                "payload": {
                    "block_type": "table",
                    "content": "\n".join(parts),
                    "table": {
                        "caption": caption,
                        "n_rows": n_rows,
                        "n_cols": n_cols,
                        "layout": layout,
                        "merged_filled": filled,
                        "rows": rows,
                        "labels": labels,
                    },
                },
            })
            stats["table"] += 1
            if layout == "bitfield":
                stats["bitfield"] += 1
            if filled:
                stats["filled_tables"] += 1

        # 图形块
        for vi, visual in enumerate(visuals):
            if visual["kind"] != "figure":
                continue
            cap_i = visual["caption_idx"]
            caption = lines[cap_i]["text"] if cap_i is not None else ""
            labels = label_texts(vi)
            if caption:
                labels = [x for x in labels if x != caption]
            if not caption and not labels:
                continue

            boxes = [visual["bbox"]]
            if cap_i is not None:
                boxes.append(lines[cap_i]["bbox"])
            boxes.extend(lines[i]["bbox"] for i in labels_by_visual.get(vi, []))
            bbox = union_bbox(boxes)

            parts = []
            if caption:
                parts.append(caption)
            if labels:
                parts.append("图示标签：" + "；".join(labels))

            cluster = visual["cluster"]
            elements.append({
                "kind": "figure",
                "y0": bbox[1],
                "x0": bbox[0],
                "bbox": bbox,
                "payload": {
                    "block_type": "figure",
                    "content": "\n".join(parts),
                    "figure": {
                        "caption": caption,
                        "labels": labels,
                        "n_labels": len(labels),
                        "has_vector": bool(cluster and cluster["has_vector"]),
                        "has_image": bool(cluster and cluster["has_image"]),
                    },
                },
            })
            stats["figure"] += 1

        # 正文与标题。图内的标签行已在 label_owner 里排除，图内的长行
        # （正文段落偶然跨过 cluster bbox）仍然保留。
        text_lines = []
        for i, line in enumerate(lines):
            if i in used or i in label_owner:
                continue
            if line_in_table(line["bbox"], tables):
                continue
            if is_header_footer(line, page, headers, footers, has_page_num):
                continue
            text_lines.append(line)

        text_lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
        heights = sorted(l["bbox"][3] - l["bbox"][1] for l in text_lines if l["bbox"][3] > l["bbox"][1])
        median_h = heights[len(heights) // 2] if heights else 12.0

        paragraph: list[dict] = []

        def flush_paragraph():
            nonlocal paragraph
            if not paragraph:
                return
            content = "\n".join(l["text"] for l in paragraph)
            bbox = union_bbox([l["bbox"] for l in paragraph])
            elements.append({
                "kind": "text",
                "y0": bbox[1],
                "x0": bbox[0],
                "bbox": bbox,
                "payload": {
                    "block_type": "text",
                    "content": content,
                },
            })
            stats["text"] += 1
            paragraph = []

        previous = None
        for line in text_lines:
            depth = title_depth(line["text"], line["size"], line["bold"], body_size)
            if depth is not None:
                flush_paragraph()
                # 目录行带点线和页码，去掉后再进 section_path。
                title_text = re.sub(r"[.．·\s]{4,}\d+\s*$", "", line["text"]).strip()
                if not title_text:
                    title_text = line["text"]
                elements.append({
                    "kind": "title",
                    "y0": line["bbox"][1],
                    "x0": line["bbox"][0],
                    "bbox": line["bbox"],
                    "payload": {
                        "block_type": "title",
                        "content": title_text,
                        "depth": depth,
                    },
                })
                stats["title"] += 1
                previous = line
                continue

            if previous is not None:
                gap = line["bbox"][1] - previous["bbox"][3]
                if gap > 1.6 * median_h:
                    flush_paragraph()
            paragraph.append(line)
            previous = line

        flush_paragraph()

        # OCR 保底
        if page["ocr_text"]:
            elements.append({
                "kind": "text",
                "y0": 0.0,
                "x0": 0.0,
                "bbox": (0.0, 0.0, page["width"], page["height"]),
                "payload": {
                    "block_type": "text",
                    "content": normalize_text(page["ocr_text"]),
                    "is_ocr": True,
                },
            })
            stats["ocr"] += 1

        elements.sort(key=lambda e: (e["y0"], e["x0"]))
        for order, element in enumerate(elements, start=1):
            payload = element["payload"]
            if payload["block_type"] == "title":
                depth = payload.pop("depth")
                stack = stack[: depth - 1] + [payload["content"]]
            block = {
                "block_id": f"{doc_id}:p{page_no:03d}:b{order:02d}",
                "block_type": payload["block_type"],
                "page_num": page_no,
                "printed_page": printed,
                "section_path": list(stack),
                "content": payload["content"],
                "bbox": [round(v, 2) for v in element["bbox"]],
                "is_ocr": payload.get("is_ocr", False),
            }
            if "table" in payload:
                block["table"] = payload["table"]
            if "figure" in payload:
                block["figure"] = payload["figure"]
            blocks.append(block)

    return blocks, stats, merged_total


# ── 主流程 ────────────────────────────────────────────────────────────────────


def parse_one(pdf_path: Path):
    doc_id = pdf_path.stem
    print(f"\n[{doc_id}] 扫描 {pdf_path.name}")
    pages = scan_document(pdf_path)

    headers, footers, has_page_num = detect_header_footer(pages)
    body_size = compute_body_font_size(pages)

    blocks, stats, merged_total = build_document_blocks(
        pages, doc_id, headers, footers, has_page_num, body_size
    )

    ocr_pages = [p["page"] for p in pages if p["ocr_text"]]
    meta = {
        "doc_id": doc_id,
        "source_file": str(pdf_path.relative_to(ROOT)).replace("\\", "/"),
        "pages": len(pages),
        "body_font_size": body_size,
        "running_headers": headers,
        "running_footers": footers + (["<纯数字页码>"] if has_page_num else []),
        "ocr_pages": ocr_pages,
    }

    output = {"meta": meta, "blocks": blocks}
    PARSED_DIR.mkdir(exist_ok=True)
    out_path = PARSED_DIR / f"{doc_id}.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    captionless = sum(1 for b in blocks if b["block_type"] == "table" and not b["table"]["caption"])
    doc_stats = {
        "doc_id": doc_id,
        "pages": len(pages),
        "blocks": len(blocks),
        "titles": stats["title"],
        "texts": stats["text"],
        "tables": stats["table"],
        "bitfields": stats["bitfield"],
        "figures": stats["figure"],
        "merged_filled": merged_total,
        "captionless_tables": captionless,
        "ocr_pages": ocr_pages,
        "body_font_size": body_size,
    }

    print(f"  正文基准字号 {body_size}  页眉 {len(headers)} 条  页脚 {len(footers)} 条")
    print(
        f"  blocks {len(blocks)}  title {stats['title']}  text {stats['text']}  "
        f"table {stats['table']}（位域 {stats['bitfield']}）  figure {stats['figure']}"
    )
    print(f"  合并格填充 {merged_total}  无题表格 {captionless}")
    print(f"  → {out_path.relative_to(ROOT)}")
    return doc_stats


def main():
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print("没有找到 PDF")
        return

    documents = [parse_one(p) for p in pdfs]

    report = {
        "documents": documents,
        "totals": {
            "documents": len(documents),
            "pages": sum(d["pages"] for d in documents),
            "blocks": sum(d["blocks"] for d in documents),
            "titles": sum(d["titles"] for d in documents),
            "texts": sum(d["texts"] for d in documents),
            "tables": sum(d["tables"] for d in documents),
            "bitfields": sum(d["bitfields"] for d in documents),
            "figures": sum(d["figures"] for d in documents),
            "merged_filled": sum(d["merged_filled"] for d in documents),
            "captionless_tables": sum(d["captionless_tables"] for d in documents),
        },
    }
    PARSED_DIR.mkdir(exist_ok=True)
    report_path = PARSED_DIR / "_parse_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n汇总 → {report_path.relative_to(ROOT)}")
    print(json.dumps(report["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
