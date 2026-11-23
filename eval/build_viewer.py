"""把 hierarchical / api 的评测结果渲染成单文件 HTML 查看页。

数据直接内嵌，浏览器以 file:// 打开即可，不需要起本地服务。
"""

import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
SRC = EVAL_DIR / "results" / "hierarchical_api.json"
OUT = EVAL_DIR / "hierarchical_api_viewer.html"

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>hierarchical / api 评测结果</title>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7;
    --surface: #fcfcfb;
    --ink: #0b0b0b;
    --ink-2: #52514e;
    --muted: #898781;
    --grid: #e1e0d9;
    --border: rgba(11, 11, 11, 0.10);
    --accent: #2a78d6;
    --track: #cde2fb;
    --chip: #f0efec;
    --up: #006300;
    --down: #d03b3b;
  }
  html[data-theme="dark"] {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface: #1a1a19;
    --ink: #f0efec;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --border: rgba(255, 255, 255, 0.10);
    --accent: #3987e5;
    --track: #184f95;
    --chip: #262624;
    --up: #0ca30c;
    --down: #e66767;
  }
  @media (prefers-color-scheme: dark) {
    html:not([data-theme="light"]):not([data-theme="dark"]) {
      color-scheme: dark;
      --page: #0d0d0d;
      --surface: #1a1a19;
      --ink: #f0efec;
      --ink-2: #c3c2b7;
      --muted: #898781;
      --grid: #2c2c2a;
      --border: rgba(255, 255, 255, 0.10);
      --accent: #3987e5;
      --track: #184f95;
      --chip: #262624;
      --up: #0ca30c;
      --down: #e66767;
    }
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--page);
    color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", "Microsoft YaHei",
                 "PingFang SC", sans-serif;
    font-size: 14px;
    line-height: 1.6;
  }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 28px 20px 72px; }

  header { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }
  h1 { margin: 0 0 6px; font-size: 22px; font-weight: 600; letter-spacing: 0.01em; }
  .meta { margin: 0; color: var(--muted); font-size: 13px; }
  .theme-btn {
    flex: none; height: 32px; padding: 0 12px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--surface);
    color: var(--ink-2); font: inherit; font-size: 13px; cursor: pointer;
  }
  .theme-btn:hover { background: var(--chip); }

  .tiles {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
    margin: 22px 0 18px;
  }
  .tile {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 14px 14px;
  }
  .tile .label { color: var(--ink-2); font-size: 12px; }
  .tile .value { font-size: 26px; font-weight: 600; line-height: 1.25; margin-top: 2px; }
  .tile .value.na { color: var(--muted); font-size: 20px; }
  .meter { height: 4px; border-radius: 2px; background: var(--track); margin-top: 10px; opacity: 0.9; }
  .meter > i {
    display: block; height: 4px; background: var(--accent);
    border-radius: 0 2px 2px 0;
  }

  .toolbar {
    position: sticky; top: 0; z-index: 5;
    display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
    padding: 12px 0; background: var(--page);
    border-bottom: 1px solid var(--grid);
  }
  .search {
    flex: 1 1 260px; min-width: 200px; height: 34px; padding: 0 12px;
    border: 1px solid var(--border); border-radius: 8px;
    background: var(--surface); color: var(--ink); font: inherit; font-size: 13px;
  }
  .search:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
  .chips { display: flex; gap: 6px; flex-wrap: wrap; }
  .chip {
    height: 30px; padding: 0 11px; border-radius: 999px;
    border: 1px solid var(--border); background: var(--surface);
    color: var(--ink-2); font: inherit; font-size: 12.5px; cursor: pointer;
  }
  .chip:hover { background: var(--chip); }
  .chip[aria-pressed="true"] {
    background: var(--accent); border-color: var(--accent); color: #fff;
  }
  .control {
    height: 30px; padding: 0 8px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--surface);
    color: var(--ink-2); font: inherit; font-size: 12.5px; cursor: pointer;
  }
  .toggle { display: inline-flex; align-items: center; gap: 6px; color: var(--ink-2); font-size: 12.5px; cursor: pointer; user-select: none; }
  .toggle input { accent-color: var(--accent); }
  .count { color: var(--muted); font-size: 12.5px; margin-left: auto; font-variant-numeric: tabular-nums; }

  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 18px; margin-top: 12px;
  }
  .card-top { display: flex; align-items: center; gap: 8px; }
  .qid { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 12.5px; }
  .tag {
    padding: 1px 8px; border-radius: 6px; background: var(--chip);
    color: var(--ink-2); font-size: 11.5px;
  }
  .avg { margin-left: auto; color: var(--ink-2); font-size: 12px; font-variant-numeric: tabular-nums; }
  .question { margin: 8px 0 12px; font-size: 15.5px; line-height: 1.6; }

  .scores { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
  .cell { background: var(--page); border-radius: 8px; padding: 8px 10px 10px; }
  .m-label { color: var(--muted); font-size: 11.5px; }
  .m-value { font-size: 17px; font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1.35; }
  .m-value.na { color: var(--muted); font-weight: 500; }
  .cell .meter { margin-top: 6px; height: 3px; }
  .cell .meter > i { height: 3px; }

  .qa { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px; }
  .qa h3, .block h3 {
    margin: 0 0 4px; color: var(--muted); font-size: 11.5px; font-weight: 500;
  }
  .qa p { margin: 0; font-size: 13.5px; line-height: 1.75; white-space: pre-wrap; word-break: break-word; }

  details { margin-top: 12px; border-top: 1px solid var(--grid); padding-top: 10px; }
  summary {
    cursor: pointer; color: var(--ink-2); font-size: 12.5px;
    list-style-position: inside;
  }
  summary:hover { color: var(--ink); }
  .ctx-list { list-style: none; margin: 10px 0 0; padding: 0; counter-reset: ctx; }
  .ctx-list li { margin-top: 10px; }
  .ctx-head { color: var(--muted); font-size: 11.5px; margin-bottom: 4px; }
  .ctx-body {
    margin: 0; padding: 10px 12px; border-radius: 8px; background: var(--page);
    font-family: inherit; font-size: 12.5px; line-height: 1.65;
    white-space: pre-wrap; word-break: break-word;
    max-height: 320px; overflow: auto;
  }
  .hits { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12.5px; }
  .hits-note { margin: 8px 0 0; color: var(--muted); font-size: 11.5px; line-height: 1.6; }
  .hits th {
    text-align: left; font-weight: 500; color: var(--muted);
    padding: 4px 8px; border-bottom: 1px solid var(--grid);
  }
  .hits td { padding: 6px 8px; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }
  .hits tr:last-child td { border-bottom: 0; }
  .hits .cid { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: 11.5px; word-break: break-all; }
  .up { color: var(--up); }
  .down { color: var(--down); }

  .empty { padding: 40px 0; text-align: center; color: var(--muted); }
  footer { margin-top: 28px; color: var(--muted); font-size: 12px; }
  footer a { color: inherit; }

  @media (max-width: 760px) {
    .tiles, .scores { grid-template-columns: repeat(2, 1fr); }
    .qa { grid-template-columns: 1fr; }
    .count { margin-left: 0; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1 id="title">hierarchical / api</h1>
      <p class="meta" id="meta"></p>
    </div>
    <button class="theme-btn" id="theme" type="button">深色</button>
  </header>

  <section class="tiles" id="tiles" aria-label="四项指标均值"></section>

  <div class="toolbar">
    <input class="search" id="search" type="search" placeholder="搜索题号、问题、回答或参考答案">
    <div class="chips" id="chips"></div>
    <label class="toggle"><input type="checkbox" id="low">只看低分（任一指标 &lt; 0.5）或缺分</label>
    <select class="control" id="sort" aria-label="排序">
      <option value="default">按题号</option>
      <option value="avg-asc">按均值升序</option>
      <option value="avg-desc">按均值降序</option>
      <option value="type">按题型</option>
    </select>
    <span class="count" id="count"></span>
  </div>

  <main id="list"></main>

  <footer>
    数据来源 <a href="results/hierarchical_api.json">eval/results/hierarchical_api.json</a>
  </footer>
</div>

<script type="application/json" id="payload">__DATA__</script>
<script>
(function () {
  "use strict";

  var DATA = JSON.parse(document.getElementById("payload").textContent);
  var METRICS = [
    { key: "faithfulness", label: "Faithfulness" },
    { key: "answer_relevancy", label: "Answer relevancy" },
    { key: "context_precision", label: "Context precision" },
    { key: "context_recall", label: "Context recall" }
  ];
  var RECORDS = DATA.records;
  var PER_Q = DATA.per_question;
  var N = RECORDS.length;
  var TYPE_LABELS = { text: "text", table: "table", cross_block: "cross_block" };
  var state = { q: "", type: "all", low: false, sort: "default" };

  function esc(value) {
    return String(value).replace(/[&<>"]/g, function (ch) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch];
    });
  }
  function fmt(value) {
    return value === null || value === undefined ? "—" : Number(value).toFixed(3);
  }
  function pct(value) {
    return (Math.max(0, Math.min(1, Number(value))) * 100).toFixed(2);
  }
  function avg(i) {
    var vals = METRICS.map(function (m) { return PER_Q[m.key][i]; })
      .filter(function (v) { return v !== null && v !== undefined; });
    if (!vals.length) return null;
    return vals.reduce(function (a, b) { return a + b; }, 0) / vals.length;
  }
  function isLow(i) {
    return METRICS.some(function (m) {
      var v = PER_Q[m.key][i];
      return v === null || v === undefined || v < 0.5;
    });
  }

  function renderMeta() {
    var c = DATA.config;
    document.getElementById("title").textContent = c.index + " / " + c.reranker;
    document.getElementById("meta").textContent =
      c.question_count + " 题 · 生成模型 " + c.generator +
      " · 评判模型 " + c.judge + " · " + c.timestamp;
  }

  function renderTiles() {
    var html = METRICS.map(function (m) {
      var v = DATA.mean[m.key];
      var has = v !== null && v !== undefined;
      return '<div class="tile"><div class="label">' + esc(m.label) + '</div>' +
        '<div class="value' + (has ? "" : " na") + '">' + fmt(v) + '</div>' +
        '<div class="meter"><i style="width:' + (has ? pct(v) : 0) + '%"></i></div></div>';
    }).join("");
    document.getElementById("tiles").innerHTML = html;
  }

  function renderChips() {
    var counts = { all: N };
    RECORDS.forEach(function (r) {
      counts[r.type] = (counts[r.type] || 0) + 1;
    });
    var items = [{ key: "all", label: "全部" }];
    Object.keys(TYPE_LABELS).forEach(function (key) {
      if (counts[key]) items.push({ key: key, label: TYPE_LABELS[key] });
    });
    document.getElementById("chips").innerHTML = items.map(function (item) {
      var active = state.type === item.key;
      return '<button class="chip" type="button" data-type="' + item.key +
        '" aria-pressed="' + active + '">' + esc(item.label) +
        " " + (counts[item.key] || 0) + "</button>";
    }).join("");
  }

  function hitRows(rec) {
    var hasRerank = rec.hits.some(function (h) {
      return h.rerank_score !== null && h.rerank_score !== undefined;
    });
    var head = "<tr><th>重排后名次</th><th>chunk_id</th><th>RRF 分</th><th>RRF 名次</th>" +
      "<th>较 RRF 名次</th>" + (hasRerank ? "<th>重排分</th>" : "") + "</tr>";
    var body = rec.hits.map(function (h) {
      var move = "";
      if (h.retrieval_rank !== null && h.retrieval_rank !== undefined) {
        var delta = h.retrieval_rank - h.rank;
        if (delta > 0) move = '<span class="up">↑' + delta + "</span>";
        else if (delta < 0) move = '<span class="down">↓' + (-delta) + "</span>";
        else move = "—";
      } else {
        move = "—";
      }
      return "<tr><td>" + h.rank + '</td><td class="cid">' + esc(h.chunk_id) +
        "</td><td>" + fmt(h.score) + "</td><td>" +
        (h.retrieval_rank === null || h.retrieval_rank === undefined ? "—" : h.retrieval_rank) +
        "</td><td>" + move + "</td>" +
        (hasRerank ? "<td>" + fmt(h.rerank_score) + "</td>" : "") + "</tr>";
    }).join("");
    return '<p class="hits-note">重排后名次：api 重排在最终 5 条内给出的顺序。RRF 名次：重排前在 30 条候选中按融合分的排名。↑ 前移，↓ 后移。</p>' +
      '<table class="hits"><thead>' + head + "</thead><tbody>" + body + "</tbody></table>";
  }

  function contextList(rec) {
    return rec.contexts.map(function (text, idx) {
      var hit = rec.hits[idx] || {};
      var label = "[" + (idx + 1) + "] " + (hit.chunk_id || "") +
        " · RRF 分 " + fmt(hit.score) +
        (hit.retrieval_rank ? " · RRF 名次 " + hit.retrieval_rank : "");
      return '<li><div class="ctx-head">' + esc(label) + "</div>" +
        '<pre class="ctx-body">' + esc(text) + "</pre></li>";
    }).join("");
  }

  function card(rec, i) {
    var scoreCells = METRICS.map(function (m) {
      var v = PER_Q[m.key][i];
      var has = v !== null && v !== undefined;
      return '<div class="cell"><div class="m-label">' + esc(m.label) + "</div>" +
        '<div class="m-value' + (has ? "" : " na") + '">' + fmt(v) + "</div>" +
        '<div class="meter"><i style="width:' + (has ? pct(v) : 0) + '%"></i></div></div>';
    }).join("");
    var mean = avg(i);
    return '<article class="card" data-i="' + i + '">' +
      '<div class="card-top"><span class="qid">' + esc(rec.qid) + "</span>" +
      '<span class="tag">' + esc(rec.type) + "</span>" +
      '<span class="avg">均值 ' + (mean === null ? "—" : mean.toFixed(3)) + "</span></div>" +
      '<p class="question">' + esc(rec.question) + "</p>" +
      '<div class="scores">' + scoreCells + "</div>" +
      '<div class="qa"><div class="block"><h3>参考答案</h3><p>' + esc(rec.ground_truth) + "</p></div>" +
      '<div class="block"><h3>模型回答</h3><p>' + esc(rec.answer) + "</p></div></div>" +
      '<details><summary>参考上下文（' + rec.contexts.length + " 条）</summary>" +
      '<ol class="ctx-list">' + contextList(rec) + "</ol></details>" +
      '<details><summary>命中明细</summary>' + hitRows(rec) + "</details>" +
      "</article>";
  }

  function render() {
    var q = state.q.trim().toLowerCase();
    var rows = RECORDS.map(function (rec, i) { return { rec: rec, i: i }; })
      .filter(function (row) {
        var rec = row.rec;
        if (state.type !== "all" && rec.type !== state.type) return false;
        if (q) {
          var hay = [rec.qid, rec.question, rec.answer, rec.ground_truth].join(" ").toLowerCase();
          if (hay.indexOf(q) < 0) return false;
        }
        if (state.low && !isLow(row.i)) return false;
        return true;
      });

    if (state.sort === "avg-asc" || state.sort === "avg-desc") {
      rows.sort(function (a, b) {
        var x = avg(a.i), y = avg(b.i);
        if (x === null && y === null) return a.i - b.i;
        if (x === null) return 1;
        if (y === null) return -1;
        return state.sort === "avg-asc" ? x - y : y - x;
      });
    } else if (state.sort === "type") {
      rows.sort(function (a, b) {
        var t = a.rec.type.localeCompare(b.rec.type);
        return t !== 0 ? t : a.i - b.i;
      });
    } else {
      rows.sort(function (a, b) { return a.i - b.i; });
    }

    document.getElementById("count").textContent = "显示 " + rows.length + " / " + N + " 题";
    document.getElementById("list").innerHTML = rows.length
      ? rows.map(function (row) { return card(row.rec, row.i); }).join("")
      : '<p class="empty">没有符合条件的题目</p>';
  }

  document.getElementById("chips").addEventListener("click", function (event) {
    var btn = event.target.closest(".chip");
    if (!btn) return;
    state.type = btn.dataset.type;
    renderChips();
    render();
  });
  document.getElementById("search").addEventListener("input", function (event) {
    state.q = event.target.value;
    render();
  });
  document.getElementById("low").addEventListener("change", function (event) {
    state.low = event.target.checked;
    render();
  });
  document.getElementById("sort").addEventListener("change", function (event) {
    state.sort = event.target.value;
    render();
  });

  var themeBtn = document.getElementById("theme");
  function applyTheme(theme) {
    if (theme) document.documentElement.setAttribute("data-theme", theme);
    else document.documentElement.removeAttribute("data-theme");
    var dark = theme === "dark" ||
      (!theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
    themeBtn.textContent = dark ? "浅色" : "深色";
  }
  themeBtn.addEventListener("click", function () {
    var current = document.documentElement.getAttribute("data-theme");
    var dark = current === "dark" ||
      (!current && window.matchMedia("(prefers-color-scheme: dark)").matches);
    var next = dark ? "light" : "dark";
    try { localStorage.setItem("viewer-theme", next); } catch (err) {}
    applyTheme(next);
  });
  var saved = null;
  try { saved = localStorage.getItem("viewer-theme"); } catch (err) {}
  applyTheme(saved);

  renderMeta();
  renderTiles();
  renderChips();
  render();
})();
</script>
</body>
</html>
"""


def clean(value):
    """把 NaN / Infinity 换成 null，保证内嵌 JSON 能被 JSON.parse 接受。"""
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    records = data["records"]
    for name, values in data["per_question"].items():
        if len(values) != len(records):
            raise SystemExit(
                f"{name} 分项长度 {len(values)} 与记录数 {len(records)} 不一致"
            )

    payload = {
        "config": data["config"],
        "mean": data["mean"],
        "per_question": data["per_question"],
        "records": [
            {
                "qid": record["qid"],
                "type": record["type"],
                "question": record["question"],
                "answer": record["answer"],
                "contexts": record["contexts"],
                "ground_truth": record["ground_truth"],
                "hits": record["hits"],
            }
            for record in records
        ],
    }
    blob = json.dumps(
        clean(payload), ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    OUT.write_text(TEMPLATE.replace("__DATA__", blob), encoding="utf-8")
    print(f"已生成 {OUT}  题目 {len(records)} 条  大小 {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
