# 北斗 ICD 文档 RAG 问答系统

针对北斗卫星导航系统**公开接口控制文件（ICD）**构建的检索增强问答系统。

选这个语料，是因为技术规范类 PDF 的解析和检索难点是普通文档没有的：表格密集、位域字段多、章节层级深、页眉页脚重复、部分页面是扫描件。这个项目把这些问题逐个处理掉，并用 RAGAS 做了六种检索配置的对照实验。

---

## 结果速览

同一批 **25 道题**、同一生成模型、同一评判模型，只改变切块策略与重排方式：

| 索引 | 重排 | faithfulness | answer_relevancy | context_precision | context_recall | 四项均值 |
|---|---|---:|---:|---:|---:|---:|
| semantic | none | 0.9640 | 0.6880 | 0.8567 | 0.9657 | 0.8686 |
| semantic | bge | 0.9407 | 0.7108 | 0.8367 | 0.9443 | 0.8581 |
| semantic | api | 0.8897 | 0.7410 | 0.9685 | 0.9943 | 0.8984 |
| hierarchical | none | 0.9533 | 0.7414 | 0.8526 | 1.0000 | 0.8868 |
| hierarchical | bge | 0.9012 | 0.7252 | 0.8149 | 0.9943 | 0.8589 |
| **hierarchical** | **api** | 0.9207 | 0.7583 | 0.9508 | 1.0000 | **0.9074** |

**两个结论值得单独说：**

1. **`hierarchical + api` 四项均值最高（0.9074）**，且 `context_recall` 达到 1.0000。
2. **本地 `bge-reranker-v2-m3` 重排没有增益，反而低于不重排**（semantic：0.8686 → 0.8581）。这是一个负结论，但它是这组实验里最有价值的部分：重排不是装上就一定涨点的组件。

---

## 语料

7 份公开文档，共 **329 页 / 1304 个 block**：

| 文档 | 特点 |
|---|---|
| B1C ICD 1.0、B1I ICD 3.0、BDSBAS-B1C ICD 1.0、PPP-B2b ICD 1.0 | 表格与位域密集，合计 250 张表、68 张图、71 张位域表 |
| 北斗发展报告 3.0、北斗白皮书 2016 | 连续长文，合计 3 张表、0 张图 |
| 公开服务性能规范 3.0 | 27 张表、0 张图 |

正文均带文本层；仅白皮书第 1 页是扫描封面，按需触发 OCR（可选依赖，未安装时降级为跳过）。

---

## 处理流程

```
docs/*.pdf
  → parse_pdf.py        解析为结构化 block（表格 / 正文 / 图 / 标题）
  → parsed/*.json
  → chunk_documents.py  两种粒度切块（semantic / hierarchical）
  → chunks/*.json
  → embed_chunks.py     bge-large-zh-v1.5 → FAISS
  → embeddings/*.faiss
  → search_chunks.py    向量 + BM25 双路召回，RRF 融合
  → rerank_*.py         可选二阶段重排（本地 bge / LLM 重排 / 两者对比）
  → LLM_generate.py     生成带引用的答案
  → eval/evaluate.py    RAGAS 四项指标评测
```

**两种切块策略：**

- `semantic`：按章节与块边界合并，文本块最多 800 字，表格、图单独成块。
- `hierarchical`：父块最多 1600 字，子块最多 400 字。**检索用子块，生成用父块**——子块提升召回精度，父块保证上下文完整。

---

## 目录结构

```
week10/
├── parse_pdf.py            # PDF → 结构化 block
├── chunk_documents.py      # block → 两种粒度切块
├── embed_chunks.py         # 切块 → FAISS 向量索引
├── search_chunks.py        # 向量 + BM25 + RRF 检索
├── rerank_bge.py           # 本地 bge-reranker 重排
├── rerank_api.py           # LLM 重排（OpenAI 兼容接口）
├── rerank_compare.py       # 两种重排的并列对比
├── LLM_generate.py         # 检索 → 重排 → 生成
├── eval/
│   ├── evaluate.py         # RAGAS 评测
│   ├── questions.json      # 评估题集
│   └── build_viewer.py     # 结果渲染成单文件 HTML
├── docs/                   # 源 PDF
├── week10_report.md        # 完整实验报告
└── requirements.txt
```

`parsed/`、`chunks/`、`embeddings/`、`models/` 都在 `.gitignore` 里，需要本地生成。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

`torch` 请按自己的 CUDA 版本单独安装。扫描页 OCR 默认关闭，需要时再装 `pytesseract` 与系统级 tesseract + `chi_sim` 语言包。

### 2. 下载模型（约 3.4GB）

```bash
huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir models/bge-large-zh-v1.5
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3
```

### 3. 配置 LLM 接口

打开 `rerank_api.py`，修改文件顶部的三个常量：

```python
API_BASE_URL = "XXX"            # OpenAI 兼容接口地址
API_KEY = "XXX"                 # 你的密钥
API_MODEL = "deepseek-flash"    # 模型名
```

> 仓库中这三项已用 `XXX` 占位，原实验使用的是课程网关的 `deepseek-flash`。

### 4. 跑通

```bash
python parse_pdf.py
python chunk_documents.py
python embed_chunks.py

# 检索
python search_chunks.py --index hierarchical --query "北斗系统公开服务性能规范涵盖哪几类服务"

# 检索 + 重排 + 生成
python LLM_generate.py --index hierarchical --reranker api --query "同上"

# 全量评测
python eval/evaluate.py --index hierarchical --reranker api
```

首次运行需下载模型；有 GPU 时 embedding 与 rerank 走 CUDA，无 GPU 会退回 CPU（慢但可用）。

---

## 实验结论

1. **没有一种配置在四项指标上同时最优**，但 `hierarchical + api` 均值最高，`semantic + api` 次之。
2. **重排模式整体排序：`api` > `none` > `bge`。** LLM 重排的 `context_precision`、`context_recall` 明显更高，但 `faithfulness` 下降。
3. **`hierarchical` 相比 `semantic`**：`answer_relevancy`、`context_recall` 更高，`context_precision` 稍逊。
4. **RAGAS 超时会制造虚假的 NaN。** 单题请求超过 `RunConfig(timeout=...)` 即失败并记 NaN。实测 180 秒下出现 5–6 条 NaN，600 秒下为 0——说明 NaN 来自评判超时，不是检索或生成失败。**评测超时不能设太小**，否则结果会失真。
5. **重排增益不能完全归因于重排能力**：本实验中 `api` 重排与答案生成用的是同一个 LLM，后续要把重排器本身的能力分离出来，需要引入独立模型。

---

## 设计要点（解析与检索真正难的地方）

1. **表格不切开。** 位域表一旦被按长度切碎，「字段 → 比特位 → 定义」的对应关系就断了，检索命中也无法回答。
2. **私有区字符按字体分别映射。** Symbol、MT Extra、Wingdings 各用一套映射表，避免公式符号被统一减 `F000` 后变成错字。
3. **页眉页脚按页高比例划定上下边界**，并要求候选文本在至少 30% 的页面重复出现才判定；横版页同样适用。
4. **标题层级靠三件事判断**：先按基线合并碎片，再用字号差、编号形式、句末标点判定层级，最后把目录行去掉点线和页码后写入 `section_path`。
5. **同一页文本与 bbox 完全相同的行只保留一条**，否则重复内容流会让正文和图标签翻倍。
6. **子块检索、父块召回。** 这是 `hierarchical` 策略的核心：召回精度来自小粒度，答案质量来自大上下文。

---

## 已知限制

- 单机脚本，非服务化；没有并发、鉴权、多用户。
- 评估集只有 25 题，指标波动较大，结论应视为方向性而非定论。
- 答案引用只做到「标注第 N 条上下文」，未做强制引用校验。
- 未处理多轮对话与追问改写。

---

## 技术栈

Python · PyMuPDF / pdfplumber · BAAI/bge-large-zh-v1.5 · bge-reranker-v2-m3 · FAISS · rank-bm25 · jieba · OpenAI 兼容接口 · RAGAS
