# 一、知识库语料介绍

本项目选取七份 PDF 作为外部知识库语料进行RAG实验：
《北斗卫星导航系统空间信号接口控制文件 公开服务信号B1C（1.0版）》
《北斗卫星导航系统空间信号接口控制文件 公开服务信号B1I（3.0版）》
《北斗卫星导航系统空间信号接口控制文件 星基增强服务信号BDSBAS-B1C（1.0版）》
《北斗卫星导航系统空间信号接口控制文件 精密单点定位服务信号PPP-B2b（1.0版）》
《北斗卫星导航系统发展报告（3.0版）》
《北斗卫星导航系统白皮书（2016）》
《北斗卫星导航系统公开服务性能规范（3.0版）》
共 329 页、1304 个 block。结构上分三类：前四份 ICD 表格与图数量多，合计 250 张表、68 个图、71 张位域表；两份政策报告（发展报告、白皮书）以连续长文为主，合计 3 张表、0 个图；一份公开服务性能规范有 27 张表、0 个图。七份文档的正文均带文本层，仅白皮书第 1 页为扫描封面，没有文本层。主流程不需要全量 OCR；代码只在文本层缺失的页面触发 OCR，本语料仅白皮书第 1 页触发。

解析侧处理细节：
 - 表格保留合并单元格填充结果（499 处）
 - 页眉页脚不进入正文
 - 图块只保留题注和图内文字标签，不解析图像像素内容
 - 页眉页脚按页高比例划定上下边界，并要求候选文本在至少 30% 的页面重复出现，横版页同样适用
 - 私有区字符按字体分别映射，Symbol、MT Extra、Wingdings 各用一套表，避免公式符号被统一减 F000 后变成错字
 - 标题先按基线合并碎片，再用字号差、编号形式和句末标点判断层级，目录行去掉点线和页码后写入 section_path
 - 同一页文本和 bbox 完全相同的行只保留一条，避免重复内容流让正文和图标签翻倍

# 二、数据结构设计

## 1. parsed/<doc_id>.json

### 1.1 顶层

```json
{
  "meta": {},
  "blocks": []
}
```

### 1.2 meta

| 字段              | 类型         | 含义         |
|-------------------|--------------|--------------|
| `doc_id`          | string       | 文档标识     |
| `source_file`     | string       | 源 PDF 路径  |
| `pages`           | int          | 物理页数     |
| `body_font_size`  | float        | 正文基准字号 |
| `running_headers` | list[string] | 页眉文本     |
| `running_footers` | list[string] | 页脚文本     |
| `ocr_pages`       | list[int]    | OCR 页号     |

### 1.3 block

| 字段           | 类型           | 含义                               |
|----------------|----------------|------------------------------------|
| `block_id`     | string         | 块标识                             |
| `block_type`   | string         | `title`、`text`、`table`、`figure` |
| `page_num`     | int            | 物理页号                           |
| `printed_page` | string 或 null | 印刷页码                           |
| `section_path` | list[string]   | 章节路径                           |
| `content`      | string         | 块文本                             |
| `bbox`         | list[float]    | `[x0, y0, x1, y1]`，单位 point     |
| `is_ocr`       | bool           | OCR 标记                           |


`table` 专属字段：

| 字段            | 类型               | 含义           |
|-----------------|--------------------|----------------|
| `caption`       | string             | 题注           |
| `n_rows`        | int                | 行数           |
| `n_cols`        | int                | 列数           |
| `layout`        | string             | 布局类型       |
| `merged_filled` | int                | 填充的合并格数 |
| `rows`          | list[list[string]] | 单元格文本     |
| `labels`        | list[string]       | 标签           |

`figure` 专属字段：

| 字段         | 类型         | 含义           |
|--------------|--------------|----------------|
| `caption`    | string       | 题注           |
| `labels`     | list[string] | 标签           |
| `n_labels`   | int          | 标签数         |
| `has_vector` | bool         | 是否含矢量绘图 |
| `has_image`  | bool         | 是否含位图     |

## 2. chunks/*.json

| `strategy`     | 切块方式                               | 召回与上下文                               |
|----------------|----------------------------------------|--------------------------------------------|
| `semantic`     | 按标题和章节变化切分，正文上限 800 字   | 直接召回切块                               |
| `hierarchical` | 父块上限 1600 字，子块上限 400 字       | 召回子块，命中后按 `parent_id` 回填父块正文 |

| 字段           | 类型         | `semantic` | `hierarchical` | 含义                           |
|----------------|--------------|------------|----------------|--------------------------------|
| `chunk_id`     | string       | ✓          | ✓              | 切块标识                       |
| `parent_id`    | string       |            | ✓              | 父块标识；子块中指向所属父块   |
| `doc_id`       | string       | ✓          | ✓              | 文档标识                       |
| `strategy`     | string       | ✓          | ✓              | 切块策略                       |
| `content`      | string       | ✓          | ✓              | 文本                           |
| `block_ids`    | list[string] | ✓          | ✓              | 来源块标识                     |
| `page_start`   | int          | ✓          | ✓              | 起始页                         |
| `page_end`     | int          | ✓          | ✓              | 结束页                         |
| `section_path` | list[string] | ✓          | ✓              | 章节路径                       |
| `block_types`  | list[string] | ✓          | ✓              | 块类型                         |

`chunk.content` ：表格块依次拼接题注、Markdown 表格和表格标签；正文块按段落拼接。
`block_ids`、`page_start/page_end`、`section_path` 分别保留来源块、页码范围和章节路径，供引用与回溯。
`hierarchical` 的父块以 `parent_id` 为主键，子块以 `chunk_id` 为主键，并通过 `parent_id` 关联。

# 三、本地embedding配置

向量化使用 `BAAI/bge-large-zh-v1.5`，取最后一层 CLS 向量，L2 归一化，维度 1024。文档侧不加查询指令，查询侧由检索脚本添加 BGE 指令前缀，保证查询与文档落在同一向量空间。批大小 32，最大长度 512 token，超长文本截断。

| 索引                    | 切块数  | 维度 | 超长截断数  | 
|-------------------------|-------:|-----:|-----------:|
| `semantic`              |    895 | 1024 |         83 | 
| `hierarchical_children` |   1080 | 1024 |         20 |    

# 四、检索配置

查询侧与文档侧使用同一个 BGE 模型，查询前加指令前缀 `为这个句子生成表示以用于检索相关文章：`，取 CLS 向量并做 L2 归一化，维度 1024，批大小 32，最大长度 512 token。两路召回始终同时启用：
 - 向量dense路用 FAISS `IndexFlatIP` 检索；
 - BM25路用 jieba 分词后计算 `BM25Okapi`，英文、数字和型号保持整词。
 RRF不看两路分数而是仅以两路排名进行融合排序

# 五、reranker对比配置

一阶段 RRF 输出 30 条候选，重排阶段在这 30 条上比较三种配置，最终都保留 5 条作为生成上下文。

| 配置   | 重排方式                                                                 |
|--------|--------------------------------------------------------------------------|
| `none` | 不重排，直接取 RRF 前 5 条，作为基线                                     |
| `bge`  | 本地 `BAAI/bge-reranker-v2-m3` 对查询与候选逐对打分，按分数排序取前 5 条 |
| `api`  | 网关 `deepseek-flash` 对 30 条候选整体排序，返回前 5 个编号              |

 - `bge` 使用 cross-encoder，把查询和候选正文拼成一对输入，最大长度 512 token，批大小 16，输出 logits 经 sigmoid 转为 0 到 1 的分数。`hierarchical` 仍以子块作为打分单位，父块只在命中后补上下文。
 - `api` 把 30 条候选按编号列出，每条正文截断到 500 字，要求模型只返回 `{"ranking": [...]}`，温度设为 0。

# 六、LLM生成与 RAGAS 评测配置

生成模型固定为 `deepseek-flash`，温度 0，最大输出 4096 token，不做多模型对比。上下文按 `[编号]` 逐条拼接，每条带章节路径和页码；`hierarchical` 使用父块正文，`semantic` 使用切块正文。系统提示要求只依据资料回答，资料不足时明确说明无法回答，并用 `[编号]` 标注引用。

评测使用 RAGAS，固定 25 道题，题型包括表格、正文和跨块问题，计算 `faithfulness`、`answer_relevancy`、`context_precision` 和 `context_recall` 四项指标。

评判模型为 `claude-sonnet-5`，与生成模型分开。`RunConfig(timeout=600)`，逐题计算四项分数，剔除 NaN 后取均值。六种配置为 `semantic` 和 `hierarchical` 各配 `none`、`bge`、`api` 三种重排。

# 七、对比实验

六种配置均使用每路 100 条召回、RRF 前 30 条，使用同一批 25 道题、同一生成模型和同一评判模型，仅改变切块策略与重排方式。

| 索引           | 重排   | faithfulness | answer_relevancy | context_precision | context_recall |
|----------------|--------|-------------:|-----------------:|------------------:|---------------:|
| `semantic`     | `none` |       0.9640 |           0.6880 |            0.8567 |         0.9657 |
| `semantic`     | `bge`  |       0.9407 |           0.7108 |            0.8367 |         0.9443 |
| `semantic`     | `api`  |       0.8897 |           0.7410 |            0.9685 |         0.9943 |
| `hierarchical` | `none` |       0.9533 |           0.7414 |            0.8526 |         1.0000 |
| `hierarchical` | `bge`  |       0.9012 |           0.7252 |            0.8149 |         0.9943 |
| `hierarchical` | `api`  |       0.9207 |           0.7583 |            0.9508 |         1.0000 |

 - 按四项算术平均由高到低：
`hierarchical / api` 0.9074 > `semantic / api`     0.8984 > `hierarchical / none` 0.8868 >
`semantic / none`    0.8686 > `hierarchical / bge` 0.8589 > `semantic / bge`      0.8581

 - 本地 `BAAI/bge-reranker-v2-m3` 重排未带来预期增益，与RRF相比，甚至有副作用。

 - 重排模式整体比较：`api` > `none` > `bge`
`api` 的 answer_relevancy、context_precision、context_recall 高于或持平 `none`，faithfulness 有所下降。
然而，本实验 `api` 重排与LLM生成均使用 `deepseek-flash`，重排增益不能完全归因于独立的重排能力。后续若要比较重排器本身，应增加独立的重排模型。

 - `hierarchical` 的 answer_relevancy、context_recall 高于 `semantic`，context_precision 有所下降，faithfulness 无稳定优势；`hierarchical` 和 `semantic` 的效果在 `none` 下差异稍大，在 `api` 下差异相对较小。

 - RAGAS 调用外部LLM评判模型时，单题请求耗时超过 `RunConfig(timeout=...)` 就失败，导致该题分数记为 NaN。本实验额外比较了 `RunConfig(timeout=180)` 与 `RunConfig(timeout=600)` 下 `semantic` 的 context_precision 为 NaN 的数量。

| 重排   | 180 秒 NaN 数 | 600 秒 NaN 数 |
|--------|--------------:|--------------:|
| `none` |             5 |             0 |
| `bge`  |             5 |             0 |
| `api`  |             6 |             0 |

600 秒下均为 0，说明 NaN 来自评判超时，不是检索或生成失败。这表明RAGAS环节的 `RunConfig(timeout=...)` 不可设置过小，否则会出现非预期NAN值影响真实结果。

## 单题结果展示

`hierarchical / api` 处理 query21 ，四项分数为 faithfulness 1.000、answer_relevancy 0.944、context_precision 1.000、context_recall 1.000。

| 字段     | 内容                                                                                                                                                                                         |
|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 问题     | 北斗系统公开服务性能规范涵盖哪几类服务？                                                                                                                                                     |
| 参考答案 | RNSS、PPP、RSMC、SAR、GAS。                                                                                                                                                                  |
| 模型回答 | 根据资料，北斗系统公开服务性能规范涵盖以下五类服务：定位导航授时（RNSS）、精密单点定位（PPP）、区域短报文通信（RSMC）、国际搜救（SAR）和地基增强（GAS）；其他服务性能将在后续版本中提供。[1] |

模型回答中的 `[1]` 对应重排后第 1 条上下文。5 条上下文的重排后名次、来源与 RRF 名次如下：

| 重排后名次  | chunk_id                         | RRF 分 | RRF 名次 | 较 RRF 名次 |
|-----------:|:---------------------------------|-------:|---------:|------------:|
|          1 | `北斗卫星导航系统:hier:c00033`    | 0.0290 |        3 |          ↑2 |
|          2 | `北斗卫星导航系统:hier:c00002`    | 0.0277 |       11 |          ↑9 |
|          3 | `bds-dev-report-3.0:hier:c00024` | 0.0278 |        9 |          ↑6 |
|          4 | `北斗卫星导航系统:hier:c00042`    | 0.0220 |       25 |         ↑21 |
|          5 | `北斗卫星导航系统:hier:c00102`    | 0.0256 |       17 |         ↑12 |

[1]上下文正文：

> 1 范围
>
> 本文档给出了北斗系统定位导航授时（RNSS）、精密单点定位（PPP）、区域短报文通信（RSMC）、国际搜救（SAR）和地基增强（GAS）的服务性能，其他服务性能在后续版本中提供。

# 八、结论

1. 本次 25 题评测中，没有一种配置在四项指标上同时最优，但整体而言，`hierarchical / api` 的四项均值最高，`semantic / api` 次之。

2. 本实验重排模式对比下，`api` > `none` > `bge` 。`api` 的 answer_relevancy、context_precision、context_recall 高于或持平 `none`，但faithfulness 有所下降；本地 `BAAI/bge-reranker-v2-m3` 重排未带来预期增益。

3.  `hierarchical` 的 answer_relevancy、context_recall 高于 `semantic`，context_precision 稍逊；两种切块方式在 `none` 下的效果差距，稍大于在 `api` 下的效果差距。