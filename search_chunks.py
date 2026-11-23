"""用 BGE 向量与 BM25 召回切块，再按 RRF 融合。

向量侧：FAISS 余弦相似度。
词侧：jieba 分词后的 BM25Okapi。
两路各取候选，按名次累加 1/(60+rank)，固定融合。
semantic：检索语义切块。
hierarchical：检索子块，再按 parent_id 取父块正文作为上下文。
查询文本加 BGE 指令前缀；文档侧向量由 embed_chunks.py 生成，未加前缀。
"""

import argparse
import json
import re
import sys
from pathlib import Path

import faiss
import jieba
import numpy as np
import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "bge-large-zh-v1.5"
EMB_DIR = ROOT / "embeddings"
CHUNKS_DIR = ROOT / "chunks"

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

INDEXES = {
    "semantic": {
        "index": "semantic.faiss",
        "ids": "semantic_ids.json",
        "chunks": "semantic.json",
        "parents": None,
    },
    "hierarchical": {
        "index": "hierarchical_children.faiss",
        "ids": "hierarchical_children_ids.json",
        "chunks": "hierarchical_children.json",
        "parents": "hierarchical_parents.json",
    },
}


RRF_K = 60
# 两路各自召回深度，与 RRF 最终返回条数解耦。
CANDIDATE_K = 100


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


# 拉丁字母、数字和型号保持整词，中文交给 jieba 分词。
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._\-/][a-z0-9]+)*|[一-鿿]+")


def tokenize(text):
    tokens = []
    for match in TOKEN_RE.finditer(text.lower()):
        piece = match.group()
        if "一" <= piece[0] <= "鿿":  # 判断是否是中文
            tokens.extend(jieba.lcut(piece))
        else:  # 英文
            tokens.append(piece)
    return tokens


class Retriever:
    def __init__(self, strategy):
        cfg = INDEXES[strategy]
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 加载模型。查询与文档使用同一个模型，才能落在同一向量空间。
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        self.tokenizer.model_max_length = 10**6
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            str(MODEL_PATH), torch_dtype=dtype
        ).to(self.device).eval()

        # 索引与 chunk_id 顺序一一对应，row 号通过 ids 映射回 chunk_id。
        self.index = faiss.read_index(str(EMB_DIR / cfg["index"]))  # 读.faiss文件
        self.ids = load_json(EMB_DIR / cfg["ids"])

        # 检索结果只带 chunk_id，正文和来源块从切块文件回查。
        chunks = load_json(CHUNKS_DIR / cfg["chunks"])
        self.chunks = {c["chunk_id"]: c for c in chunks}

        # BM25 与 FAISS 共用 ids 顺序，融合时按 chunk_id 对齐。
        self.bm25 = BM25Okapi(
            [tokenize(self.chunks[chunk_id]["content"]) for chunk_id in self.ids]
        )

        # 父块不建索引，只用于命中子块后补上下文。
        self.parents = {}
        if cfg["parents"]:
            parents = load_json(CHUNKS_DIR / cfg["parents"])
            self.parents = {p["parent_id"]: p for p in parents}

    def encode_queries(self, queries, batch_size=32, max_length=512):
        vectors = []
        for start in range(0, len(queries), batch_size):
            batch = [QUERY_PREFIX + q for q in queries[start : start + batch_size]]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                out = self.model(**enc).last_hidden_state[:, 0]
                out = F.normalize(out.float(), p=2, dim=1)
            vectors.append(out.cpu().numpy())
        return np.vstack(vectors).astype(np.float32)

    def _dense_ranking(self, queries, k):
        # 向量已归一化，内积即余弦相似度，FAISS 返回的 score 可直接比较。
        scores, rows = self.index.search(self.encode_queries(queries), k)
        return [
            [(self.ids[int(row)], float(score)) for score, row in zip(qs, qr)]
            for qs, qr in zip(scores, rows)
        ]

    def _bm25_ranking(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:k]
        return [
            (self.ids[int(i)], float(scores[i]))
            for i in order
            if scores[i] > 0
        ]

    def _rrf(self, dense, bm25, top_k):
        # 两路分数尺度不同，只按名次累加 1/(60+rank)。
        fused = {}
        dense_scores = {}
        bm25_scores = {}
        for rank, (chunk_id, score) in enumerate(dense, 1):
            fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (RRF_K + rank)
            dense_scores[chunk_id] = score
        for rank, (chunk_id, score) in enumerate(bm25, 1):
            fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (RRF_K + rank)
            bm25_scores[chunk_id] = score

        ordered = sorted(fused, key=lambda cid: (-fused[cid], cid))[:top_k]
        return [
            (cid, fused[cid], dense_scores.get(cid), bm25_scores.get(cid))
            for cid in ordered
        ]

    # 把检索命中的“一行结果”组装成完整的返回对象
    def _make_hit(self, chunk_id, rank, score, dense_score=None, bm25_score=None):
        # 行号已在调用处换成 chunk_id，这里只组装展示与引用需要的字段。
        chunk = self.chunks[chunk_id]
        hit = {
            "rank": rank,
            "score": round(float(score), 4),
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "content": chunk["content"],
            "block_ids": chunk["block_ids"],
            "block_types": chunk["block_types"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "section_path": chunk["section_path"],
        }
        if dense_score is not None:
            hit["dense_score"] = round(dense_score, 4)
        if bm25_score is not None:
            hit["bm25_score"] = round(bm25_score, 4)

        # 层级检索以子块为单位，命中后补父块正文作为上下文。
        if self.parents:
            parent = self.parents[chunk["parent_id"]]
            hit["parent_id"] = chunk["parent_id"]
            hit["parent_content"] = parent["content"]
            hit["parent_block_ids"] = parent["block_ids"]
            hit["parent_page_start"] = parent["page_start"]
            hit["parent_page_end"] = parent["page_end"]
            hit["parent_section_path"] = parent["section_path"]
        return hit

    def search(self, queries, top_k=5):
        top_k = min(top_k, self.index.ntotal)  # index.add的总向量条数
        candidate_k = min(CANDIDATE_K, self.index.ntotal)

        dense = self._dense_ranking(queries, candidate_k)  # 放循环外：query embed可批处理，比逐条算快
        results = []
        for query, dense_ranking in zip(queries, dense):
            ranking = self._rrf(
                dense_ranking,
                self._bm25_ranking(query, candidate_k),  # 无法批处理，故放循环内
                top_k,
            )
            hits = [
                self._make_hit(cid, rank, score, dense_score, bm25_score)
                for rank, (cid, score, dense_score, bm25_score) in enumerate(
                    ranking, 1
                )
            ]
            results.append({"query": query, "hits": hits})
        return results


def snippet(text, limit=300):
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + " ..."


def print_results(results, strategy):
    print(f"策略 {strategy}  查询 {len(results)} 条")
    for result in results:
        print("=" * 80)
        print(f"查询：{result['query']}")
        for hit in result["hits"]:
            line = (
                f"[{hit['rank']}] {hit['score']:.4f}  {hit['chunk_id']}  "
                f"页 {hit['page_start']}-{hit['page_end']}  {', '.join(hit['block_types'])}"
            )
            scores = []
            if "dense_score" in hit:
                scores.append(f"dense {hit['dense_score']:.4f}")
            if "bm25_score" in hit:
                scores.append(f"bm25 {hit['bm25_score']:.4f}")
            if scores:
                line += "  " + "  ".join(scores)
            print(line)
            print(f"    章节：{' > '.join(hit['section_path'])}")
            print(f"    内容：{snippet(hit['content'])}")
            if "parent_content" in hit:
                print(
                    f"    父块：{hit['parent_id']}  "
                    f"页 {hit['parent_page_start']}-{hit['parent_page_end']}"
                )
                print(f"    父块内容：{snippet(hit['parent_content'], 500)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", choices=list(INDEXES), default="semantic")
    parser.add_argument("--query", action="append", required=True)
    args = parser.parse_args()

    retriever = Retriever(args.index)
    results = retriever.search(args.query)
    print_results(results, args.index)


if __name__ == "__main__":
    main()
