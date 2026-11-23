"""用 bge-reranker-v2-m3 重排一阶段检索结果。

一阶段：search_chunks.Retriever 的向量 + BM25 + RRF，取 RERANK_CANDIDATES 条。
二阶段：把「查询, 子块正文」逐对送入 cross-encoder，按相关性分数取 FINAL_TOP_K 条。
层级检索仍以子块为打分单位，父块只在展示时作为上下文。
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast

from search_chunks import INDEXES, Retriever, snippet

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "bge-reranker-v2-m3"

RERANK_CANDIDATES = 30
FINAL_TOP_K = 5
MAX_LENGTH = 512
BATCH_SIZE = 16  # reranker一批处理多少个召回候选chunk


def load_reranker(device):
    # 模型自带 tokenizer.json，直接走 fast tokenizer。
    # 当前环境的 sentencepiece 0.2.2 一导入就崩溃，不能走 XLMRobertaTokenizer。
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(MODEL_PATH / "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH), torch_dtype=dtype
    ).to(device).eval()
    return tokenizer, model


def rerank_scores(query, texts, tokenizer, model, device):
    scores = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        enc = tokenizer(
            [query] * len(batch),  # 用户提问(把同一个查询复制成和当前批次候选一样多份)
            batch,  # 候选chunk
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc).logits.view(-1).float()  # 相当于一query对多候选的分数
        scores.extend(torch.sigmoid(logits).cpu().tolist())  # 每个候选分数化为0到1
    return scores


def rerank(results, tokenizer, model, device):
    for result in results:
        hits = result["hits"]
        scores = rerank_scores(
            result["query"],
            [hit["content"] for hit in hits],
            tokenizer,
            model,
            device,
        )
        for hit, score in zip(hits, scores):
            hit["retrieval_rank"] = hit["rank"]
            hit["retrieval_score"] = hit["score"]
            hit["rerank_score"] = score

        hits.sort(key=lambda hit: -hit["rerank_score"])  # 等效于按logits高低排序
        result["hits"] = hits[:FINAL_TOP_K]
        for rank, hit in enumerate(result["hits"], 1):
            hit["rank"] = rank
            hit["rerank_score"] = round(hit["rerank_score"], 4)
    return results


def print_results(results, strategy):
    print(
        f"策略 {strategy}  查询 {len(results)} 条  "
        f"重排候选 {RERANK_CANDIDATES}  返回 {FINAL_TOP_K}"
    )
    for result in results:
        print("=" * 80)
        print(f"查询：{result['query']}")
        for hit in result["hits"]:
            line = (
                f"[{hit['rank']}] rerank {hit['rerank_score']:.4f}  "
                f"{hit['chunk_id']}  页 {hit['page_start']}-{hit['page_end']}  "
                f"{', '.join(hit['block_types'])}  "
                f"RRF #{hit['retrieval_rank']} {hit['retrieval_score']:.4f}"
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
    parser.add_argument(
        "--index", choices=list(INDEXES), default="semantic"
    )
    parser.add_argument("--query", action="append", required=True)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        sys.exit(f"缺少模型目录 {MODEL_PATH}")

    # 一阶段召回。检索完成后释放向量模型，避免与 reranker 同时占显存。
    retriever = Retriever(args.index)  # 向量化query耗显存不多，但初始化时把 BGE 加载到显卡上，会占大量显存
    results = retriever.search(args.query, top_k=RERANK_CANDIDATES)
    del retriever
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 二阶段重排。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_reranker(device)

    rerank(results, tokenizer, model, device)
    print_results(results, args.index)


if __name__ == "__main__":
    main()
