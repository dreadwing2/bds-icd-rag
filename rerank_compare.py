"""同一批一阶段候选分别交给本地 bge-reranker-v2-m3 与 deepseek-flash 重排。

一阶段：search_chunks.Retriever 的向量 + BM25 + RRF，取 RERANK_CANDIDATES 条。
两路重排共用这份候选，各自输出 FINAL_TOP_K 条，不做指标计算。
"""

import argparse
import sys

import torch
from openai import OpenAI

from rerank_api import API_BASE_URL, API_KEY, deepseek_ranking
from rerank_bge import (
    FINAL_TOP_K,
    MODEL_PATH,
    RERANK_CANDIDATES,
    load_reranker,
    rerank_scores,
)
from search_chunks import INDEXES, Retriever, snippet

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def local_ranking(query, hits, tokenizer, model, device):
    scores = rerank_scores(
        query,
        [hit["content"] for hit in hits],
        tokenizer,
        model,
        device,
    )
    order = sorted(range(len(hits)), key=lambda i: -scores[i])[:FINAL_TOP_K]
    return [(index, scores[index]) for index in order]


def print_comparison(query, hits, local, remote):
    print("=" * 80)
    print(f"查询：{query}")
    print("  bge-reranker-v2-m3")
    for rank, (index, score) in enumerate(local, 1):
        hit = hits[index]
        print(
            f"    [{rank}] {score:.4f}  {hit['chunk_id']}  "
            f"RRF #{hit['rank']}  {snippet(hit['content'], 120)}"
        )
    print("  deepseek-flash")
    for rank, index in enumerate(remote, 1):
        hit = hits[index - 1]
        print(
            f"    [{rank}] {hit['chunk_id']}  "
            f"RRF #{hit['rank']}  {snippet(hit['content'], 120)}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index", choices=list(INDEXES), default="semantic"
    )
    parser.add_argument("--query", action="append", required=True)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        sys.exit(f"缺少模型目录 {MODEL_PATH}")

    # 一阶段只召回一次，两路重排共用同一份候选。
    retriever = Retriever(args.index)
    results = retriever.search(args.query, top_k=RERANK_CANDIDATES)
    del retriever
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 本地重排先算完并释放显存，再调用网关。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_reranker(device)
    local_rankings = [
        local_ranking(
            result["query"], result["hits"], tokenizer, model, device
        )
        for result in results
    ]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    for result, local in zip(results, local_rankings):
        remote = deepseek_ranking(client, result["query"], result["hits"])
        print_comparison(result["query"], result["hits"], local, remote)


if __name__ == "__main__":
    main()
