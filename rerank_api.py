"""用网关的 deepseek-flash 重排一阶段检索结果。

一阶段：search_chunks.Retriever 的向量 + BM25 + RRF，取 RERANK_CANDIDATES 条。
二阶段：把查询和候选编号交给 deepseek-flash，由它返回最相关的前 FINAL_TOP_K 个编号。
"""

import argparse
import json
import sys

import torch
from openai import OpenAI

from rerank_bge import FINAL_TOP_K, RERANK_CANDIDATES
from search_chunks import INDEXES, Retriever, snippet

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API_BASE_URL = "XXX"
API_KEY = "XXX"
API_MODEL = "deepseek-flash"
API_MAX_TOKENS = 4096
API_MAX_CHARS = 500

SYSTEM_PROMPT = (
    f"你是检索结果重排器。根据用户问题，从候选片段中选出最相关的 "
    f"{FINAL_TOP_K} 条，按相关性从高到低排列。"
    '只输出 JSON：{"ranking": [编号, ...]}，不要解释。'
)


def build_prompt(query, hits):
    lines = [f"用户问题：{query}", "", "候选片段："]
    for index, hit in enumerate(hits, 1):
        content = " ".join(hit["content"].split())
        if len(content) > API_MAX_CHARS:
            content = content[:API_MAX_CHARS] + " ..."
        lines.append(f"[{index}] {content}")
    lines.append("")
    lines.append(f"输出最相关的前 {FINAL_TOP_K} 个编号。")
    return "\n".join(lines)


def deepseek_ranking(client, query, hits):
    response = client.chat.completions.create(
        model=API_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(query, hits)},
        ],
        temperature=0.0,
        max_tokens=API_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return [int(item) for item in data["ranking"]][:FINAL_TOP_K]


def rerank(results, client):
    for result in results:
        hits = result["hits"]
        ranking = deepseek_ranking(client, result["query"], hits)
        for hit in hits:
            hit["retrieval_rank"] = hit["rank"]
            hit["retrieval_score"] = hit["score"]
        result["hits"] = [hits[index - 1] for index in ranking]
        for rank, hit in enumerate(result["hits"], 1):
            hit["rank"] = rank
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
            print(
                f"[{hit['rank']}] {hit['chunk_id']}  "
                f"页 {hit['page_start']}-{hit['page_end']}  "
                f"{', '.join(hit['block_types'])}  "
                f"RRF #{hit['retrieval_rank']}"
            )
            print(f"    章节：{' > '.join(hit['section_path'])}")
            print(f"    内容：{snippet(hit['content'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index", choices=list(INDEXES), default="semantic"
    )
    parser.add_argument("--query", action="append", required=True)
    args = parser.parse_args()

    retriever = Retriever(args.index)
    results = retriever.search(args.query, top_k=RERANK_CANDIDATES)
    del retriever
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    rerank(results, client)
    print_results(results, args.index)


if __name__ == "__main__":
    main()
