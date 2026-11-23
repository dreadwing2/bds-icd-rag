"""检索 + 可选重排，再让 deepseek-flash 依据上下文生成答案。

一阶段：向量 + BM25 + RRF 召回 RERANK_CANDIDATES 条。
二阶段：按 --reranker 选择 none（直接取 RRF 前 5）、bge 或 api 重排。
三阶段：把整理后的上下文交给 deepseek-flash 生成答案。
层级检索用父块正文作为生成上下文，semantic 用切块正文。
"""

import argparse
import sys

import torch
from openai import OpenAI

from rerank_api import (
    API_BASE_URL,
    API_KEY,
    API_MAX_TOKENS,
    API_MODEL,
    rerank as rerank_api,
)
from rerank_bge import (
    FINAL_TOP_K,
    MODEL_PATH,
    RERANK_CANDIDATES,
    load_reranker,
    rerank as rerank_bge,
)
from search_chunks import INDEXES, Retriever

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SYSTEM_PROMPT = (
    "你是北斗导航领域的问答助手。只根据给定资料回答；"
    "资料不足时直接说明无法回答。回答中用 [编号] 标注引用来源。"
)


def build_context(hits):
    lines = ["资料："]
    for index, hit in enumerate(hits, 1):
        content = hit.get("parent_content", hit["content"])
        page_start = hit.get("parent_page_start", hit["page_start"])
        page_end = hit.get("parent_page_end", hit["page_end"])
        section = " > ".join(
            hit.get("parent_section_path", hit["section_path"])
        )
        lines.append(
            f"[{index}] 章节：{section}  页码：{page_start}-{page_end}"
        )
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


def generate_answer(client, query, hits):
    response = client.chat.completions.create(
        model=API_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"问题：{query}\n\n{build_context(hits)}",
            },
        ],
        temperature=0.0,
        max_tokens=API_MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def print_answer(query, hits, answer):
    print("=" * 80)
    print(f"查询：{query}")
    print("回答：")
    print(answer)
    print("来源：")
    for index, hit in enumerate(hits, 1):
        line = (
            f"  [{index}] {hit['chunk_id']}  "
            f"页 {hit['page_start']}-{hit['page_end']}  "
            f"{' > '.join(hit['section_path'])}"
        )
        if "parent_id" in hit:
            line += f"  父块 {hit['parent_id']}"
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index", choices=list(INDEXES), default="semantic"
    )
    parser.add_argument(
        "--reranker",
        choices=["none", "bge", "api"],
        default="bge",
        help="none 不重排；bge 本地重排；api 网关重排",
    )
    parser.add_argument("--query", action="append", required=True)
    args = parser.parse_args()

    retriever = Retriever(args.index)
    results = retriever.search(args.query, top_k=RERANK_CANDIDATES)
    del retriever
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    if args.reranker == "bge":
        if not MODEL_PATH.exists():
            sys.exit(f"缺少模型目录 {MODEL_PATH}")
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        tokenizer, model = load_reranker(device)
        rerank_bge(results, tokenizer, model, device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif args.reranker == "api":
        rerank_api(results, client)
    else:
        for result in results:
            result["hits"] = result["hits"][:FINAL_TOP_K]

    for result in results:
        answer = generate_answer(
            client, result["query"], result["hits"]
        )
        print_answer(result["query"], result["hits"], answer)


if __name__ == "__main__":
    main()
