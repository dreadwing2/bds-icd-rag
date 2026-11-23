"""用 RAGAS 评测检索、重排与生成的整条链路。

流程：一阶段召回 RERANK_CANDIDATES 条，二阶段按 --reranker 重排到 FINAL_TOP_K 条，
三阶段由 deepseek-flash 依据上下文作答，四阶段交给 RAGAS 计算四项指标。
semantic 用子块正文作为上下文；hierarchical 用父块正文作为上下文。
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from langchain_openai import ChatOpenAI
from openai import OpenAI
from ragas import RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from LLM_generate import generate_answer  # noqa: E402
from rerank_api import (  # noqa: E402
    API_BASE_URL,
    API_KEY,
    API_MODEL,
    rerank as rerank_api,
)
from rerank_bge import (  # noqa: E402
    FINAL_TOP_K,
    MODEL_PATH as RERANKER_MODEL_PATH,
    RERANK_CANDIDATES,
    load_reranker,
    rerank as rerank_bge,
)
from search_chunks import (  # noqa: E402
    INDEXES,
    MODEL_PATH as BGE_MODEL_PATH,
    QUERY_PREFIX,
    Retriever,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVAL_DIR = ROOT / "eval"
QUESTIONS_PATH = EVAL_DIR / "questions.json"
RESULTS_DIR = EVAL_DIR / "results"

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
METRIC_NAMES = [metric.name for metric in METRICS]

# 评判模型与生成模型分开，避免同一个模型给自己的答案打分。
JUDGE_MODEL = "claude-sonnet-5"

EMBED_BATCH_SIZE = 32
EMBED_MAX_LENGTH = 512


class BGEEmbeddings:
    """把本地 BGE 包装成 RAGAS 需要的向量接口。

    文档侧不加前缀，查询侧加 BGE 指令前缀，与检索脚本保持同一向量空间。
    """

    def __init__(self, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(str(BGE_MODEL_PATH))
        self.tokenizer.model_max_length = 10**6
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.bert = (
            AutoModel.from_pretrained(
                str(BGE_MODEL_PATH), torch_dtype=dtype
            )
            .to(device)
            .eval()
        )
        # RAGAS 会把 model 字段写进用量事件，这里必须是字符串，不能是模型对象。
        self.model = BGE_MODEL_PATH.name

    def _embed(self, texts, prefix=""):
        vectors = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = [
                prefix + text
                for text in texts[start : start + EMBED_BATCH_SIZE]
            ]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=EMBED_MAX_LENGTH,
                return_tensors="pt",
            )
            enc = {key: value.to(self.device) for key, value in enc.items()}
            with torch.inference_mode():
                out = self.bert(**enc).last_hidden_state[:, 0]
                out = F.normalize(out.float(), p=2, dim=1)
            vectors.append(out.cpu().numpy())
        return np.vstack(vectors)

    def embed_documents(self, texts):
        return self._embed(texts).tolist()

    def embed_query(self, text):
        return self._embed([text], QUERY_PREFIX)[0].tolist()

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)

    async def aembed_query(self, text):
        return self.embed_query(text)


def load_questions():
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return data["questions"]


def build_contexts(hits):
    """RAGAS 按条比对上下文，层级检索取父块正文，语义检索取子块正文。"""
    return [hit.get("parent_content", hit["content"]) for hit in hits]


def run_rag(questions, index_name, reranker):
    """跑完召回、重排和生成，返回每题的答案与上下文。"""
    retriever = Retriever(index_name)
    results = retriever.search(
        [question["question"] for question in questions],
        top_k=RERANK_CANDIDATES,
    )
    # 先释放向量模型，给重排模型腾显存。
    del retriever
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    if reranker == "bge":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        tokenizer, model = load_reranker(device)
        rerank_bge(results, tokenizer, model, device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif reranker == "api":
        rerank_api(results, client)
    else:
        for result in results:
            result["hits"] = result["hits"][:FINAL_TOP_K]

    records = []
    for question, result in zip(questions, results):
        hits = result["hits"]
        answer = generate_answer(client, question["question"], hits)
        records.append(
            {
                "qid": question["qid"],
                "type": question["type"],
                "question": question["question"],
                "answer": answer,
                "contexts": build_contexts(hits),
                "ground_truth": question["answer"],
                "hits": [
                    {
                        "rank": hit["rank"],
                        "chunk_id": hit["chunk_id"],
                        "block_ids": hit["block_ids"],
                        "score": hit["score"],
                        "retrieval_rank": hit.get("retrieval_rank"),
                        "rerank_score": hit.get("rerank_score"),
                    }
                    for hit in hits
                ],
            }
        )
        print(f"[{question['qid']}] {question['type']} 已生成", flush=True)

    return records


def score_records(records):
    """调用 RAGAS，返回每题分数与四项均值。"""
    dataset = Dataset.from_list(
        [
            {
                "question": record["question"],
                "answer": record["answer"],
                "contexts": record["contexts"],
                "ground_truth": record["ground_truth"],
            }
            for record in records
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ragas_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=JUDGE_MODEL,
            openai_api_key=API_KEY,
            openai_api_base=API_BASE_URL,
            temperature=0,
        ),
        # 网关对单次 n=3 只返回 1 个结果，拆成 3 次调用才能满足 RAGAS 的 strictness。
        bypass_n=True,
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(BGEEmbeddings(device))

    score = evaluate(
        dataset,
        metrics=METRICS,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        # 评判调用偶尔超过 180 秒，放宽到 600 秒以减少超时型 NaN。
        run_config=RunConfig(timeout=600),
    )

    # ragas 0.4.x 每个指标返回一题一个分，取均值前先去掉 NaN。
    per_question = {}
    means = {}
    for name in METRIC_NAMES:
        values = [
            value
            for value in score[name]
            if value is not None and not (isinstance(value, float) and np.isnan(value))
        ]
        per_question[name] = score[name]
        means[name] = float(np.mean(values)) if values else float("nan")

    return per_question, means


def save_results(index_name, reranker, records, per_question, means):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "config": {
            "index": index_name,
            "reranker": reranker,
            "generator": API_MODEL,
            "judge": JUDGE_MODEL,
            "question_count": len(records),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "mean": means,
        "per_question": per_question,
        "records": records,
    }
    path = RESULTS_DIR / f"{index_name}_{reranker}.json"
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"结果已保存：{path}")


def print_summary(index_name, reranker, means):
    print("=" * 60)
    print(f"配置：{index_name} / {reranker}  评判模型：{JUDGE_MODEL}")
    for name in METRIC_NAMES:
        print(f"  {name:<20} {means[name]:.4f}")


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
    args = parser.parse_args()

    if args.reranker == "bge" and not RERANKER_MODEL_PATH.exists():
        sys.exit(f"缺少模型目录 {RERANKER_MODEL_PATH}")

    questions = load_questions()
    records = run_rag(questions, args.index, args.reranker)
    per_question, means = score_records(records)
    save_results(args.index, args.reranker, records, per_question, means)
    print_summary(args.index, args.reranker, means)


if __name__ == "__main__":
    main()
