"""把 chunks/*.json 的 content 编码成向量，并写入 FAISS 索引。

模型 BAAI/bge-large-zh-v1.5，CLS 池化，L2 归一化，最长 512 token。
文档侧不加查询指令前缀；查询侧由检索脚本自行添加。
父块不生成向量，命中子块后按 parent_id 回查父块正文。
"""

import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
CHUNKS_DIR = ROOT / "chunks"
EMB_DIR = ROOT / "embeddings"

FILES = [
    ("semantic", "semantic.json"),
    ("hierarchical_children", "hierarchical_children.json"),
]

MODEL_NAME = "BAAI/bge-large-zh-v1.5"
MODEL_PATH = ROOT / "models" / "bge-large-zh-v1.5"

BATCH_SIZE = 32
MAX_LENGTH = 512


def count_truncated(texts: list[str], tokenizer, max_length: int) -> int:
    n = 0
    for text in texts:
        ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        if len(ids) > max_length:
            n += 1
    return n


def embed_texts(texts, tokenizer, model, device, batch_size, max_length):
    vectors = []
    t0 = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]  # texts是chunks列表
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():  # 关闭梯度记录
            out = model(**enc).last_hidden_state[:, 0]  # 取[cls]作为整句话的表示
            out = F.normalize(out.float(), p=2, dim=1)  # 模长化为1
        vectors.append(out.cpu().numpy())

        done = min(start + batch_size, len(texts))
        if done == len(texts) or done % (batch_size * 20) == 0:
            elapsed = time.perf_counter() - t0
            print(f"    {done}/{len(texts)}  {done / elapsed:.1f} 条/s")

    return np.vstack(vectors).astype(np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"模型 {MODEL_NAME}  设备 {device}  batch {BATCH_SIZE}  最长 {MAX_LENGTH}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    # 统计截断时要完整分词超过 512 token 的文本，避免 tokenizer 逐条告警。
    tokenizer.model_max_length = 10**6
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(str(MODEL_PATH), torch_dtype=dtype)
    model.to(device).eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    report = {
        "model": MODEL_NAME,
        "model_path": str(MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "device": str(device),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "files": {},
    }

    for name, filename in FILES:  # semantic/hierarchical(_children)
        path = CHUNKS_DIR / filename

        # 读chunk.json
        chunks = json.loads(path.read_text(encoding="utf-8"))

        # 获取待向量化的正文，及其对应标识
        texts = [c["content"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]

        # 统计长度超过BGE max_length的文本数
        truncated = count_truncated(texts, tokenizer, MAX_LENGTH)
        print(f"[{name}] {len(texts)} 条，超过 {MAX_LENGTH} token 的 {truncated} 条")

        # 向量化
        t0 = time.perf_counter()
        vectors = embed_texts(
            texts, tokenizer, model, device, BATCH_SIZE, MAX_LENGTH
        )
        elapsed = time.perf_counter() - t0

        # 保存标识与 FAISS 索引
        (EMB_DIR /f"{name}_ids.json").write_text(
            json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 向量已归一化，内积等于余弦相似度
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(EMB_DIR /f"{name}.faiss"))

        report["files"][name] = {
            "chunks": len(texts),
            "dim": int(vectors.shape[1]),
            "truncated": truncated,
            "seconds": round(elapsed, 2),
            "chunks_per_second": round(len(texts) / elapsed, 1),
            "index": f"{name}.faiss",
            "index_type": "IndexFlatIP",
        }
        print(f"    -> {EMB_DIR /(name + '.faiss')}  {vectors.shape}  {elapsed:.1f}s")

    if device.type == "cuda":
        report["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

    (EMB_DIR /"_embed_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
