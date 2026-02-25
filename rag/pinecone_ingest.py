"""
RAG 管道步骤 2：Pinecone 向量化

将 Markdown 文档分块后生成混合向量（密集 + 稀疏），
并上传到 Pinecone 向量数据库。
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from tenacity import retry, stop_after_attempt, wait_exponential

# ──────────────────────────────────────────────────────────────────────────────
# 嵌入服务配置
# ──────────────────────────────────────────────────────────────────────────────

EMBED_URL = "http://160.72.54.197/v1/embeddings"
EMBED_MODEL_NAME = "ses_embeddings"
HEADERS = {"accept": "application/json", "Content-Type": "application/json"}


# ──────────────────────────────────────────────────────────────────────────────
# 文本清理辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def sanitize_text_for_api(s: str) -> str:
    """移除不可打印/控制字符以确保嵌入 API 接受有效 JSON"""
    if not s:
        return ""
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", s)
    return s.encode("utf-8", "ignore").decode("utf-8")


def sanitize_for_pinecone_metadata(s: str) -> str:
    """确保文本对 Pinecone 内部元数据存储是 ASCII 安全的"""
    if not s:
        return ""
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", s)
    return s.encode("ascii", "replace").decode("ascii")


def parse_id_number(id_str: str) -> int:
    """从块 ID 中提取数字后缀（例如 'chunk-100' -> 100）"""
    match = re.search(r"(\d+)$", str(id_str))
    if not match:
        raise ValueError(f"ID 格式错误: {id_str}")
    return int(match.group(1))


# ──────────────────────────────────────────────────────────────────────────────
# 标签加载
# ──────────────────────────────────────────────────────────────────────────────

def load_tags_map(jsonl_path: Path) -> Dict[str, List[str]]:
    """从步骤 1 生成的 JSONL 文件构建标签查找字典"""
    tags_map: Dict[str, List[str]] = {}
    if not jsonl_path.exists():
        print(f"警告: 标签文件未找到: {jsonl_path}")
        return tags_map

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                source = entry.get("source")
                tags = entry.get("tags", [])
                if source:
                    tags_map[str(source)] = tags
                    tags_map[Path(source).name] = tags
            except Exception:
                continue
    return tags_map


# ──────────────────────────────────────────────────────────────────────────────
# 向量生成
# ──────────────────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def get_dense_embeddings(texts: List[str]) -> List[List[float]]:
    """通过 POST 请求到本地嵌入服务器生成密集向量"""
    payload = {
        "input": [sanitize_text_for_api(t) for t in texts],
        "model": EMBED_MODEL_NAME,
        "input_type": "passage",
    }
    resp = requests.post(
        EMBED_URL,
        headers=HEADERS,
        data=json.dumps(payload).encode("utf-8"),
        timeout=180,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json().get("data", [])]


def get_sparse_embeddings(pc_client: Pinecone, texts: List[str]) -> List[Dict]:
    """使用 Pinecone 推理模型为混合搜索生成稀疏向量"""
    try:
        sanitized = [sanitize_for_pinecone_metadata(t) for t in texts]
        out = pc_client.inference.embed(
            model="pinecone-sparse-english-v0",
            inputs=sanitized,
            parameters={"input_type": "passage", "return_tokens": True},
        )
        items = out.data if hasattr(out, "data") else out.get("data", [])
        return [
            {
                "indices": getattr(i, "sparse_indices", i.get("sparse_indices", [])),
                "values": getattr(i, "sparse_values", i.get("sparse_values", [])),
            }
            for i in items
        ]
    except Exception as e:
        print(f"警告: 稀疏向量生成失败，回退到空列表。错误: {e}")
        return [{"indices": [], "values": []} for _ in texts]


# ──────────────────────────────────────────────────────────────────────────────
# 步骤入口
# ──────────────────────────────────────────────────────────────────────────────

def step_ingest_to_pinecone(
    input_dir: Path,
    tags_file: Path,
    index_name: str,
    namespace: str,
    start_id: str,
) -> None:
    """
    步骤 2：向量化并上传到 Pinecone

    参数:
        input_dir: 包含 Markdown 文件的目录
        tags_file: 步骤 1 生成的 document_tags.jsonl 路径
        index_name: Pinecone 索引名称
        namespace: Pinecone 命名空间
        start_id: 起始块 ID（例如 'chunk-0'），用于断点续传
    """
    print("\n" + "=" * 60)
    print("步骤 2: Pinecone 向量化")
    print("=" * 60)

    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY 未设置。请运行: export PINECONE_API_KEY=...")

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    tags_map = load_tags_map(tags_file)
    md_paths = sorted(list(input_dir.rglob("*.md")))

    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
    next_id_num = parse_id_number(start_id)

    print(f"开始向量化... 目标命名空间: {namespace}")

    for idx, path in enumerate(md_paths, 1):
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        if not raw_text.strip():
            continue

        chunks = [c for c in splitter.split_text(raw_text) if c.strip()]
        tags = tags_map.get(str(path)) or tags_map.get(path.name) or []

        print(f"[{idx}/{len(md_paths)}] {path.name} | {len(chunks)} 个块")

        # 每批 96 条（针对本地服务器和 Pinecone 限制优化）
        for i in range(0, len(chunks), 96):
            batch_texts = chunks[i: i + 96]
            batch_ids = [f"chunk-{next_id_num + j + 1}" for j in range(len(batch_texts))]

            try:
                dense = get_dense_embeddings(batch_texts)
                sparse = get_sparse_embeddings(pc, batch_texts)

                vectors = []
                for vid, d, s, txt in zip(batch_ids, dense, sparse, batch_texts):
                    v_record = {
                        "id": vid,
                        "values": d,
                        "metadata": {
                            "context": sanitize_for_pinecone_metadata(txt),
                            "source": sanitize_for_pinecone_metadata(str(path)),
                            "tags": tags,
                        },
                    }
                    if s and len(s.get("indices", [])) > 0:
                        v_record["sparse_values"] = s
                    vectors.append(v_record)

                index.upsert(vectors=vectors, namespace=namespace)
                next_id_num += len(batch_texts)

            except Exception as e:
                print(f"在 {batch_ids[0]} 处发生错误: {e}")
                raise

    print(f"向量化完成。最后 ID: chunk-{next_id_num}")
