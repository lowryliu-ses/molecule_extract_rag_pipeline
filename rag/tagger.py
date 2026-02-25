"""
RAG 管道步骤 1：文档标记

使用 OpenAI GPT 对 Markdown 文档分配电池科学分类标签，
同时生成文档-标签映射 JSONL 文件和知识图谱三元组 JSON 文件。
"""

import ast
import json
import re
from pathlib import Path
from typing import Dict, List

from openai import OpenAI
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# 标签常量
# ──────────────────────────────────────────────────────────────────────────────

STANDARDIZED_TAGS: List[str] = [
    "lithium_ion", "sodium_ion", "lithium_sulfur", "potassium_ion", "zinc_air",
    "flow_battery", "lithium_metal", "solid_state", "electrolyte", "anode",
    "cathode", "capacity", "cycle_life", "energy_density", "power_density",
    "safety", "thermal_stability", "mechanical_stability", "fast_charging", "cost",
    "sustainability", "recycling", "manufacturing", "battery_management",
    "electric_vehicles", "grid_storage", "wearable_electronics",
    "life_cycle_assessment", "machine_learning", "electrochemical_kinetics",
    "interface_stability", "nanotechnology", "electrode_design", "separator",
    "sodium_metal", "other_types_of_battery", "post_mortem_analysis",
    "electrochemical_test", "coulumbic_efficiency", "rate_capability",
    "cell_design", "system_design", "cell_assembly_process", "material_synthesis",
]

ALLOWED_SET = set(t.lower() for t in STANDARDIZED_TAGS)


# ──────────────────────────────────────────────────────────────────────────────
# 解析辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def _strip_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"```(?:json|python)?", "", s, flags=re.IGNORECASE)
    return s.replace("```", "").strip()


def _extract_first_list_like(s: str) -> str:
    m = re.search(r"\[[\s\S]*?\]", s)
    return m.group(0).strip() if m else "[]"


def _coerce_to_list(parsed) -> List[str]:
    if parsed is None:
        return []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    if isinstance(parsed, str):
        return [p.strip() for p in re.split(r"[,\n]+", parsed) if p.strip()]
    return []


def parse_tags_from_model_output(raw: str) -> List[str]:
    """解析模型输出，先尝试 JSON，失败则回退到 ast.literal_eval"""
    cleaned = _strip_fences(raw)
    blob = _extract_first_list_like(cleaned)
    try:
        return _coerce_to_list(json.loads(blob))
    except Exception:
        try:
            return _coerce_to_list(ast.literal_eval(blob))
        except Exception:
            return []


def normalize_and_filter_tags(tags: List[str], max_tags: int = 5) -> List[str]:
    """小写、去重并验证标签合法性"""
    out: List[str] = []
    for t in tags:
        tl = str(t).strip().lower()
        if tl in ALLOWED_SET and tl not in out:
            out.append(tl)
        if len(out) >= max_tags:
            break
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 文档处理函数
# ──────────────────────────────────────────────────────────────────────────────

def collect_md_paths(base_dir: Path) -> List[Path]:
    """递归查找目录下所有 .md 文件"""
    md_files = sorted(base_dir.rglob("*.md"))
    print(f"找到 {len(md_files)} 个 Markdown 文件在 {base_dir}")
    return md_files


def read_text_safe(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def make_snippet(content: str, head_chars: int = 18000, tail_chars: int = 2000) -> str:
    """截断长文档以适应 LLM 上下文限制"""
    if len(content) <= head_chars + tail_chars:
        return content
    return content[:head_chars] + "\n\n[...TRUNCATED...]\n\n" + content[-tail_chars:]


def assign_tags_via_openai(
    content: str,
    doc_id: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
) -> List[str]:
    """将文档发送给 OpenAI 并返回过滤后的标签列表"""
    snippet = make_snippet(content)
    prompt = (
        "You are an expert battery scientist.\n"
        "Select up to 5 most relevant tags from ALLOWED_TAGS.\n"
        "Return ONLY a JSON array: [\"tag1\", \"tag2\"].\n\n"
        f"ALLOWED_TAGS = {STANDARDIZED_TAGS}\n\n"
        f"DOCUMENT: {snippet}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.1,
        )
        raw_out = (resp.choices[0].message.content or "").strip()
        return normalize_and_filter_tags(parse_tags_from_model_output(raw_out))
    except Exception as e:
        print(f"标记 {doc_id} 时出错: {e}")
        return []


def process_documents(
    md_paths: List[Path],
    doc_tags_file: Path,
    tag_docs_file: Path,
    client: OpenAI,
) -> Dict[str, List[str]]:
    """遍历文件，标记，增量保存结果并构建倒排索引"""
    tag_docs: Dict[str, List[str]] = {t: [] for t in STANDARDIZED_TAGS}

    with open(doc_tags_file, "w", encoding="utf-8") as f_jsonl:
        for path in tqdm(md_paths, desc="标记文档", unit="doc"):
            src = str(path)
            text = read_text_safe(path)
            tags = assign_tags_via_openai(text, src, client)

            json.dump({"source": src, "tags": tags}, f_jsonl, ensure_ascii=False)
            f_jsonl.write("\n")

            for t in tags:
                tag_docs[t].append(src)

    with open(tag_docs_file, "w", encoding="utf-8") as f:
        json.dump(tag_docs, f, indent=2, ensure_ascii=False)

    return tag_docs


def generate_triples(
    tag_docs: Dict[str, List[str]],
    output_file: Path,
    cap: int = 100000,
) -> int:
    """创建 [docA, tag, docB] 三元组用于知识图谱"""
    triples = []
    for tag, docs in tqdm(tag_docs.items(), desc="生成三元组"):
        if len(docs) < 2:
            continue
        count = 0
        for i, d1 in enumerate(docs):
            for d2 in docs[i + 1:]:
                triples.append([d1, tag, d2])
                count += 1
                if count >= cap:
                    break
            if count >= cap:
                break

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)
    return len(triples)


# ──────────────────────────────────────────────────────────────────────────────
# 步骤入口
# ──────────────────────────────────────────────────────────────────────────────

def step_tag_documents(input_dir: Path, artifact_dir: Path) -> Path:
    """
    步骤 1：文档标记

    参数:
        input_dir: 包含 Markdown 文件的目录
        artifact_dir: 工件输出目录（保存标签文件和三元组文件）

    返回:
        document_tags.jsonl 文件路径
    """
    import os
    print("\n" + "=" * 60)
    print("步骤 1: 文档标记")
    print("=" * 60)

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 未设置。请运行: export OPENAI_API_KEY='your-key'")

    client = OpenAI()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    doc_tags_file = artifact_dir / "document_tags.jsonl"
    tag_docs_file = artifact_dir / "tag_document_paths.json"
    triples_file = artifact_dir / "graph_triples.json"

    md_paths = collect_md_paths(input_dir)
    tag_docs_map = process_documents(md_paths, doc_tags_file, tag_docs_file, client)
    total_triples = generate_triples(tag_docs_map, triples_file)

    print(f"\n处理完成！")
    print(f"  文档数: {len(md_paths)}")
    print(f"  三元组数: {total_triples}")
    print(f"  工件保存到: {artifact_dir}")

    return doc_tags_file
