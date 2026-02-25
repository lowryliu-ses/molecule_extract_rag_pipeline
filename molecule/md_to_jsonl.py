"""
分子提取管道步骤 1：Markdown → JSONL

将 Markdown 文件转换为分子提取管道所需的 JSONL 格式。
来源：molecule_extraction/Codes/json_processer_helper.py
"""

import json
import os
import shutil
from pathlib import Path
from typing import Optional

from tqdm import tqdm


def write_jsonl_for_markdowns(md_root: str, output_dir: str) -> None:
    """
    将 md_root 目录下所有 Markdown 文件转换为 JSONL 文件，
    写入 output_dir。

    预期目录结构:
        md_root/
            10.1016_xxx/
                10.1016_xxx.md

    输出:
        output_dir/
            10.1016_xxx-0.jsonl
    """
    md_root_path = Path(md_root)
    output_path = Path(output_dir)
    md_files = sorted(list(md_root_path.glob("**/*.md")))
    os.makedirs(output_path, exist_ok=True)

    for md_path in tqdm(md_files, desc="生成 JSONL 文件"):
        doi_name_clean = md_path.stem

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"[警告] 无法读取 '{md_path}': {e}")
            continue

        payload = {
            "text": text,
            "filename": str(md_path),
            "id": doi_name_clean,
        }

        out_path = output_path / f"{doi_name_clean}-0.jsonl"
        try:
            with open(out_path, "w", encoding="utf-8") as out_f:
                json.dump(payload, out_f, ensure_ascii=False)
                out_f.write("\n")
        except Exception as e:
            print(f"[警告] 无法写入 '{out_path}': {e}")


def extract_pdf_files(source_root: Path, destination_folder: Path) -> None:
    """
    从 source_root 的子目录中提取 PDF 文件，
    复制到 destination_folder（去重，跳过已存在的文件名）。
    """
    destination_folder.mkdir(parents=True, exist_ok=True)

    for subfolder in source_root.iterdir():
        if subfolder.is_dir():
            for pdf_file in subfolder.glob("*.pdf"):
                dest_file = destination_folder / pdf_file.name
                if not dest_file.exists():
                    shutil.copy2(pdf_file, dest_file)
                    print(f"已复制: {pdf_file.name}")
                else:
                    print(f"跳过重复: {pdf_file.name}")

    print("完成：所有非重复 PDF 已复制。")


def step_md_to_jsonl(md_dir: str, jsonl_dir: str) -> None:
    """
    步骤 1（分子管道）：Markdown → JSONL

    参数:
        md_dir: 包含 Markdown 文件的根目录
        jsonl_dir: JSONL 文件输出目录
    """
    print("\n" + "=" * 60)
    print("步骤 1 (分子): Markdown 转 JSONL")
    print("=" * 60)

    write_jsonl_for_markdowns(md_dir, jsonl_dir)
    print(f"\nJSONL 文件已生成到: {jsonl_dir}")
