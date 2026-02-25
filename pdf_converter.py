"""
共享 PDF → Markdown 转换模块

合并自：
  - rag_pipeline/run_pipeline.py (步骤0，含artifact目录清理)
  - molecule_extraction/Codes/process_pdf_assb.py (DOI命名模式)

关键改进：
  - 消除重复函数 pdf_files_paths / sanitize_filename / parallel_process_files
  - 修复双重PDF转换问题：use_doi=True 时只转换一次，从生成的Markdown文本中提取DOI
  - use_doi=False（RAG管道默认）：使用原始文件名，保留artifact目录清理逻辑
  - use_doi=True（分子提取管道）：提取DOI作为输出目录/文件名
"""

import os
import re
import shutil
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from docling_core.types.doc import ImageRefMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)
IMAGE_RESOLUTION_SCALE = 2.0


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def sanitize_filename(filename: str) -> str:
    """清理文件名，移除不安全字符"""
    name_without_ext = filename.replace(".pdf", "")
    sanitized = re.sub("clean_[0-9]_", "", name_without_ext)
    sanitized = re.sub(r'[\\/:"*?<>|]+', "_", sanitized)
    return sanitized.strip()


def pdf_files_paths(pdf_folders_path: List[str]) -> List[str]:
    """递归收集所有 PDF 文件路径"""
    pdf_files = []
    for folder_path in pdf_folders_path:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
    return pdf_files


def _build_converter(device: str = "cuda") -> DocumentConverter:
    """构建 Docling PDF 转换器"""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options.device = device
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def _fix_artifacts_dir(output_dir: Path, doc_file_name: str) -> None:
    """
    修复 artifact 目录路径问题：
    如果存在嵌套的 xxx/xxx_artifacts 结构，将其移动到正确位置，
    并递归清理 output_dir 内所有因此产生的空中间目录。
    """
    expected_artifacts_dir = output_dir / f"{doc_file_name}_artifacts"
    nested_artifacts_dirs = list(output_dir.rglob(f"{doc_file_name}_artifacts"))

    for nested_dir in nested_artifacts_dirs:
        if nested_dir == expected_artifacts_dir or not nested_dir.is_dir():
            continue

        start_cleanup_from = nested_dir.parent

        if not expected_artifacts_dir.exists():
            nested_dir.rename(expected_artifacts_dir)
            _log.info(f"移动 artifacts 目录: {nested_dir} -> {expected_artifacts_dir}")
        else:
            for item in nested_dir.iterdir():
                dest = expected_artifacts_dir / item.name
                if item.is_file() and not dest.exists():
                    item.rename(dest)
                elif item.is_dir():
                    if dest.exists():
                        for subitem in item.rglob("*"):
                            rel_path = subitem.relative_to(item)
                            dest_subitem = dest / rel_path
                            dest_subitem.parent.mkdir(parents=True, exist_ok=True)
                            if subitem.is_file() and not dest_subitem.exists():
                                subitem.rename(dest_subitem)
                    else:
                        item.rename(dest)
            try:
                if nested_dir.exists():
                    shutil.rmtree(nested_dir)
            except Exception as e:
                _log.warning(f"删除嵌套目录失败: {nested_dir}, 错误: {e}")

        # 从 nested_dir 的父目录向上逐级清理所有空目录，直到 output_dir 为止
        current = start_cleanup_from
        while current != output_dir:
            try:
                if current.exists() and not any(current.iterdir()):
                    current.rmdir()
                    _log.debug(f"清理空中间目录: {current}")
                else:
                    break
            except Exception as e:
                _log.warning(f"清理中间目录失败: {current}, 错误: {e}")
                break
            current = current.parent


def _fix_md_image_paths(md_filename: Path, doc_file_name: str) -> None:
    """更新 Markdown 文件中的图片路径，移除多余的嵌套层级"""
    if not md_filename.exists():
        return
    md_content = md_filename.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        rf"paper_md/{re.escape(doc_file_name)}/{re.escape(doc_file_name)}_artifacts/"
    )
    fixed = pattern.sub(f"{doc_file_name}_artifacts/", md_content)
    md_filename.write_text(fixed, encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# 核心转换函数
# ──────────────────────────────────────────────────────────────────────────────

def process_pdf_and_save(
    pdf_file_path: str,
    root_output_dir: str,
    use_doi: bool = False,
) -> Optional[Path]:
    """
    将单个 PDF 转换为 Markdown 并保存。

    参数:
        pdf_file_path: 输入 PDF 的完整路径
        root_output_dir: 输出根目录；每个 PDF 将在此目录下创建独立子目录
        use_doi: 若为 True，调用 GPT 从生成的 Markdown 文本中提取 DOI，
                 并以 DOI 作为输出目录和文件名（分子提取管道使用）；
                 若为 False，使用原始文件名（RAG 管道使用）。

    返回:
        生成的 Markdown 文件路径，失败时返回 None。
    """
    logging.basicConfig(level=logging.INFO)
    input_doc_path = Path(pdf_file_path)
    original_name = os.path.basename(pdf_file_path).replace(".pdf", "")

    doc_converter = _build_converter(device="cuda")

    start_time = time.time()
    conv_res = doc_converter.convert(input_doc_path)
    elapsed = time.time() - start_time
    _log.info(f"PDF 转换完成: {input_doc_path.name}，耗时 {elapsed:.2f} 秒")

    if use_doi:
        # 从已生成的 Markdown 文本中提取 DOI，避免二次 PDF 转换
        from molecule.extract_metadata import call_gpt_extract
        md_text = conv_res.document.export_to_markdown(image_mode=ImageRefMode.REFERENCED)
        doi = None
        for attempt in range(10):
            try:
                extracted = call_gpt_extract(md_text)
                doi = extracted.get("DOI") or extracted.get("doi")
                if doi:
                    break
            except Exception as e:
                _log.warning(f"第 {attempt+1} 次 DOI 提取失败: {e}")
        if not doi:
            raise ValueError(f"无法从 PDF 中提取 DOI: {pdf_file_path}")
        doc_file_name = sanitize_filename(doi)
    else:
        doc_file_name = original_name

    output_dir = Path(root_output_dir) / doc_file_name
    output_dir.mkdir(parents=True, exist_ok=True)

    md_filename = output_dir / f"{doc_file_name}.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

    # 修复 artifact 目录路径（仅 RAG 管道需要；DOI 模式下结构已正确）
    if not use_doi:
        _fix_artifacts_dir(output_dir, doc_file_name)
        _fix_md_image_paths(md_filename, doc_file_name)

    return md_filename


def parallel_process_files(
    file_paths: List[str],
    root_output_dir: str,
    max_workers: int = 4,
    use_doi: bool = False,
) -> None:
    """并行处理多个 PDF 文件"""
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_pdf_and_save, fp, root_output_dir, use_doi): fp
            for fp in file_paths
        }
        for future in as_completed(futures):
            fp = futures[future]
            try:
                future.result()
                print(f"成功处理: {fp}")
            except Exception as exc:
                print(f"处理文件 {fp} 时出错: {exc}")


def step_pdf_to_markdown(
    pdf_input_paths: List[str],
    output_dir: str,
    max_workers: int = 4,
    use_doi: bool = False,
) -> Optional[Path]:
    """
    顶层步骤函数：扫描 PDF 文件并并行转换为 Markdown。

    参数:
        pdf_input_paths: PDF 文件路径列表（可以是文件或目录）
        output_dir: Markdown 输出目录
        max_workers: 并行 worker 数量
        use_doi: 是否使用 DOI 命名（分子提取管道传 True）

    返回:
        输出目录 Path，或在未找到 PDF 时返回 None。
    """
    print("\n" + "=" * 60)
    print(f"步骤 0: PDF 转 Markdown {'(DOI 命名模式)' if use_doi else '(原文件名模式)'}")
    print("=" * 60)

    pdf_files = []
    for p in pdf_input_paths:
        if os.path.isfile(p) and p.lower().endswith('.pdf'):
            pdf_files.append(p)
        elif os.path.isdir(p):
            pdf_files.extend(pdf_files_paths([p]))
        else:
            print(f"警告: 跳过无效路径: {p}")

    if not pdf_files:
        print("未找到任何 PDF 文件。")
        return None

    print(f"找到 {len(pdf_files)} 个 PDF 文件，使用 {max_workers} 个 worker 处理...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    parallel_process_files(pdf_files, str(output_path), max_workers, use_doi)

    print(f"\nPDF 转 Markdown 完成！输出目录: {output_path}")
    return output_path
