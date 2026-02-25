#!/usr/bin/env python3
"""
统一数据管道入口

将分子提取管道（molecule）与 RAG 管道（rag）合并为单一脚本，
共享 PDF → Markdown 转换模块，消除重复代码。

管道流程:

  RAG 管道 (--pipeline rag):
    步骤 0 (共享): PDF → Markdown  [pdf_converter.py, 原文件名模式]
    步骤 1: 文档标记               [rag/tagger.py, 使用 OpenAI]
    步骤 2: Pinecone 向量化        [rag/pinecone_ingest.py]
    步骤 3: S3 同步                [rag/s3_sync.py]

  分子提取管道 (--pipeline molecule):
    步骤 0 (共享): PDF → Markdown  [pdf_converter.py, DOI 命名模式]
    步骤 1: Markdown → JSONL       [molecule/md_to_jsonl.py]
    步骤 2: Snowflake 元数据更新   [molecule/extract_metadata.py]
    步骤 3: 分子提取 + 入库        [molecule/llm_judge/code/]

环境变量:
  OPENAI_API_KEY    — RAG 文档标记 & 分子管道 DOI/元数据提取
  PINECONE_API_KEY  — RAG Pinecone 向量化
  CONFIG_PATH       — config.json 路径（可选，默认查找当前目录）

配置文件 (config.json):
  aws_access_key_id, aws_secret_access_key  — S3 同步
  SNOWFLAKE_*                                — Snowflake 连接（可选，优先使用环境变量）
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────────────────────────────────────
# 通用辅助
# ──────────────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# 步骤执行器：RAG 管道
# ──────────────────────────────────────────────────────────────────────────────

def run_rag_pipeline(args, script_dir: Path) -> None:
    """执行 RAG 管道的所有指定步骤"""
    _section("RAG 管道")

    from pdf_converter import step_pdf_to_markdown

    steps = _resolve_steps(args.steps)
    print(f"执行步骤: {', '.join(steps)}")

    input_dir: Path | None = None

    # 步骤 0: PDF → Markdown（原文件名模式）
    if "0" in steps:
        if not args.pdf_input:
            raise ValueError("步骤 0 需要 --pdf-input 参数。")
        output_path = step_pdf_to_markdown(
            args.pdf_input,
            args.pdf_output or args.input or "./paper_md",
            max_workers=args.max_workers,
            use_doi=False,
        )
        if not output_path:
            raise RuntimeError("PDF 转 Markdown 失败：未找到任何 PDF 文件。")
        input_dir = output_path
    else:
        if not args.input:
            raise ValueError("跳过步骤 0 时必须提供 --input 参数。")
        input_dir = Path(args.input)
        if not input_dir.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    artifact_dir = Path(args.output)
    print(f"\n输入目录: {input_dir}")
    print(f"工件目录: {artifact_dir}")

    tags_file: Path | None = None

    # 步骤 1: 文档标记
    if "1" in steps:
        from rag.tagger import step_tag_documents
        tags_file = step_tag_documents(input_dir, artifact_dir)
    else:
        tags_file = artifact_dir / "document_tags.jsonl"
        if not tags_file.exists():
            print(f"警告: 跳过步骤 1，但未找到标签文件 {tags_file}")

    # 步骤 2: Pinecone 向量化
    if "2" in steps:
        if not tags_file or not tags_file.exists():
            raise FileNotFoundError(
                f"步骤 2 需要标签文件，但未找到: {tags_file}。请先运行步骤 1。"
            )
        from rag.pinecone_ingest import step_ingest_to_pinecone
        step_ingest_to_pinecone(
            input_dir,
            tags_file,
            args.index,
            args.namespace,
            args.start_id,
        )

    # 步骤 3: S3 同步
    if "3" in steps:
        from rag.s3_sync import step_sync_to_s3
        config_path = Path(os.environ.get("CONFIG_PATH", script_dir / "config.json"))
        step_sync_to_s3(input_dir, args.bucket, args.prefix, config_path)

    _section("RAG 管道完成")


# ──────────────────────────────────────────────────────────────────────────────
# 步骤执行器：分子提取管道
# ──────────────────────────────────────────────────────────────────────────────

def run_molecule_pipeline(args, script_dir: Path) -> None:
    """执行分子提取管道的所有指定步骤"""
    _section("分子提取管道")

    from pdf_converter import step_pdf_to_markdown

    steps = _resolve_steps(args.steps)
    print(f"执行步骤: {', '.join(steps)}")

    config_path = Path(os.environ.get("CONFIG_PATH", script_dir / "config.json"))

    md_dir: str | None = None

    # 步骤 0: PDF → Markdown（DOI 命名模式）
    if "0" in steps:
        if not args.pdf_input:
            raise ValueError("步骤 0 需要 --pdf-input 参数。")
        output_path = step_pdf_to_markdown(
            args.pdf_input,
            args.pdf_output or args.input or "./paper_md",
            max_workers=args.max_workers,
            use_doi=True,
        )
        if not output_path:
            raise RuntimeError("PDF 转 Markdown 失败：未找到任何 PDF 文件。")
        md_dir = str(output_path)
    else:
        if not args.input:
            raise ValueError("跳过步骤 0 时必须提供 --input 参数。")
        md_dir = args.input
        if not Path(md_dir).exists():
            raise FileNotFoundError(f"输入目录不存在: {md_dir}")

    print(f"\nMarkdown 目录: {md_dir}")

    # 步骤 1: Markdown → JSONL
    if "1" in steps:
        from molecule.md_to_jsonl import step_md_to_jsonl
        step_md_to_jsonl(md_dir, args.jsonl_dir)

    # 步骤 2: 元数据提取 → Snowflake
    if "2" in steps:
        from molecule.extract_metadata import step_update_snowflake_metadata
        step_update_snowflake_metadata(
            md_root_dir=Path(md_dir),
            temp_meta_path=args.temp_meta_path,
            config_path=config_path,
        )

    # 步骤 3: 分子提取 + 入库
    if "3" in steps:
        from molecule.llm_judge.code.build_graph_paper_patent import main as extract_main
        from molecule.llm_judge.code.add_molecules_to_database import main as add_main
        extract_main(args.jsonl_dir, md_dir, args.temp_meta_path)
        add_main()

    _section("分子提取管道完成")


# ──────────────────────────────────────────────────────────────────────────────
# 步骤解析
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_steps(steps_arg) -> list:
    if "all" in steps_arg:
        return ["0", "1", "2", "3"]
    return list(steps_arg)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统一数据管道：分子提取 + RAG（共享 PDF 转换模块）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行完整 RAG 管道
  python run_pipeline.py --pipeline rag --pdf-input ./pdfs --pdf-output ./paper_md

  # 仅运行 RAG 步骤 1-3（跳过 PDF 转换）
  python run_pipeline.py --pipeline rag --steps 1 2 3 --input ./paper_md

  # 运行完整分子提取管道
  python run_pipeline.py --pipeline molecule --pdf-input ./pdfs --pdf-output ./paper_md_doi

  # 同时运行两个管道
  python run_pipeline.py --pipeline all --pdf-input ./pdfs

环境变量:
  OPENAI_API_KEY    — 文档标记 & DOI/元数据提取（必须）
  PINECONE_API_KEY  — Pinecone 向量化（RAG 步骤 2 必须）
  CONFIG_PATH       — config.json 路径（可选）
        """,
    )

    # 管道选择
    parser.add_argument(
        "--pipeline",
        choices=["rag", "molecule", "all"],
        default="all",
        help="要执行的管道 (默认: all)",
    )

    # 步骤选择
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["0", "1", "2", "3", "all"],
        default=["all"],
        help="要执行的步骤 (默认: all)",
    )

    # PDF 转换参数（步骤 0，两个管道共用）
    parser.add_argument(
        "--pdf-input",
        nargs="+",
        default=None,
        help="PDF 文件或目录路径（步骤 0 使用）",
    )
    parser.add_argument(
        "--pdf-output",
        default=None,
        help="PDF 转 Markdown 的输出目录（步骤 0 使用；不指定则使用 --input）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="PDF 处理并行 worker 数量 (默认: 4)",
    )

    # 通用参数
    parser.add_argument(
        "--input",
        default="./paper_md",
        help="包含 Markdown 文件的目录（步骤 1+ 使用；跳过步骤 0 时必需）",
    )
    parser.add_argument(
        "--output",
        default="./artifacts",
        help="RAG 工件输出目录 (默认: ./artifacts)",
    )

    # RAG 专用参数
    rag_group = parser.add_argument_group("RAG 管道参数")
    rag_group.add_argument("--index", default="rag-improvements-v2", help="Pinecone 索引名称")
    rag_group.add_argument("--namespace", default="test", help="Pinecone 命名空间")
    rag_group.add_argument(
        "--start-id",
        default="chunk-0",
        help="断点续传起始块 ID (例如: chunk-500)",
    )
    rag_group.add_argument("--bucket", default="mol-u-llm", help="S3 存储桶名称")
    rag_group.add_argument("--prefix", default="test", help="S3 键前缀")

    # 分子提取专用参数
    mol_group = parser.add_argument_group("分子提取管道参数")
    mol_group.add_argument(
        "--jsonl-dir",
        default="./paper_jsonl",
        help="JSONL 文件输出目录 (默认: ./paper_jsonl)",
    )
    mol_group.add_argument(
        "--temp-meta-path",
        default="./temp_meta.csv",
        help="临时元数据 CSV 文件路径 (默认: ./temp_meta.csv)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()

    # 将脚本目录加入 sys.path，确保模块导入正常
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    _section("统一数据管道")
    print(f"管道模式: {args.pipeline}")
    print(f"步骤: {args.steps}")
    print(f"脚本目录: {script_dir}")

    pipelines_to_run = []
    if args.pipeline == "all":
        pipelines_to_run = ["rag", "molecule"]
    else:
        pipelines_to_run = [args.pipeline]

    for pipeline in pipelines_to_run:
        if pipeline == "rag":
            run_rag_pipeline(args, script_dir)
        elif pipeline == "molecule":
            run_molecule_pipeline(args, script_dir)

    _section("所有管道执行完毕")


if __name__ == "__main__":
    main()
