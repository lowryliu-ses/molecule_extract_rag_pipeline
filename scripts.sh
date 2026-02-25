  # 仅转换 PDF → Markdown（原始文件名）
  python run_pipeline.py --pipeline pdf --pdf-input ./papers/pdfs --pdf-output ./papers/paper_md

  # 仅转换 PDF → Markdown（DOI 命名）
  python run_pipeline.py --pipeline pdf --pdf-input ./papers/pdfs --pdf-output ./papers/paper_md_doi --use-doi

  # 运行完整 RAG 管道
  python run_pipeline.py --pipeline rag --pdf-input ./papers/pdfs --pdf-output ./papers/paper_md

  # 仅运行 RAG 步骤 1-3（跳过 PDF 转换）
  python run_pipeline.py --pipeline rag --steps 1 2 3 --input ./papers/paper_md

  # 运行完整分子提取管道
  python run_pipeline.py --pipeline molecule --pdf-input ./papers/pdfs --pdf-output-mol ./papers/paper_md_doi

  # 同时运行两个管道（PDF 转换各执行一次，不重复）
  python run_pipeline.py --pipeline all --pdf-input ./papers/pdfs --pdf-output ./papers/paper_md --pdf-output-mol ./papers/paper_md_doi
