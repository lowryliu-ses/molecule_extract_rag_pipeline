"""
分子提取管道步骤 2：元数据提取与 Snowflake 更新

使用 GPT 从 Markdown 文件中提取论文元数据（DOI、标题、作者、期刊、年份），
并将结果写入本地 CSV 临时文件后上传至 Snowflake。

来源：molecule_extraction/Codes/extract_metadata.py
修复：移除模块顶层的硬编码初始化，改为按需从 config.json 加载凭证。
"""

import io
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import boto3
import pandas as pd
import requests
from botocore.exceptions import ClientError
from dateutil.parser import parse as dtparse
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from requests.exceptions import HTTPError, RequestException, Timeout
from tqdm.auto import tqdm

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────────────────

MODEL = os.getenv("GPT_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

TIMEOUT_S = 6000
COLS = ["doi", "title", "authors", "journal", "year", "source"]


# ──────────────────────────────────────────────────────────────────────────────
# 凭证加载（按需，避免模块导入时报错）
# ──────────────────────────────────────────────────────────────────────────────

def _load_config(config_path: Optional[Path] = None) -> Dict:
    """
    加载 config.json。优先使用传入路径，否则按以下顺序查找：
    1. 环境变量 CONFIG_PATH
    2. 当前工作目录下的 config.json
    3. 脚本目录下的 config.json
    """
    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    if os.environ.get("CONFIG_PATH"):
        candidates.append(Path(os.environ["CONFIG_PATH"]))
    candidates.append(Path.cwd() / "config.json")
    candidates.append(Path(__file__).parent.parent / "config.json")

    for p in candidates:
        if p.exists():
            with open(p, "r") as f:
                return json.load(f)

    raise FileNotFoundError(
        "未找到 config.json。请确保文件存在于项目根目录，"
        "或通过 CONFIG_PATH 环境变量指定路径。"
    )


def _get_s3_client(config_path: Optional[Path] = None):
    """按需创建 S3 客户端"""
    config = _load_config(config_path)
    return boto3.client(
        "s3",
        aws_access_key_id=config["aws_access_key_id"],
        aws_secret_access_key=config["aws_secret_access_key"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def build_doi(folder_name: str) -> str:
    """将文件夹名（下划线）转换为 DOI（斜杠）"""
    return folder_name.replace("_", "/").strip()


def normalize_authors_list(authors: List[str]) -> str:
    """将作者列表规范化为分号分隔字符串"""
    normed = []
    for a in authors:
        if not isinstance(a, str):
            continue
        s = " ".join(a.strip().split())
        if s:
            normed.append(s)
    return "; ".join(normed)


def coerce_year(value) -> Optional[int]:
    """将各种年份格式转换为整数 YYYY"""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if not s:
        return None
    if s.isdigit() and len(s) == 4:
        return int(s)
    try:
        return dtparse(s, fuzzy=True).year
    except Exception:
        return None


def messages_for_md(md_text: str) -> List[Dict]:
    """构建元数据提取的 chat 消息列表"""
    system = (
        "You are a precise information extractor. "
        "Given a markdown article (from an academic paper's metadata + abstract), "
        "return a STRICT JSON object with keys exactly: "
        'DOI (string), title (string), authors (array of strings, each "First Last"), '
        "journal (string), year (integer YYYY). "
        "If a field is missing, infer conservatively from the text; avoid hallucinations."
    )
    user = (
        "Extract these fields from the markdown below:\n\n"
        "Return ONLY JSON, no commentary.\n\n"
        f"{md_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=2, max=20),
    retry=retry_if_exception_type((HTTPError, Timeout, RequestException)),
)
def call_gpt_extract(md_text: str) -> Dict:
    """
    调用 OpenAI Chat Completions 提取论文元数据。
    返回包含 DOI、title、authors、journal、year 的字典。
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 未设置。")
    url = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "response_format": {"type": "json_object"},
        "messages": messages_for_md(md_text[:80000]),
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_S)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except requests.exceptions.HTTPError as e:
        print(f"[错误] HTTP {resp.status_code}: {resp.text}")
        raise
    except json.JSONDecodeError:
        raise


def read_md_for_folder(folder: Path) -> Optional[str]:
    """读取文件夹内 {folder.name}.md 的内容"""
    md_path = folder / f"{folder.name}.md"
    if md_path.exists() and md_path.is_file():
        try:
            return md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return md_path.read_text(encoding="latin-1", errors="ignore")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# S3 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def s3_object_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = (
            e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            or e.response.get("Error", {}).get("Code")
        )
        if code in (404, "404", "NoSuchKey", "NotFound"):
            return False
        raise


def read_excel_from_s3(s3_client, bucket: str, key: str) -> pd.DataFrame:
    """从 S3 读取 .xlsx 并返回 DataFrame（不存在则返回空 DataFrame）"""
    if not s3_object_exists(s3_client, bucket, key):
        return pd.DataFrame(columns=COLS)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    return pd.read_excel(
        io.BytesIO(data),
        dtype={"doi": str, "title": str, "authors": str, "journal": str, "year": object, "source": str},
    )


def write_excel_to_s3(df: pd.DataFrame, s3_client, bucket: str, key: str) -> None:
    """将 DataFrame 写入内存 .xlsx 并上传到 S3"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.getvalue(),
        ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Snowflake 操作
# ──────────────────────────────────────────────────────────────────────────────

def insert_to_snowflake(rows: List[Dict], config_path: Optional[Path] = None) -> None:
    """将元数据记录 upsert 到 Snowflake RAG_METADATA 表"""
    if not rows:
        print("[INFO] 没有需要插入 Snowflake 的记录。")
        return

    from snowflake.connector import connect

    config = _load_config(config_path)

    conn = connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT", config.get("SNOWFLAKE_ACCOUNT", "SESAI-MAIN")),
        user=os.getenv("SNOWFLAKE_USER", config.get("SNOWFLAKE_USER", "")),
        private_key_file=os.getenv(
            "SNOWFLAKE_PRIVATE_KEY_FILE",
            config.get("SNOWFLAKE_PRIVATE_KEY_FILE", ""),
        ),
        role=os.getenv("SNOWFLAKE_ROLE", config.get("SNOWFLAKE_ROLE", "MOLECULAR_UNIVERSE_ROLE")),
        warehouse=os.getenv(
            "SNOWFLAKE_WAREHOUSE",
            config.get("SNOWFLAKE_WAREHOUSE", "MOLECULAR_UNIVERSE_WH"),
        ),
        database=os.getenv("SNOWFLAKE_DATABASE", config.get("SNOWFLAKE_DATABASE", "UMAP_DATA")),
        schema=os.getenv("SNOWFLAKE_SCHEMA", config.get("SNOWFLAKE_SCHEMA", "PUBLIC")),
        client_session_keep_alive=True,
    )

    cursor = conn.cursor()
    inserted = 0
    try:
        for row in rows:
            cursor.execute(
                """
                MERGE INTO RAG_METADATA AS target
                USING (SELECT %(MARKDOWN_FILE_PATH)s AS MARKDOWN_FILE_PATH) AS source
                ON target.MARKDOWN_FILE_PATH = source.MARKDOWN_FILE_PATH
                WHEN MATCHED THEN UPDATE SET
                  CATEGORY = %(CATEGORY)s,
                  DOI = %(DOI)s,
                  TITLE = %(TITLE)s,
                  AUTHORS = %(AUTHORS)s,
                  JOURNAL = %(JOURNAL)s,
                  YEAR = %(YEAR)s,
                  META_INFO_SOURCE = %(META_INFO_SOURCE)s
                WHEN NOT MATCHED THEN INSERT (
                  MARKDOWN_FILE_PATH, CATEGORY, DOI, TITLE, AUTHORS, JOURNAL, YEAR, META_INFO_SOURCE
                ) VALUES (
                  %(MARKDOWN_FILE_PATH)s, %(CATEGORY)s, %(DOI)s, %(TITLE)s,
                  %(AUTHORS)s, %(JOURNAL)s, %(YEAR)s, %(META_INFO_SOURCE)s
                )
                """,
                row,
            )
            inserted += 1
    finally:
        cursor.close()
        conn.close()

    print(f"成功插入或更新 {inserted} 条记录到 Snowflake。")


# ──────────────────────────────────────────────────────────────────────────────
# 步骤入口
# ──────────────────────────────────────────────────────────────────────────────

def step_update_snowflake_metadata(
    md_root_dir: Path,
    temp_meta_path: str,
    source_label: str = "llm",
    category: str = "",
    config_path: Optional[Path] = None,
) -> None:
    """
    步骤 2（分子管道）：从 Markdown 提取元数据并写入 Snowflake

    参数:
        md_root_dir: 包含各论文子目录（每个子目录含同名 .md 文件）的根目录
        temp_meta_path: 本地临时 CSV 文件路径（保存提取结果备份）
        source_label: 元数据来源标签（默认 "llm"）
        category: 论文分类标签（可选）
        config_path: config.json 路径（None 则自动查找）
    """
    print("\n" + "=" * 60)
    print("步骤 2 (分子): 元数据提取 → Snowflake")
    print("=" * 60)

    if not md_root_dir.exists():
        raise FileNotFoundError(f"Markdown 根目录不存在: {md_root_dir}")

    candidate_folders = [
        sub for sub in md_root_dir.iterdir()
        if sub.is_dir() and (sub / f"{sub.name}.md").exists()
    ]

    if not candidate_folders:
        print(f"[INFO] 未在 {md_root_dir} 找到含 .md 文件的子目录")
        return

    records = []
    with tqdm(total=len(candidate_folders), desc="提取元数据") as pbar:
        for sub in candidate_folders:
            md_path = sub / f"{sub.name}.md"
            md_text = read_md_for_folder(sub)

            if not md_text:
                pbar.write(f"[跳过] {sub.name} 无 MD 内容")
                pbar.update(1)
                continue
            try:
                extracted = call_gpt_extract(md_text)

                raw_authors = extracted.get("authors") or []
                if isinstance(raw_authors, str):
                    raw_authors = [
                        a.strip()
                        for a in raw_authors.replace(";", ",").split(",")
                        if a.strip()
                    ]
                elif not isinstance(raw_authors, list):
                    raw_authors = []

                authors_str = normalize_authors_list(raw_authors)
                year = coerce_year(extracted.get("year"))

                record = {
                    "MARKDOWN_FILE_PATH": str(md_path),
                    "CATEGORY": category,
                    "DOI": extracted.get("doi", "") or extracted.get("DOI", "") or build_doi(sub.name),
                    "TITLE": extracted.get("title", ""),
                    "AUTHORS": authors_str,
                    "JOURNAL": extracted.get("journal", ""),
                    "YEAR": year if year is not None else None,
                    "META_INFO_SOURCE": json.dumps({
                        k: source_label
                        for k in ("title", "authors", "journal", "year", "doi")
                    }),
                }
                records.append(record)
                pbar.set_postfix_str(f"ok:{sub.name}")
            except requests.exceptions.HTTPError as e:
                print(f"[HTTPError] {e.response.status_code} | {e.response.text}")
                raise
            except Exception as e:
                pbar.write(f"[错误] {sub.name}: {e}")
            finally:
                pbar.update(1)

    print(f"元数据已保存到本地临时文件: {temp_meta_path}")
    pd.DataFrame(records).to_csv(temp_meta_path, index=False)

    print("开始 upsert 到 Snowflake...")
    insert_to_snowflake(records, config_path)
