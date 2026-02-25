"""
RAG 管道步骤 3：S3 同步

将 Markdown 文件及其关联的图片（_artifacts 目录下的 PNG）
上传到 AWS S3，跳过已存在的文件。
"""

import json
from pathlib import Path
from typing import Tuple

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# AWS 凭证加载
# ──────────────────────────────────────────────────────────────────────────────

def load_aws_config(config_path: Path) -> Tuple[str, str]:
    """
    从 config.json 加载 AWS 凭证。

    参数:
        config_path: config.json 文件路径

    返回:
        (aws_access_key_id, aws_secret_access_key)
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"未找到 config.json: {config_path}。"
            "请在项目目录下创建包含 aws_access_key_id 和 aws_secret_access_key 的配置文件。"
        )
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config["aws_access_key_id"], config["aws_secret_access_key"]
    except KeyError as e:
        raise RuntimeError(f"config.json 中缺少字段: {e}")
    except Exception as e:
        raise RuntimeError(f"无法加载 AWS 配置: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# S3 操作辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def key_exists(s3_client, bucket: str, key: str) -> bool:
    """检查 S3 对象是否存在"""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def upload_file_to_s3(s3_client, local_path: Path, bucket: str, s3_key: str) -> str:
    """上传单个文件到 S3，返回 'uploaded' 或 'error: ...'"""
    try:
        s3_client.upload_file(str(local_path), bucket, s3_key)
        return "uploaded"
    except Exception as e:
        return f"error: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# 步骤入口
# ──────────────────────────────────────────────────────────────────────────────

def step_sync_to_s3(
    input_dir: Path,
    bucket: str,
    prefix: str,
    config_path: Path,
) -> None:
    """
    步骤 3：将 Markdown 和图片同步到 S3

    目录结构要求：
        input_dir/
            paper_name/
                paper_name.md
                paper_name_artifacts/
                    image1.png
                    ...

    参数:
        input_dir: 包含各论文子目录的根目录
        bucket: S3 存储桶名称
        prefix: S3 键前缀（基础文件夹路径）
        config_path: 包含 AWS 凭证的 config.json 路径
    """
    print("\n" + "=" * 60)
    print("步骤 3: S3 多模态同步")
    print("=" * 60)

    ak, sk = load_aws_config(config_path)
    s3 = boto3.client("s3", aws_access_key_id=ak, aws_secret_access_key=sk)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    md_queue = []
    png_queue = []

    print(f"扫描 {input_dir} 查找待上传文件...")
    subfolders = [d for d in input_dir.iterdir() if d.is_dir()]

    for folder in subfolders:
        # Markdown 文件
        md_file = folder / f"{folder.name}.md"
        if md_file.exists():
            s3_md_key = f"{prefix}/{folder.name}/{md_file.name}".replace("\\", "/")
            if not key_exists(s3, bucket, s3_md_key):
                md_queue.append((md_file, s3_md_key))

        # PNG 图片（_artifacts 目录）
        artifact_dir = folder / f"{folder.name}_artifacts"
        if not artifact_dir.is_dir():
            # 兼容嵌套路径
            nested_dirs = list(folder.rglob(f"{folder.name}_artifacts"))
            if nested_dirs:
                artifact_dir = nested_dirs[0]

        if artifact_dir.is_dir():
            for png_file in artifact_dir.rglob("*.png"):
                rel_to_artifacts = png_file.relative_to(artifact_dir)
                s3_png_key = (
                    f"{prefix}/{folder.name}/{folder.name}_artifacts/{rel_to_artifacts}"
                ).replace("\\", "/")
                if not key_exists(s3, bucket, s3_png_key):
                    png_queue.append((png_file, s3_png_key))

    # 批量上传
    for label, queue in [("MD 文件", md_queue), ("PNG 图片", png_queue)]:
        if not queue:
            print(f"  没有新的 {label} 需要上传。")
            continue

        print(f"\n开始上传 {label}（{len(queue)} 个文件）...")
        uploaded, errors = 0, 0

        with tqdm(total=len(queue), desc=f"上传 {label}", unit="file") as pbar:
            for local, remote in queue:
                status = upload_file_to_s3(s3, local, bucket, remote)
                if status == "uploaded":
                    uploaded += 1
                else:
                    errors += 1
                    pbar.write(f"上传失败 {remote}: {status}")
                pbar.update(1)
                pbar.set_postfix(success=uploaded, fail=errors)

    print("\nS3 同步完成。")
