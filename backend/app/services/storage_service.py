import os
from urllib.parse import quote

import boto3
from botocore.config import Config

R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "")

_client = None


def _get_client():
    """S3互換クライアントを遅延初期化して返す"""
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT_URL,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
        )
    return _client


def upload_file(local_path: str, r2_key: str) -> None:
    """ローカルファイルをR2にアップロードする"""
    _get_client().upload_file(local_path, R2_BUCKET_NAME, r2_key)


def generate_presigned_url(
    r2_key: str, expires_in: int = 3600, download_filename: str | None = None
) -> str:
    """期限付きダウンロードURLを生成する

    download_filename を指定すると Content-Disposition: attachment 付きの URL になり、
    ブラウザはインライン再生ではなくファイル保存として扱う（モバイルでの
    ダウンロードに必須）。日本語ファイル名は RFC 5987 (filename*) で渡す。
    """
    params: dict[str, str] = {"Bucket": R2_BUCKET_NAME, "Key": r2_key}
    if download_filename is not None:
        params["ResponseContentDisposition"] = (
            "attachment; filename=\"video.mp4\"; "
            f"filename*=UTF-8''{quote(download_filename)}"
        )
    return _get_client().generate_presigned_url(
        "get_object",
        Params=params,
        ExpiresIn=expires_in,
    )


def delete_file(r2_key: str) -> None:
    """R2からファイルを削除する"""
    _get_client().delete_object(Bucket=R2_BUCKET_NAME, Key=r2_key)
