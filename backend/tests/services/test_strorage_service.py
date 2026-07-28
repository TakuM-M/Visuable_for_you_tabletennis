"""R2 ストレージサービスのテスト。

storage_service は boto3 (S3互換クライアント) の薄いラッパ。
本物の R2 には繋がず、_get_client が返す client を Mock に差し替えて
「正しいメソッドを正しい引数で呼んでいるか」を検証する。
"""

from unittest.mock import patch, Mock

from app.services import storage_service


def test_upload_file_calls_client_upload_file():
    """upload_file は client.upload_file(local_path, bucket, key) を呼ぶ"""
    client = Mock()
    with patch("app.services.storage_service._get_client", return_value=client):
        storage_service.upload_file("/tmp/a.mp4", "videos/a.mp4")
    client.upload_file.assert_called_once_with(
        "/tmp/a.mp4", storage_service.R2_BUCKET_NAME, "videos/a.mp4", ExtraArgs=None
    )


def test_upload_file_sets_content_type_when_given():
    """content_type 指定時は ExtraArgs でオブジェクトのメタデータに反映する

    サムネイルは <img> から直接読ませるため、octet-stream のままだと
    ブラウザによっては表示されない。
    """
    client = Mock()
    with patch("app.services.storage_service._get_client", return_value=client):
        storage_service.upload_file(
            "/tmp/a.jpg", "thumbnails/a.jpg", content_type="image/jpeg"
        )
    client.upload_file.assert_called_once_with(
        "/tmp/a.jpg",
        storage_service.R2_BUCKET_NAME,
        "thumbnails/a.jpg",
        ExtraArgs={"ContentType": "image/jpeg"},
    )


def test_generate_presigned_url_passes_params():
    """generate_presigned_url は client.generate_presigned_url を正しい引数で呼ぶ"""
    client = Mock()
    client.generate_presigned_url.return_value = "http://signed"
    with patch("app.services.storage_service._get_client", return_value=client):
        url = storage_service.generate_presigned_url("videos/a.mp4")
    assert url == "http://signed"
    client.generate_presigned_url.assert_called_once_with(
        "get_object",
        Params={"Bucket": storage_service.R2_BUCKET_NAME, "Key": "videos/a.mp4"},
        ExpiresIn=3600,
    )


def test_generate_presigned_url_with_download_filename():
    """download_filename 指定時は Content-Disposition: attachment が付き、
    日本語ファイル名は RFC 5987 (filename*) でパーセントエンコードされる"""
    client = Mock()
    client.generate_presigned_url.return_value = "http://signed"
    with patch("app.services.storage_service._get_client", return_value=client):
        storage_service.generate_presigned_url(
            "outputs/a.mp4", download_filename="試合 vs 田中.mp4"
        )
    params = client.generate_presigned_url.call_args.kwargs["Params"]
    disposition = params["ResponseContentDisposition"]
    assert disposition.startswith('attachment; filename="video.mp4"; ')
    assert (
        "filename*=UTF-8''%E8%A9%A6%E5%90%88%20vs%20%E7%94%B0%E4%B8%AD.mp4"
        in disposition
    )


def test_delete_file_calls_delete_object():
    client = Mock()
    with patch("app.services.storage_service._get_client", return_value=client):
        storage_service.delete_file("videos/a.mp4")
    client.delete_object.assert_called_once_with(
        Bucket=storage_service.R2_BUCKET_NAME, Key="videos/a.mp4"
    )


def test_get_client_is_cached():
    """_get_client は最初の1回だけ boto3.client を呼び、同じインスタンスを返す"""
    storage_service._client = None  # キャッシュをリセット
    with patch("boto3.client") as mock_boto_client:
        mock_boto_client.return_value = Mock()
        client1 = storage_service._get_client()
        client2 = storage_service._get_client()
    mock_boto_client.assert_called_once()  # boto3.client は1回だけ呼ばれる
    assert client1 is client2  # 同じインスタンスが返る


# test_generate_presigned_url_passes_params:
#   client.generate_pr 値（例"http://signed"）を
#   return_value で固
#   検証ポイント:
#     - storage_servicの戻り値がそれと一致するか
#     - "get_object" /...} / ExpiresInが正しいか
#       → assert_calle, ExpiresIn=3600)
#   余裕があれば expires_in を省略した時のデフォルト 3600 も別ケースで。

# test_delete_file_calls_delete_object:
#   client.delete_obje回呼ばれることを検証。

# test_get_client_is_c扱う):
#   storage_service._client = None でリセット → boto3.client を patch →
#   _get_client() を2 だけ」呼ばれ、
#   2回とも同じインスタンスが返ることを確認（assert a is b）。
