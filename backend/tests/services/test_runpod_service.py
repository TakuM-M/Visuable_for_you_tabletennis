"""runpod_service の mock テスト。

httpx を差し替えて状態取得・停止の挙動を検証する。この2関数は
「呼び出しが失敗しても例外を外に出さない」ことが呼び出し側との約束なので、
戻り値と併せてそこを重点的に見る。
"""

from unittest.mock import patch

from app.services import runpod_service


# --- get_job_status ---------------------------------------------------------


def test_get_job_status_returns_status_string() -> None:
    with (
        patch("app.services.runpod_service.RUNPOD_ENDPOINT_ID", "ep"),
        patch("app.services.runpod_service.httpx") as mock_httpx,
    ):
        client = mock_httpx.Client.return_value.__enter__.return_value
        client.get.return_value.json.return_value = {"status": "IN_PROGRESS"}
        result = runpod_service.get_job_status("rp-1")

    assert result == "IN_PROGRESS"
    assert "/status/rp-1" in client.get.call_args.args[0]


def test_get_job_status_returns_none_on_request_error() -> None:
    """問い合わせ失敗は None。

    None は「状態不明」であって「死んでいる」ではない。呼び出し側（reconcile）が
    一時的な通信断で稼働中のジョブを殺さないための約束。
    """
    with (
        patch("app.services.runpod_service.RUNPOD_ENDPOINT_ID", "ep"),
        patch("app.services.runpod_service.httpx") as mock_httpx,
    ):
        client = mock_httpx.Client.return_value.__enter__.return_value
        client.get.side_effect = RuntimeError("network down")
        result = runpod_service.get_job_status("rp-1")

    assert result is None


def test_get_job_status_returns_none_when_field_absent() -> None:
    """レスポンスに status が無い場合も None（未知の形式で誤判定しない）"""
    with (
        patch("app.services.runpod_service.RUNPOD_ENDPOINT_ID", "ep"),
        patch("app.services.runpod_service.httpx") as mock_httpx,
    ):
        client = mock_httpx.Client.return_value.__enter__.return_value
        client.get.return_value.json.return_value = {}
        result = runpod_service.get_job_status("rp-1")

    assert result is None


# --- cancel_job -------------------------------------------------------------


def test_cancel_job_posts_to_cancel_endpoint() -> None:
    with (
        patch("app.services.runpod_service.RUNPOD_ENDPOINT_ID", "ep"),
        patch("app.services.runpod_service.httpx") as mock_httpx,
    ):
        client = mock_httpx.Client.return_value.__enter__.return_value
        result = runpod_service.cancel_job("rp-1")

    assert result is True
    assert "/cancel/rp-1" in client.post.call_args.args[0]


def test_cancel_job_returns_false_on_error() -> None:
    """停止に失敗しても例外を投げない。

    キャンセルできないことを理由に失敗処理そのものを止めてはいけないため。
    """
    with (
        patch("app.services.runpod_service.RUNPOD_ENDPOINT_ID", "ep"),
        patch("app.services.runpod_service.httpx") as mock_httpx,
    ):
        client = mock_httpx.Client.return_value.__enter__.return_value
        client.post.side_effect = RuntimeError("network down")
        result = runpod_service.cancel_job("rp-1")

    assert result is False
