"""RunPod Serverless API のクライアント。

ジョブの投入（/run）は歴史的経緯で video_service.call_ml_service にあるが、
状態問い合わせ（/status）と停止（/cancel）はここに置く。

RunPod のジョブ状態:
    IN_QUEUE    ワーカー待ち
    IN_PROGRESS ワーカーが処理中
    RUNNING     同上（API バージョンによりこちらが返る）
    COMPLETED   正常終了
    FAILED      ハンドラがエラーで終了
    CANCELLED   /cancel で停止された
    TIMED_OUT   実行時間超過、またはキューで拾われる前に TTL 切れ
"""

import os

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")

# 「まだ生きている」とみなす状態。これ以外は終了済みとして扱う
ACTIVE_STATUSES = {"IN_QUEUE", "IN_PROGRESS", "RUNNING"}
# GPU 側で異常終了した状態
DEAD_STATUSES = {"FAILED", "CANCELLED", "TIMED_OUT"}


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {RUNPOD_API_KEY}"}


def get_job_status(runpod_job_id: str) -> str | None:
    """RunPod のジョブ状態を取得する。取得できなければ None を返す。

    None は「状態が不明」であって「死んでいる」ではない。呼び出し側は
    None のときに失敗扱いにしてはならない（一時的な通信断で生きている
    ジョブを殺してしまうため）。
    """
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(
                f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{runpod_job_id}",
                headers=_headers(),
            )
            response.raise_for_status()
            return response.json().get("status")
    except Exception as e:
        logger.warning(
            "RunPod ジョブ状態の取得に失敗 runpod_job_id=%s: %s", runpod_job_id, e
        )
        return None


def cancel_job(runpod_job_id: str) -> bool:
    """RunPod のジョブを停止して GPU の課金を止める。成功したら True。

    既に終了しているジョブへの cancel は無害なので、迷ったら呼んでよい。
    失敗しても例外は投げない（キャンセルできないことを理由に失敗処理そのものを
    止めてはいけないため）。
    """
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(
                f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/cancel/{runpod_job_id}",
                headers=_headers(),
            )
            response.raise_for_status()
        logger.info("RunPod ジョブを停止 runpod_job_id=%s", runpod_job_id)
        return True
    except Exception as e:
        logger.warning(
            "RunPod ジョブの停止に失敗 runpod_job_id=%s: %s", runpod_job_id, e
        )
        return False
