import os
import resend

from app.core.logging import get_logger

logger = get_logger(__name__)

resend.api_key = os.getenv("RESEND_API_KEY", "")
FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev")


def send_clip_completion_email(
    to_email: str,
    video_title: str,
    clip_count: int,
    video_url: str,
) -> bool:
    """切り抜き完了メールを送信。成功したらTrue、失敗したらFalseを返す"""
    try:
        resend.Emails.send(
            {
                "from": FROM_EMAIL,
                "to": to_email,
                "subject": f"【{video_title}】の切り抜き動画が完成しました",
                "html": f"""
                <h2>切り抜き動画が完成しました！</h2>
                <p>「{video_title}」の解析が完了し、{clip_count}件のプレーシーンが検出されました。</p>
                <p><a href="{video_url}">こちらから確認する</a></p>
            """,
            }
        )
        return True
    except Exception as e:
        logger.exception("メール送信失敗: %s", e)
        return False


def send_analysis_complete_email(
    to_email: str,
    video_title: str,
    clip_count: int,
    video_url: str,
) -> bool:
    """ML 解析完了（切り抜きが編集可能になった）ことを通知する。

    出力動画はユーザーが編集画面で書き出し操作をしたときに生成されるため、
    このメールは「編集できる状態になった」ことを知らせる役割を持つ。
    """
    try:
        resend.Emails.send(
            {
                "from": FROM_EMAIL,
                "to": to_email,
                "subject": f"【{video_title}】の解析が完了しました（編集できます）",
                "html": f"""
                <h2>解析が完了しました！</h2>
                <p>「{video_title}」を解析し、{clip_count}件のプレーシーンを検出しました。</p>
                <p>切り抜きの区間を編集してから動画を書き出すことができます。</p>
                <p><a href="{video_url}">編集画面を開く</a></p>
            """,
            }
        )
        return True
    except Exception as e:
        logger.exception("解析完了通知メール送信失敗: %s", e)
        return False


def send_clip_failure_email(
    to_email: str,
    video_title: str,
    video_url: str,
    error_message: str,
) -> bool:
    """切り抜き失敗（自動リトライ枠を使い切った最終失敗時）にメールを送信する"""
    try:
        resend.Emails.send(
            {
                "from": FROM_EMAIL,
                "to": to_email,
                "subject": f"【{video_title}】の解析に失敗しました",
                "html": f"""
                <h2>解析に失敗しました</h2>
                <p>「{video_title}」の解析を試みましたが、自動リトライを含めて失敗しました。</p>
                <p>動画の詳細画面から「再実行」ボタンで再度お試しいただけます。</p>
                <p><a href="{video_url}">動画を確認する</a></p>
                <p style="color:#888;font-size:12px;">エラー: {error_message}</p>
            """,
            }
        )
        return True
    except Exception as e:
        logger.exception("失敗通知メール送信失敗: %s", e)
        return False
