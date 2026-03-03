import os
import resend

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
        resend.Emails.send({
            "from": FROM_EMAIL,
            "to": to_email,
            "subject": f"【{video_title}】の切り抜き動画が完成しました",
            "html": f"""
                <h2>切り抜き動画が完成しました！</h2>
                <p>「{video_title}」の解析が完了し、{clip_count}件のプレーシーンが検出されました。</p>
                <p><a href="{video_url}">こちらから確認する</a></p>
            """,
        })
        return True
    except Exception as e:
        print(f"メール送信失敗: {e}")
        return False
