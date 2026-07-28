/**
 * 動画の保存期限（保持ポリシーによる自動削除）の表示ヘルパー。
 *
 * 期限そのものは backend が VideoResponse.expires_at として返す
 * （created_at + video_retention_days）。フロントは日数の丸めと文言だけを担当する。
 */

export type Retention = {
  /** 「あと3日」「本日中に削除」などの見出し文言 */
  label: string;
  /** 期限の日付（YYYY-MM-DD） */
  date: string;
  /** 残り1日以下。UI 側で警告色に切り替える目印 */
  urgent: boolean;
  /** 期限を過ぎている（削除バッチ待ちの状態） */
  expired: boolean;
};

const DAY_MS = 24 * 60 * 60 * 1000;

export function formatRetention(expiresAt: string, now: Date = new Date()): Retention {
  const expires = new Date(expiresAt);
  const remainingMs = expires.getTime() - now.getTime();
  // 切り上げ。残り 0.5 日なら「あと1日」と表示して、実際より短く見せない
  const remainingDays = Math.ceil(remainingMs / DAY_MS);
  const date = expires
    .toLocaleDateString("ja-JP", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    })
    .replaceAll("/", "-");

  if (remainingMs <= 0) {
    return { label: "まもなく削除", date, urgent: true, expired: true };
  }
  if (remainingDays <= 1) {
    return { label: "本日中に削除", date, urgent: true, expired: false };
  }
  return {
    label: `あと${remainingDays}日`,
    date,
    urgent: remainingDays <= 2,
    expired: false,
  };
}
