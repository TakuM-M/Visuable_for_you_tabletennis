/**
 * 切り抜き区間（clip）をサーバへ送る前の正規化。
 *
 * サーバは受け取った配列の並び順をそのまま sort_order として採番し、end_time が
 * 元動画長を超える区間は 422 で弾く（backend の video_service.replace_clips）。
 * シーンの表示順・連結順を時系列に保ちつつ 422 を避ける整形をここに集約する。
 */

export type ClipRange = {
  start_time: number;
  end_time: number;
};

const round2 = (n: number) => Math.round(n * 100) / 100;

// 丸め → 元動画長でのクランプ → start_time 昇順ソート。
export function normalizeClips(
  clips: ClipRange[],
  sourceDuration: number | null,
): ClipRange[] {
  return clips
    .map((clip) => {
      const endTime = round2(clip.end_time);
      return {
        start_time: round2(clip.start_time),
        end_time:
          sourceDuration !== null && endTime > sourceDuration
            ? sourceDuration
            : endTime,
      };
    })
    .sort((a, b) => a.start_time - b.start_time);
}
