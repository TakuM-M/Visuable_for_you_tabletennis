import os
import subprocess
import tempfile
from pathlib import Path


def clip_video(input_path: str, clips: list[dict], output_path: str) -> None:
    """
    FFmpeg でシーン区間をカット・結合して output_path に保存する。

    Args:
        input_path: 元動画のパス
        clips: [{"start_time": float, "end_time": float}, ...] のリスト
        output_path: 出力先パス
    """
    if not clips:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        # 各シーンを再エンコードして一時ファイルに切り出す。
        # -ss は -i の前（入力シーク）に置く。-i の後（出力シーク）だと
        # セグメントごとに動画先頭から全デコードするため、長い動画では
        # 書き出しに数十分かかる。再エンコードを伴う場合は入力シークでも
        # フレーム精度は保たれる。
        segment_files: list[str] = []
        for i, clip in enumerate(clips):
            seg = os.path.join(tmpdir, f"seg_{i:04d}.mp4")
            length = clip["end_time"] - clip["start_time"]
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(clip["start_time"]),
                    "-i",
                    input_path,
                    "-t",
                    str(length),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-c:a",
                    "aac",
                    seg,
                ],
                check=True,
                capture_output=True,
            )
            segment_files.append(seg)

        # concat リストファイルを作成
        list_file = os.path.join(tmpdir, "concat.txt")
        with open(list_file, "w") as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")

        # セグメントを結合して出力
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file,
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
