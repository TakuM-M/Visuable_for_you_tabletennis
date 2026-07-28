import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _run_ffmpeg(cmd: list[str], timeout: int, label: str) -> None:
    """FFmpeg を起動する。失敗・タイムアウトは理由付きの RuntimeError にして送出する。

    capture_output のまま CalledProcessError を投げると stderr がどこにも出ず、
    失敗原因（コーデック非対応・ディスク不足・壊れた入力）を追えなくなる。
    timeout が無いと、進まなくなった FFmpeg を書き出しスロットごと無期限に
    抱え込んでしまう。
    """
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"{label}が {timeout} 秒を超えたため中断しました") from e
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{label}に失敗しました: {stderr[-1000:]}") from e


def _segment_workers(segment_count: int) -> int:
    """セグメント切り出しの並列数。

    プレー区間は 1 本あたり数秒と短く、1 プロセスでは FFmpeg のスレッド並列が
    効ききらないまま起動・シークのオーバーヘッドが支配的になる。複数本を重ねて
    詰めると CPU を使い切れる。ただし CPU コア数とセグメント数は超えない
    （超えても取り合いになるだけ）。
    """
    return max(1, min(settings.ffmpeg_segment_workers, os.cpu_count() or 1, segment_count))


def _extract_segment(input_path: str, clip: dict, seg_path: str) -> None:
    """1 区間を再エンコードして切り出す。

    -ss は -i の前（入力シーク）に置く。-i の後（出力シーク）だと
    セグメントごとに動画先頭から全デコードするため、長い動画では
    書き出しに数十分かかる。再エンコードを伴う場合は入力シークでも
    フレーム精度は保たれる。
    """
    length = clip["end_time"] - clip["start_time"]
    _run_ffmpeg(
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
            seg_path,
        ],
        timeout=settings.ffmpeg_segment_timeout_seconds,
        label="区間の切り出し",
    )


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

    started = time.monotonic()
    with tempfile.TemporaryDirectory() as tmpdir:
        segment_files = [
            os.path.join(tmpdir, f"seg_{i:04d}.mp4") for i in range(len(clips))
        ]
        workers = _segment_workers(len(clips))
        logger.info(
            "区間の切り出し開始 segments=%s 並列=%s", len(segment_files), workers
        )

        # 例外は future の結果取得時に送出される（map は遅延評価ではなく
        # イテレート時に再送出するので、list() で全件を確定させて拾う）
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(
                pool.map(
                    lambda args: _extract_segment(input_path, *args),
                    zip(clips, segment_files),
                )
            )

        # concat リストファイルを作成
        list_file = os.path.join(tmpdir, "concat.txt")
        with open(list_file, "w") as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")

        # セグメントを結合して出力
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        _run_ffmpeg(
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
            timeout=settings.ffmpeg_concat_timeout_seconds,
            label="区間の結合",
        )

    logger.info(
        "区間の切り出し・結合完了 segments=%s 所要=%.1f秒",
        len(clips),
        time.monotonic() - started,
    )
