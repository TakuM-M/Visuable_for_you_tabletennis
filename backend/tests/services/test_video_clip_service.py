"""video_clip_service の mock テスト。

FFmpeg そのものは起動せず、subprocess.run を差し替えてコマンドの組み立てを
検証する。特に -ss を -i の前（入力シーク）に置くことは書き出し速度に直結する
ため、退行しないようコマンドの並び順を固定する。
"""

import subprocess
from unittest.mock import patch

import pytest

from app.services.video_clip_service import _segment_workers, clip_video


def _split_calls(run):
    """subprocess.run の呼び出しを（セグメント切り出し, 結合）に分ける"""
    cmds = [call.args[0] for call in run.call_args_list]
    return [c for c in cmds if "concat" not in c], [c for c in cmds if "concat" in c]


def test_clip_video_uses_input_seek(tmp_path) -> None:
    """各セグメントは -ss（入力シーク）→ -i → -t（区間長）の順で切り出す"""
    clips = [
        {"start_time": 100.0, "end_time": 130.0},
        {"start_time": 300.0, "end_time": 310.5},
    ]
    out = tmp_path / "out" / "play_scenes.mp4"
    with patch("app.services.video_clip_service.subprocess.run") as run:
        clip_video("input.mp4", clips, str(out))

    # セグメント切り出し 2 回 + concat 結合 1 回
    assert run.call_count == 3
    seg_cmds, concat_cmds = _split_calls(run)
    assert len(seg_cmds) == 2

    # 切り出しは並列に走るため呼び出し順は保証されない。開始位置で対応付ける
    by_start = {cmd[cmd.index("-ss") + 1]: cmd for cmd in seg_cmds}
    for clip in clips:
        cmd = by_start[str(clip["start_time"])]
        # 出力シーク（-i の後）に戻すと毎回先頭からデコードして極端に遅くなる
        assert cmd.index("-ss") < cmd.index("-i")
        assert cmd[cmd.index("-t") + 1] == str(clip["end_time"] - clip["start_time"])

    assert concat_cmds[0][-1] == str(out)


def test_clip_video_noop_without_clips() -> None:
    """clips が空なら FFmpeg を一切起動しない"""
    with patch("app.services.video_clip_service.subprocess.run") as run:
        clip_video("input.mp4", [], "out.mp4")

    run.assert_not_called()


def test_clip_video_passes_timeout_to_ffmpeg(tmp_path) -> None:
    """タイムアウト無しだと、進まなくなった FFmpeg が書き出しスロットを占有し続ける"""
    clips = [{"start_time": 0.0, "end_time": 5.0}]
    with (
        patch("app.services.video_clip_service.subprocess.run") as run,
        patch("app.services.video_clip_service.settings.ffmpeg_segment_timeout_seconds", 111),
        patch("app.services.video_clip_service.settings.ffmpeg_concat_timeout_seconds", 222),
    ):
        clip_video("input.mp4", clips, str(tmp_path / "out.mp4"))

    timeouts = [call.kwargs["timeout"] for call in run.call_args_list]
    assert timeouts == [111, 222]


def test_clip_video_reports_stderr_on_failure(tmp_path) -> None:
    """FFmpeg の stderr を握り潰すと失敗原因（壊れた入力・ディスク不足）が追えない"""
    clips = [{"start_time": 0.0, "end_time": 5.0}]
    with patch(
        "app.services.video_clip_service.subprocess.run",
        side_effect=subprocess.CalledProcessError(
            returncode=1, cmd="ffmpeg", stderr=b"No space left on device"
        ),
    ):
        with pytest.raises(RuntimeError) as exc:
            clip_video("input.mp4", clips, str(tmp_path / "out.mp4"))

    assert "No space left on device" in str(exc.value)


def test_clip_video_reports_timeout_as_runtime_error(tmp_path) -> None:
    clips = [{"start_time": 0.0, "end_time": 5.0}]
    with patch(
        "app.services.video_clip_service.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1800),
    ):
        with pytest.raises(RuntimeError) as exc:
            clip_video("input.mp4", clips, str(tmp_path / "out.mp4"))

    assert "中断" in str(exc.value)


def test_segment_workers_capped_by_segment_count() -> None:
    """セグメント数より多くのワーカーを立てても取り合いになるだけ"""
    with (
        patch("app.services.video_clip_service.settings.ffmpeg_segment_workers", 4),
        patch("app.services.video_clip_service.os.cpu_count", return_value=8),
    ):
        assert _segment_workers(1) == 1
        assert _segment_workers(10) == 4


def test_segment_workers_capped_by_cpu_count() -> None:
    with (
        patch("app.services.video_clip_service.settings.ffmpeg_segment_workers", 4),
        patch("app.services.video_clip_service.os.cpu_count", return_value=2),
    ):
        assert _segment_workers(10) == 2
