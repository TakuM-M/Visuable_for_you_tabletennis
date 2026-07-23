"""video_clip_service の mock テスト。

FFmpeg そのものは起動せず、subprocess.run を差し替えてコマンドの組み立てを
検証する。特に -ss を -i の前（入力シーク）に置くことは書き出し速度に直結する
ため、退行しないようコマンドの並び順を固定する。
"""

from unittest.mock import patch

from app.services.video_clip_service import clip_video


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
    for call, clip in zip(run.call_args_list[:2], clips):
        cmd = call.args[0]
        # 出力シーク（-i の後）に戻すと毎回先頭からデコードして極端に遅くなる
        assert cmd.index("-ss") < cmd.index("-i")
        assert cmd[cmd.index("-ss") + 1] == str(clip["start_time"])
        assert cmd[cmd.index("-t") + 1] == str(clip["end_time"] - clip["start_time"])

    concat_cmd = run.call_args_list[2].args[0]
    assert "concat" in concat_cmd
    assert concat_cmd[-1] == str(out)


def test_clip_video_noop_without_clips() -> None:
    """clips が空なら FFmpeg を一切起動しない"""
    with patch("app.services.video_clip_service.subprocess.run") as run:
        clip_video("input.mp4", [], "out.mp4")

    run.assert_not_called()
