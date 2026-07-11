import { useImperativeHandle, useRef, useState, type Ref } from "react";
import { IconPlay } from "../ui/Icons";

type Seg = { start_time: number; end_time: number };

/** 親（シーン一覧）からプレビューを操作するための命令的ハンドル。 */
export type ClipPreviewHandle = { jumpTo: (index: number) => void };

/** クリップ終端の検知・末尾判定に使う許容誤差（秒）。timeupdate の粒度を吸収する。 */
const EPS = 0.03;

/**
 * 元動画を1本の <video> で再生しつつ、切り抜き区間（clips）以外を自動スキップして
 * 「プレー中のみ」を擬似的に連続再生するプレビュープレイヤー。
 *
 * - clip の終端に達したら次の clip の先頭へシーク（末尾まで来たら停止）。
 * - ユーザーが手動シークしたら、その位置を含む clip / 次の clip に同期する。
 * - 連結（書き出し）と同じく start_time 昇順で再生する。
 *
 * 再生位置（どの clip を再生中か）は ref で保持し、副作用（useEffect）に依存しない。
 * src が変わったときは呼び出し側で key を切り替えて再マウントすることを想定。
 */
export default function ClipPreviewPlayer({
  src,
  clips,
  ref,
}: {
  src: string;
  clips: Seg[];
  ref?: Ref<ClipPreviewHandle>;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const indexRef = useRef(0); // 現在再生中の clip インデックス
  const programmatic = useRef(false); // 自前のシークかどうか（onSeeked の再スナップ抑止）
  const [sceneNo, setSceneNo] = useState(1);
  const [playing, setPlaying] = useState(false);

  // 毎レンダーで最新の clips を昇順に整える。各ハンドラはこの segs をクローズャ参照する。
  const segs = [...clips].sort((a, b) => a.start_time - b.start_time);

  const seekTo = (t: number) => {
    const v = videoRef.current;
    if (!v) return;
    programmatic.current = true;
    v.currentTime = t;
  };
  const goToIndex = (i: number) => {
    indexRef.current = i;
    setSceneNo(i + 1);
  };

  const onLoadedMetadata = () => {
    goToIndex(0);
    if (segs.length) seekTo(segs[0].start_time);
  };

  const onTimeUpdate = () => {
    const v = videoRef.current;
    if (!v || segs.length === 0) return;
    const i = Math.min(indexRef.current, segs.length - 1);
    const cur = segs[i];
    if (v.currentTime >= cur.end_time - EPS) {
      if (i + 1 < segs.length) {
        goToIndex(i + 1);
        seekTo(segs[i + 1].start_time);
      } else {
        v.pause();
      }
    }
  };

  const onSeeked = () => {
    // 自前のシーク（スキップ）なら再スナップしない
    if (programmatic.current) {
      programmatic.current = false;
      return;
    }
    const v = videoRef.current;
    if (!v || segs.length === 0) return;
    const t = v.currentTime;
    const inside = segs.findIndex((c) => t >= c.start_time && t < c.end_time);
    if (inside >= 0) {
      goToIndex(inside);
      return;
    }
    // 区間外（ギャップ）に落ちたら次の clip 先頭へ
    const next = segs.findIndex((c) => c.start_time > t);
    if (next >= 0) {
      goToIndex(next);
      seekTo(segs[next].start_time);
    } else {
      goToIndex(segs.length - 1);
    }
  };

  const onPlay = () => {
    setPlaying(true);
    const v = videoRef.current;
    if (!v || segs.length === 0) return;
    // 末尾で停止した状態から再生したら先頭へ戻す
    const last = segs[segs.length - 1];
    if (indexRef.current >= segs.length - 1 && v.currentTime >= last.end_time - 0.05) {
      goToIndex(0);
      seekTo(segs[0].start_time);
    }
  };

  // ネイティブの操作バー（スクラブバー）は出さず、クリックで再生/一時停止だけ行う。
  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) void v.play();
    else v.pause();
  };

  // シーン一覧から指定シーンへ飛んで再生する（プレビューを画面内へスクロール）。
  useImperativeHandle(ref, () => ({
    jumpTo: (index: number) => {
      const v = videoRef.current;
      if (!v || index < 0 || index >= segs.length) return;
      goToIndex(index);
      seekTo(segs[index].start_time);
      void v.play();
      containerRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    },
  }));

  return (
    <div ref={containerRef} className="relative">
      <video
        ref={videoRef}
        src={src}
        onLoadedMetadata={onLoadedMetadata}
        onTimeUpdate={onTimeUpdate}
        onSeeked={onSeeked}
        onPlay={onPlay}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        className="aspect-video w-full rounded-[10px] bg-[#0e0f12]"
      />
      {/* クリックで再生/一時停止。停止中は中央に再生ボタンを表示。 */}
      <button
        type="button"
        onClick={togglePlay}
        aria-label={playing ? "一時停止" : "再生"}
        className="absolute inset-0 grid cursor-pointer place-items-center border-none bg-transparent"
      >
        {!playing && (
          <span className="grid h-14 w-14 place-items-center rounded-full bg-black/45 text-white">
            <IconPlay size={24} />
          </span>
        )}
      </button>
      <div className="pointer-events-none absolute left-2 top-2 rounded bg-[#14161a]/[0.78] px-2 py-0.5 font-mono text-[10.5px] text-white">
        プレビュー · シーン {Math.min(sceneNo, segs.length || 1)}/{segs.length}
      </div>
    </div>
  );
}
