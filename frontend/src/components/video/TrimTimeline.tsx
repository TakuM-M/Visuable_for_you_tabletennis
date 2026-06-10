import { useCallback, useEffect, useRef, useState } from "react";

/** 区間の最小長（秒）。ハンドルがすれ違わないようにする。 */
const MIN_GAP = 0.5;

function fmt(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

type Handle = "in" | "out" | null;

type Props = {
  duration: number;
  inPoint: number;
  outPoint: number;
  /** 動画の再生ヘッド位置（マーカー表示用） */
  currentTime: number;
  onChange: (inP: number, outP: number) => void;
  /** トラックの空き部分クリックでシーク（任意） */
  onSeek?: (t: number) => void;
};

/**
 * ドラッグハンドル式のトリミングタイムライン。
 * [0, duration] の水平バー上で in / out の2ハンドルを動かして区間を決める。
 */
export default function TrimTimeline({
  duration,
  inPoint,
  outPoint,
  currentTime,
  onChange,
  onSeek,
}: Props) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<Handle>(null);

  const pct = (t: number) => (duration > 0 ? Math.min(100, Math.max(0, (t / duration) * 100)) : 0);

  const timeFromClientX = useCallback(
    (clientX: number) => {
      const el = trackRef.current;
      if (!el || duration <= 0) return 0;
      const rect = el.getBoundingClientRect();
      const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      return ratio * duration;
    },
    [duration],
  );

  // ドラッグ中だけ window にリスナを張る（トラック外へ出ても追従させるため）。
  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: PointerEvent) => {
      const t = timeFromClientX(e.clientX);
      if (dragging === "in") {
        onChange(Math.max(0, Math.min(t, outPoint - MIN_GAP)), outPoint);
      } else {
        onChange(inPoint, Math.min(duration, Math.max(t, inPoint + MIN_GAP)));
      }
    };
    const onUp = () => setDragging(null);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [dragging, inPoint, outPoint, duration, onChange, timeFromClientX]);

  const onTrackClick = (e: React.MouseEvent) => {
    // ハンドルをクリックした場合は target が子要素になるのでシークしない。
    if (!onSeek || e.target !== e.currentTarget) return;
    onSeek(timeFromClientX(e.clientX));
  };

  return (
    <div className="select-none">
      <div
        ref={trackRef}
        onClick={onTrackClick}
        className="relative h-10 w-full rounded-md bg-subtle-2"
      >
        {/* 選択区間 */}
        <div
          className="pointer-events-none absolute top-0 bottom-0 border-x-2 border-accent bg-accent-soft"
          style={{ left: `${pct(inPoint)}%`, width: `${pct(outPoint) - pct(inPoint)}%` }}
        />
        {/* 再生ヘッド */}
        <div
          className="pointer-events-none absolute top-[-3px] bottom-[-3px] w-px bg-fg"
          style={{ left: `${pct(currentTime)}%` }}
        />
        {/* in ハンドル */}
        <div
          role="slider"
          aria-label="開始位置"
          aria-valuenow={Math.round(inPoint)}
          onPointerDown={(e) => {
            e.stopPropagation();
            setDragging("in");
          }}
          className="absolute top-0 bottom-0 -ml-1.5 w-3 cursor-ew-resize rounded-sm bg-accent"
          style={{ left: `${pct(inPoint)}%` }}
        />
        {/* out ハンドル */}
        <div
          role="slider"
          aria-label="終了位置"
          aria-valuenow={Math.round(outPoint)}
          onPointerDown={(e) => {
            e.stopPropagation();
            setDragging("out");
          }}
          className="absolute top-0 bottom-0 -ml-1.5 w-3 cursor-ew-resize rounded-sm bg-accent"
          style={{ left: `${pct(outPoint)}%` }}
        />
      </div>

      {/* 目盛り */}
      <div className="mt-1.5 flex justify-between font-mono text-[10.5px] text-fg-4">
        <span>0:00</span>
        <span>{fmt(duration)}</span>
      </div>

      {/* 選択中の区間 */}
      <div className="mt-2 flex items-center gap-1.5 font-mono text-[12.5px]">
        <span className="text-accent-ink">{fmt(inPoint)}</span>
        <span className="text-fg-4">–</span>
        <span className="text-accent-ink">{fmt(outPoint)}</span>
        <span className="ml-1 text-fg-4">（{Math.round(outPoint - inPoint)}s）</span>
      </div>
    </div>
  );
}
