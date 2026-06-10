import { useEffect, useRef, useState } from "react";

/** 区間の最小長（秒）。ハンドルがすれ違わないようにする。 */
const MIN_GAP = 0.5;
/** ハンドルを拡大バーの端より外へドラッグしたときの1イベントあたりのパン量（ウィンドウ幅比）。 */
const EDGE_PAN_RATIO = 0.03;
/** ナッジボタンのステップ（秒）。 */
const NUDGE_STEPS = [-1, -0.1, 0.1, 1] as const;

const round1 = (t: number) => Math.round(t * 10) / 10;

/** m:ss.d 形式（小数1桁）。0.1s 単位の微調整結果が見えるようにする。 */
function fmt1(s: number) {
  const total = Math.round(s * 10) / 10;
  const m = Math.floor(total / 60);
  const sec = total - m * 60;
  return `${m}:${sec.toFixed(1).padStart(4, "0")}`;
}

/** 選択範囲の前後に余白をとった拡大ウィンドウ（下段バーの表示範囲）を求める。 */
function fitWindow(inP: number, outP: number, duration: number) {
  const margin = Math.max(2, (outP - inP) * 0.5);
  let start = inP - margin;
  let end = outP + margin;
  if (end - start >= duration) return { start: 0, end: duration };
  if (start < 0) {
    end -= start;
    start = 0;
  }
  if (end > duration) {
    start -= end - duration;
    end = duration;
  }
  return { start, end };
}

type Handle = "in" | "out" | "window" | null;

type Props = {
  duration: number;
  inPoint: number;
  outPoint: number;
  /** 動画の再生ヘッド位置（マーカー表示用） */
  currentTime: number;
  onChange: (inP: number, outP: number) => void;
  /** ハンドル追従・空き部分クリックでのシーク（呼び出し側で throttle する想定） */
  onSeek: (t: number) => void;
  /** ハンドルのドラッグ開始（プレビューの一時停止などに使う） */
  onScrubStart?: () => void;
  /** ハンドルのドラッグ確定。t は確定したハンドル位置（正確なシークのやり直しに使う） */
  onScrubEnd?: (t: number) => void;
};

/**
 * 2段構成のクリップ区間エディタ。
 * - 上段: 動画全長のオーバービュー。選択範囲の位置を示し、クリック/ドラッグで拡大ウィンドウを移動する。
 * - 下段: 選択範囲の前後だけを拡大したバー。in / out ハンドルのドラッグで区間を微調整する。
 * 長い動画でも数秒のクリップを十分なハンドル可動域で編集できるようにするのが狙い。
 */
export default function ClipRangeEditor({
  duration,
  inPoint,
  outPoint,
  currentTime,
  onChange,
  onSeek,
  onScrubStart,
  onScrubEnd,
}: Props) {
  const overviewRef = useRef<HTMLDivElement>(null);
  const detailRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<Handle>(null);
  const [win, setWin] = useState(() => fitWindow(inPoint, outPoint, duration));

  // メタデータ読込などで duration が確定したタイミングでウィンドウを取り直す。
  // ドラッグ中の in/out 変化では再フィットしたくないので最新値は ref で参照する。
  const rangeRef = useRef({ inPoint, outPoint });
  rangeRef.current = { inPoint, outPoint };
  useEffect(() => {
    setWin(fitWindow(rangeRef.current.inPoint, rangeRef.current.outPoint, duration));
  }, [duration]);

  const winLen = win.end - win.start;
  /** 上段（全長スケール）での位置 % */
  const ovPct = (t: number) =>
    duration > 0 ? Math.min(100, Math.max(0, (t / duration) * 100)) : 0;
  /** 下段（拡大ウィンドウスケール）での位置 % */
  const dtPct = (t: number) =>
    winLen > 0 ? Math.min(100, Math.max(0, ((t - win.start) / winLen) * 100)) : 0;

  /** ウィンドウ中心を指定時刻へ移動する（長さは維持、両端でクランプ）。 */
  const moveWindowCenter = (clientX: number) => {
    const el = overviewRef.current;
    if (!el || duration <= 0) return;
    const rect = el.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    const len = winLen;
    const start = Math.min(duration - len, Math.max(0, ratio * duration - len / 2));
    setWin({ start, end: start + len });
  };

  // ドラッグ中だけ window にリスナを張る（トラック外へ出ても追従させるため）。
  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: PointerEvent) => {
      if (dragging === "window") {
        moveWindowCenter(e.clientX);
        return;
      }
      const el = detailRef.current;
      if (!el || duration <= 0) return;
      const rect = el.getBoundingClientRect();
      const ratio = (e.clientX - rect.left) / rect.width; // クランプ前（端より外の判定に使う）
      const len = win.end - win.start;
      // 端より外へ出たらウィンドウを少しずつパンして続きを編集できるようにする
      if (ratio < 0 || ratio > 1) {
        const pan = len * EDGE_PAN_RATIO * (ratio < 0 ? -1 : 1);
        const start = Math.min(duration - len, Math.max(0, win.start + pan));
        setWin({ start, end: start + len });
      }
      const t = Math.min(duration, Math.max(0, win.start + ratio * len));
      if (dragging === "in") {
        const v = Math.max(0, Math.min(t, outPoint - MIN_GAP));
        onChange(v, outPoint);
        onSeek(v);
      } else {
        const v = Math.min(duration, Math.max(t, inPoint + MIN_GAP));
        onChange(inPoint, v);
        onSeek(v);
      }
    };
    const onUp = () => {
      setDragging(null);
      if (dragging !== "window") {
        // 確定位置で正確にシークし直し、ウィンドウを選択範囲に合わせ直す
        onScrubEnd?.(dragging === "in" ? inPoint : outPoint);
        setWin(fitWindow(inPoint, outPoint, duration));
      }
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dragging, win, inPoint, outPoint, duration, onChange, onSeek, onScrubEnd]);

  const onDetailClick = (e: React.MouseEvent) => {
    // ハンドルをクリックした場合は target が子要素になるのでシークしない。
    if (e.target !== e.currentTarget || duration <= 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    onSeek(win.start + ratio * winLen);
  };

  /** ±1s / ±0.1s ボタンによる微調整。丸めてからクランプし、結果の位置へシークする。 */
  const nudge = (handle: "in" | "out", delta: number) => {
    if (duration <= 0) return;
    if (handle === "in") {
      const v = Math.max(0, Math.min(round1(inPoint + delta), outPoint - MIN_GAP));
      onChange(v, outPoint);
      onSeek(v);
      if (v < win.start || v > win.end) setWin(fitWindow(v, outPoint, duration));
    } else {
      const v = Math.min(duration, Math.max(round1(outPoint + delta), inPoint + MIN_GAP));
      onChange(inPoint, v);
      onSeek(v);
      if (v < win.start || v > win.end) setWin(fitWindow(inPoint, v, duration));
    }
  };

  const startHandleDrag = (handle: "in" | "out") => {
    setDragging(handle);
    onScrubStart?.();
  };

  return (
    <div className="select-none">
      {/* 上段: 全長オーバービュー */}
      <div
        ref={overviewRef}
        onPointerDown={(e) => {
          if (duration <= 0) return;
          moveWindowCenter(e.clientX);
          setDragging("window");
        }}
        className="relative h-5 w-full cursor-pointer rounded-md bg-subtle-2"
        role="presentation"
        aria-label="全体タイムライン"
      >
        {/* 選択区間（位置の目印） */}
        <div
          className="pointer-events-none absolute top-1 bottom-1 rounded-[2px] bg-accent"
          style={{
            left: `${ovPct(inPoint)}%`,
            width: `${Math.max(0.4, ovPct(outPoint) - ovPct(inPoint))}%`,
          }}
        />
        {/* 拡大ウィンドウの枠 */}
        <div
          className="pointer-events-none absolute top-0 bottom-0 rounded-[4px] border border-fg-3"
          style={{ left: `${ovPct(win.start)}%`, width: `${ovPct(win.end) - ovPct(win.start)}%` }}
        />
        {/* 再生ヘッド */}
        <div
          className="pointer-events-none absolute top-0 bottom-0 w-px bg-fg"
          style={{ left: `${ovPct(currentTime)}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-fg-4">
        <span>0:00.0</span>
        <span>{fmt1(duration)}</span>
      </div>

      {/* 下段: 選択範囲周辺の拡大ビュー */}
      <div
        ref={detailRef}
        onClick={onDetailClick}
        className="relative mt-2 h-10 w-full rounded-md bg-subtle-2"
      >
        {/* 選択区間 */}
        <div
          className="pointer-events-none absolute top-0 bottom-0 border-x-2 border-accent bg-accent-soft"
          style={{ left: `${dtPct(inPoint)}%`, width: `${dtPct(outPoint) - dtPct(inPoint)}%` }}
        />
        {/* 再生ヘッド */}
        <div
          className="pointer-events-none absolute top-[-3px] bottom-[-3px] w-px bg-fg"
          style={{ left: `${dtPct(currentTime)}%` }}
        />
        {/* in ハンドル */}
        <div
          role="slider"
          aria-label="開始位置"
          aria-valuenow={Math.round(inPoint)}
          onPointerDown={(e) => {
            e.stopPropagation();
            startHandleDrag("in");
          }}
          className="absolute top-0 bottom-0 -ml-1.5 w-3 cursor-ew-resize rounded-sm bg-accent"
          style={{ left: `${dtPct(inPoint)}%` }}
        />
        {/* out ハンドル */}
        <div
          role="slider"
          aria-label="終了位置"
          aria-valuenow={Math.round(outPoint)}
          onPointerDown={(e) => {
            e.stopPropagation();
            startHandleDrag("out");
          }}
          className="absolute top-0 bottom-0 -ml-1.5 w-3 cursor-ew-resize rounded-sm bg-accent"
          style={{ left: `${dtPct(outPoint)}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-fg-4">
        <span>{fmt1(win.start)}</span>
        <span>{fmt1(win.end)}</span>
      </div>

      {/* 微調整（ナッジ）と選択区間の表示 */}
      <div className="mt-2.5 space-y-1.5">
        <NudgeRow label="開始" value={inPoint} onNudge={(d) => nudge("in", d)} disabled={duration <= 0} />
        <NudgeRow label="終了" value={outPoint} onNudge={(d) => nudge("out", d)} disabled={duration <= 0} />
        <div className="font-mono text-[11.5px] text-fg-4">
          区間の長さ: {(outPoint - inPoint).toFixed(1)}s
        </div>
      </div>
    </div>
  );
}

function NudgeRow({
  label,
  value,
  onNudge,
  disabled,
}: {
  label: string;
  value: number;
  onNudge: (delta: number) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="w-7 text-[11.5px] text-fg-3">{label}</span>
      <span className="w-16 font-mono text-[12.5px] text-accent-ink">{fmt1(value)}</span>
      {NUDGE_STEPS.map((d) => (
        <button
          key={d}
          type="button"
          onClick={() => onNudge(d)}
          disabled={disabled}
          aria-label={`${label}を${d > 0 ? "+" : ""}${d}秒ずらす`}
          className="cursor-pointer rounded-md border border-border bg-surface px-1.5 py-1 font-mono text-[11px] leading-none text-fg-2 transition-colors hover:bg-subtle-2 hover:text-fg disabled:pointer-events-none disabled:opacity-40"
        >
          {d > 0 ? `+${d}` : d}s
        </button>
      ))}
    </div>
  );
}
