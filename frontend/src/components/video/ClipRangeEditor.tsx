import { useEffect, useRef, useState } from "react";

/** 区間の最小長（秒）。ハンドルがすれ違わないようにする。 */
const MIN_GAP = 0.5;
/** エッジパンが最大速度に達する、バー端からのポインタ距離（px）。 */
const EDGE_PAN_MAX_DIST = 160;
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

/** 選択範囲の前後に余白をとった拡大ウィンドウ（バーの表示範囲）を求める。 */
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

type Handle = "in" | "out" | null;

type Props = {
  duration: number;
  inPoint: number;
  outPoint: number;
  /** 動画の再生ヘッド位置（マーカー表示・スナップ・ウィンドウ追従に使う） */
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
 * 1本バーのクリップ区間エディタ。バーは選択範囲の前後だけを拡大表示し、
 * in / out ハンドルのドラッグで区間を調整する。
 *
 * - ハンドルをバーの端より外へドラッグすると、外に出た距離に応じた速度で
 *   表示範囲がスクロールし続ける（ジョイスティック式の可変速エッジパン）。
 * - 動画プレーヤー側のシーク・再生でウィンドウ外へ出たら表示範囲が追従する。
 *   離れた位置へは動画側でシークして「再生位置」ボタンでハンドルを飛ばせる。
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
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<Handle>(null);
  const [win, setWin] = useState(() => fitWindow(inPoint, outPoint, duration));

  // rAF ループや pointermove はレンダーを跨いで動くので、最新値は ref ミラーで参照する。
  const winRef = useRef(win);
  const rangeRef = useRef({ inPoint, outPoint });
  const durationRef = useRef(duration);
  const cbRef = useRef({ onChange, onSeek, onScrubEnd });
  /** ドラッグ中のポインタX座標（エッジパンの速度判定に使う） */
  const pointerXRef = useRef<number | null>(null);

  // ref ミラーの同期。後続の effect より先に宣言して常に最新値を読めるようにする。
  useEffect(() => {
    winRef.current = win;
    rangeRef.current = { inPoint, outPoint };
    durationRef.current = duration;
    cbRef.current = { onChange, onSeek, onScrubEnd };
  });

  // メタデータ読込などで duration が確定したタイミングでウィンドウを取り直す。
  useEffect(() => {
    setWin(fitWindow(rangeRef.current.inPoint, rangeRef.current.outPoint, duration));
  }, [duration]);

  // 動画プレーヤー側の操作（シーク・再生の進行）でウィンドウ外へ出たら追従する。
  useEffect(() => {
    if (dragging || duration <= 0) return;
    const w = winRef.current;
    if (currentTime >= w.start && currentTime <= w.end) return;
    const len = w.end - w.start;
    const start = Math.min(duration - len, Math.max(0, currentTime - len / 2));
    setWin({ start, end: start + len });
  }, [currentTime, dragging, duration]);

  const winLen = win.end - win.start;
  /** ウィンドウスケールでの位置 % */
  const pct = (t: number) =>
    winLen > 0 ? Math.min(100, Math.max(0, ((t - win.start) / winLen) * 100)) : 0;

  // ドラッグ中: pointermove はバー内での位置マッピング、rAF ループはバー外での
  // 可変速エッジパンを担当する（ポインタを静止させてもパンが続くように rAF で回す）。
  useEffect(() => {
    if (!dragging) return;
    const handle = dragging;

    /** MIN_GAP と [0, duration] でクランプしてハンドルを適用し、プレビューを追従させる。 */
    const apply = (t: number) => {
      const dur = durationRef.current;
      const { inPoint: i, outPoint: o } = rangeRef.current;
      if (handle === "in") {
        const v = Math.max(0, Math.min(t, o - MIN_GAP));
        cbRef.current.onChange(v, o);
        cbRef.current.onSeek(v);
      } else {
        const v = Math.min(dur, Math.max(t, i + MIN_GAP));
        cbRef.current.onChange(i, v);
        cbRef.current.onSeek(v);
      }
    };

    const onMove = (e: PointerEvent) => {
      pointerXRef.current = e.clientX;
      const el = trackRef.current;
      if (!el || durationRef.current <= 0) return;
      const rect = el.getBoundingClientRect();
      const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
      const w = winRef.current;
      apply(w.start + ratio * (w.end - w.start));
    };

    const onUp = () => {
      setDragging(null);
      pointerXRef.current = null;
      const { inPoint: i, outPoint: o } = rangeRef.current;
      // 確定位置で正確にシークし直し、ウィンドウを選択範囲に合わせ直す
      cbRef.current.onScrubEnd?.(handle === "in" ? i : o);
      setWin(fitWindow(i, o, durationRef.current));
    };

    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const el = trackRef.current;
      const x = pointerXRef.current;
      const dur = durationRef.current;
      if (!el || x == null || dur <= 0) return;
      const rect = el.getBoundingClientRect();
      const overshoot = x < rect.left ? rect.left - x : x > rect.right ? x - rect.right : 0;
      if (overshoot <= 0) return;
      const dir = x < rect.left ? -1 : 1;
      // パンするのは区間を「広げる」向きだけ。縮める向きの限界（反対側のハンドル）は
      // ウィンドウ内に必ず見えているのでパン不要。
      if ((handle === "in" && dir > 0) || (handle === "out" && dir < 0)) return;
      const w = winRef.current;
      const len = w.end - w.start;
      // 外に出た距離で速度を変える: 0.25×〜4× ウィンドウ幅/秒（二乗カーブで加速）
      const k = Math.min(1, overshoot / EDGE_PAN_MAX_DIST);
      const panPerSec = len * (0.25 + 3.75 * k * k);
      const start = Math.min(dur - len, Math.max(0, w.start + dir * panPerSec * dt));
      const nw = { start, end: start + len };
      winRef.current = nw; // 次フレームのために即時反映（再レンダー待ちにしない）
      setWin(nw);
      // ハンドルはパン方向側のウィンドウ端に張り付ける
      apply(dir < 0 ? nw.start : nw.end);
    };
    raf = requestAnimationFrame(tick);

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [dragging]);

  const onTrackClick = (e: React.MouseEvent) => {
    // ハンドルをクリックした場合は target が子要素になるのでシークしない。
    if (e.target !== e.currentTarget || duration <= 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    onSeek(win.start + ratio * winLen);
  };

  /** ハンドルを指定時刻へ動かす（ナッジ・再生位置スナップ共用）。ウィンドウ外なら再フィット。 */
  const moveHandleTo = (handle: "in" | "out", t: number) => {
    if (duration <= 0) return;
    if (handle === "in") {
      const v = Math.max(0, Math.min(round1(t), outPoint - MIN_GAP));
      onChange(v, outPoint);
      onSeek(v);
      if (v < win.start || v > win.end) setWin(fitWindow(v, outPoint, duration));
    } else {
      const v = Math.min(duration, Math.max(round1(t), inPoint + MIN_GAP));
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
      {/* 選択範囲周辺を拡大表示するバー */}
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
            startHandleDrag("in");
          }}
          className="absolute top-0 bottom-0 -ml-1.5 w-3 cursor-ew-resize rounded-sm bg-accent"
          style={{ left: `${pct(inPoint)}%` }}
        >
          {dragging === "in" && <DragBubble time={inPoint} />}
        </div>
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
          style={{ left: `${pct(outPoint)}%` }}
        >
          {dragging === "out" && <DragBubble time={outPoint} />}
        </div>
      </div>
      {/* バーの表示範囲（ウィンドウ）の目盛り */}
      <div className="mt-1 flex justify-between font-mono text-[10px] text-fg-4">
        <span>{fmt1(win.start)}</span>
        <span>{fmt1(win.end)}</span>
      </div>

      {/* 微調整（ナッジ・再生位置スナップ）と選択区間の表示 */}
      <div className="mt-2.5 space-y-1.5">
        <NudgeRow
          label="開始"
          value={inPoint}
          onNudge={(d) => moveHandleTo("in", inPoint + d)}
          onSnap={() => moveHandleTo("in", currentTime)}
          disabled={duration <= 0}
        />
        <NudgeRow
          label="終了"
          value={outPoint}
          onNudge={(d) => moveHandleTo("out", outPoint + d)}
          onSnap={() => moveHandleTo("out", currentTime)}
          disabled={duration <= 0}
        />
        <div className="font-mono text-[11.5px] text-fg-4">
          区間の長さ: {(outPoint - inPoint).toFixed(1)}s
        </div>
      </div>
    </div>
  );
}

/** ドラッグ中のハンドルの真上に出す時刻バブル。 */
function DragBubble({ time }: { time: number }) {
  return (
    <div className="pointer-events-none absolute -top-7 left-1/2 -translate-x-1/2 whitespace-nowrap rounded bg-fg px-1.5 py-0.5 font-mono text-[10.5px] text-bg">
      {fmt1(time)}
    </div>
  );
}

function NudgeRow({
  label,
  value,
  onNudge,
  onSnap,
  disabled,
}: {
  label: string;
  value: number;
  onNudge: (delta: number) => void;
  onSnap: () => void;
  disabled?: boolean;
}) {
  const btn =
    "cursor-pointer rounded-md border border-border bg-surface px-1.5 py-1 font-mono text-[11px] leading-none text-fg-2 transition-colors hover:bg-subtle-2 hover:text-fg disabled:pointer-events-none disabled:opacity-40";
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
          className={btn}
        >
          {d > 0 ? `+${d}` : d}s
        </button>
      ))}
      <button
        type="button"
        onClick={onSnap}
        disabled={disabled}
        aria-label={`${label}を再生位置に合わせる`}
        className={`${btn} ml-1 font-sans`}
      >
        再生位置
      </button>
    </div>
  );
}
