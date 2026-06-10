import { useRef, useState } from "react";
import Modal from "../ui/Modal";
import Button from "../ui/Button";
import TrimTimeline from "./TrimTimeline";
import { IconPlus } from "../ui/Icons";

type Props = {
  open: boolean;
  /** 元動画の presigned URL */
  sourceUrl: string | null;
  /** メタデータ読み込み前のフォールバック長（videos.source_duration） */
  fallbackDuration?: number | null;
  /** 追加リクエスト送信中 */
  adding?: boolean;
  onAdd: (inP: number, outP: number) => void;
  onClose: () => void;
};

/** 区間の初期長（秒） */
const DEFAULT_LEN = 5;

/**
 * 元動画を再生しつつ、タイムラインのハンドルで新規切り抜きの区間を決めるモーダル。
 *
 * 本体（AddClipBody）は open のあいだだけマウントされるので、開くたびに state が
 * 初期化される（リセット用の useEffect が不要になる）。
 */
export default function AddClipModal({
  open,
  sourceUrl,
  fallbackDuration,
  adding,
  onAdd,
  onClose,
}: Props) {
  return (
    <Modal open={open} onClose={onClose} title="新規切り抜きを追加">
      {sourceUrl ? (
        <AddClipBody
          sourceUrl={sourceUrl}
          fallbackDuration={fallbackDuration}
          adding={adding}
          onAdd={onAdd}
          onClose={onClose}
        />
      ) : (
        <div className="grid h-40 place-items-center text-[13px] text-fg-3">
          元動画を読み込んでいます...
        </div>
      )}
    </Modal>
  );
}

function AddClipBody({
  sourceUrl,
  fallbackDuration,
  adding,
  onAdd,
  onClose,
}: {
  sourceUrl: string;
  fallbackDuration?: number | null;
  adding?: boolean;
  onAdd: (inP: number, outP: number) => void;
  onClose: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const initialDur = fallbackDuration && fallbackDuration > 0 ? fallbackDuration : 0;
  const [duration, setDuration] = useState(initialDur);
  const [currentTime, setCurrentTime] = useState(0);
  const [inPoint, setInPoint] = useState(0);
  const [outPoint, setOutPoint] = useState(initialDur > 0 ? Math.min(DEFAULT_LEN, initialDur) : 0);
  // 実際の動画長が確定したら一度だけ区間を初期化するためのフラグ
  const initialized = useRef(false);

  const applyDuration = (d: number) => {
    if (!Number.isFinite(d) || d <= 0) return;
    setDuration(d);
    if (!initialized.current) {
      initialized.current = true;
      setInPoint(0);
      setOutPoint(Math.min(DEFAULT_LEN, d));
    }
  };

  const seek = (t: number) => {
    setCurrentTime(t);
    if (videoRef.current) videoRef.current.currentTime = t;
  };

  const canAdd = duration > 0 && outPoint > inPoint && !adding;

  return (
    <div className="space-y-4">
      <video
        ref={videoRef}
        src={sourceUrl}
        controls
        className="aspect-video w-full rounded-[10px] bg-[#0e0f12]"
        onLoadedMetadata={(e) => applyDuration(e.currentTarget.duration)}
        onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
      />
      <p className="m-0 text-[12px] leading-[1.6] text-fg-3">
        ハンドルをドラッグして切り抜く区間を指定してください。
        バーの空き部分をクリックすると再生位置を移動できます。
      </p>
      <TrimTimeline
        duration={duration}
        inPoint={inPoint}
        outPoint={outPoint}
        currentTime={currentTime}
        onChange={(i, o) => {
          setInPoint(i);
          setOutPoint(o);
        }}
        onSeek={seek}
      />
      <div className="flex justify-end gap-2 pt-1">
        <Button kind="ghost" size="sm" onClick={onClose} disabled={adding}>
          キャンセル
        </Button>
        <Button
          kind="primary"
          size="sm"
          onClick={() => onAdd(inPoint, outPoint)}
          disabled={!canAdd}
        >
          <IconPlus size={13} />
          {adding ? "追加中..." : "この区間を追加"}
        </Button>
      </div>
    </div>
  );
}
