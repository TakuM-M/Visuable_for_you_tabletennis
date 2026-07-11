import { useRef, useState } from "react";
import Modal from "../ui/Modal";
import Button from "../ui/Button";
import ClipRangeEditor from "./ClipRangeEditor";
import { IconCheck, IconPlus } from "../ui/Icons";

type ClipRange = { start: number; end: number };

type Props = {
  open: boolean;
  /** 元動画の presigned URL */
  sourceUrl: string | null;
  /** メタデータ読み込み前のフォールバック長（videos.source_duration） */
  fallbackDuration?: number | null;
  /** 保存リクエスト送信中 */
  saving?: boolean;
  /** 編集対象の既存区間。null なら新規追加モード */
  initialRange?: ClipRange | null;
  onSubmit: (inP: number, outP: number) => void;
  onClose: () => void;
};

/** 新規追加時の区間の初期長（秒） */
const DEFAULT_LEN = 5;

/**
 * 元動画を再生しつつ、2段タイムラインで切り抜き区間を決めるモーダル。
 * initialRange を渡すと既存切り抜きの編集モードになる。
 *
 * 本体（ClipEditBody）は open のあいだだけマウントされるので、開くたびに state が
 * 初期化される（リセット用の useEffect が不要になる）。
 */
export default function ClipEditModal({
  open,
  sourceUrl,
  fallbackDuration,
  saving,
  initialRange,
  onSubmit,
  onClose,
}: Props) {
  const isEdit = initialRange != null;
  return (
    <Modal open={open} onClose={onClose} title={isEdit ? "切り抜きを編集" : "新規切り抜きを追加"}>
      {sourceUrl ? (
        <ClipEditBody
          sourceUrl={sourceUrl}
          fallbackDuration={fallbackDuration}
          saving={saving}
          initialRange={initialRange}
          onSubmit={onSubmit}
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

function ClipEditBody({
  sourceUrl,
  fallbackDuration,
  saving,
  initialRange,
  onSubmit,
  onClose,
}: {
  sourceUrl: string;
  fallbackDuration?: number | null;
  saving?: boolean;
  initialRange?: ClipRange | null;
  onSubmit: (inP: number, outP: number) => void;
  onClose: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const isEdit = initialRange != null;
  const initialDur = fallbackDuration && fallbackDuration > 0 ? fallbackDuration : 0;
  const [duration, setDuration] = useState(initialDur);
  const [currentTime, setCurrentTime] = useState(initialRange?.start ?? 0);
  const [inPoint, setInPoint] = useState(initialRange?.start ?? 0);
  const [outPoint, setOutPoint] = useState(
    initialRange?.end ?? (initialDur > 0 ? Math.min(DEFAULT_LEN, initialDur) : 0),
  );
  // 実際の動画長が確定したら一度だけ区間を初期化するためのフラグ（編集モードは初期値があるので不要）
  const initialized = useRef(isEdit);

  // シーク多重発行の制御。R2 越しのシークは重いので、シーク中に来た要求は
  // 保留して seeked 後に最新値だけを適用する（ドラッグ追従で詰まらせない）。
  const pendingSeek = useRef<number | null>(null);
  const seekBusy = useRef(false);

  const applyDuration = (d: number) => {
    if (!Number.isFinite(d) || d <= 0) return;
    setDuration(d);
    if (!initialized.current) {
      initialized.current = true;
      setInPoint(0);
      setOutPoint(Math.min(DEFAULT_LEN, d));
    } else {
      // フォールバック長と実際の動画長がずれていた場合に備えて上限でクランプ
      setOutPoint((o) => Math.min(o, d));
    }
  };

  const requestSeek = (t: number) => {
    const v = videoRef.current;
    if (!v) return;
    setCurrentTime(t); // 再生ヘッドの表示は即時更新
    if (seekBusy.current) {
      pendingSeek.current = t;
      return;
    }
    seekBusy.current = true;
    // ドラッグ追従中はキーフレーム単位の高速シークで十分（非対応ブラウザは通常シーク）
    if (typeof v.fastSeek === "function") v.fastSeek(t);
    else v.currentTime = t;
  };

  const onSeeked = () => {
    const v = videoRef.current;
    if (!v) return;
    if (pendingSeek.current != null) {
      // 保留していた最新値へチェーンして追いつく（seekBusy は維持）
      const t = pendingSeek.current;
      pendingSeek.current = null;
      v.currentTime = t;
    } else {
      seekBusy.current = false;
    }
  };

  const onScrubStart = () => {
    videoRef.current?.pause();
  };

  // ドラッグ確定時は currentTime で正確な位置へ合わせ直す（fastSeek はキーフレーム丸めされるため）
  const onScrubEnd = (t: number) => {
    const v = videoRef.current;
    if (!v) return;
    setCurrentTime(t);
    pendingSeek.current = null;
    seekBusy.current = true;
    v.currentTime = t;
  };

  const canSubmit = duration > 0 && outPoint > inPoint && !saving;

  return (
    <div className="space-y-4">
      <video
        ref={videoRef}
        src={sourceUrl}
        controls
        preload="auto"
        className="aspect-video w-full rounded-[10px] bg-[#0e0f12]"
        onLoadedMetadata={(e) => {
          applyDuration(e.currentTarget.duration);
          // 編集モードは対象区間の先頭から確認できるようにする
          if (initialRange) requestSeek(initialRange.start);
        }}
        onTimeUpdate={(e) => {
          // ハンドル追従シーク中は requestSeek 側が再生ヘッドを管理するので上書きしない
          if (!seekBusy.current) setCurrentTime(e.currentTarget.currentTime);
        }}
        onSeeked={onSeeked}
      />
      <p className="m-0 text-[12px] leading-[1.6] text-fg-3">
        ハンドルをドラッグして区間を調整してください（プレビューが追従します）。
        バーの端より外へドラッグすると表示範囲がスクロールし、外に行くほど速く動きます。
        離れた位置へは動画をシークして「再生位置」ボタンが使えます。
      </p>
      <ClipRangeEditor
        duration={duration}
        inPoint={inPoint}
        outPoint={outPoint}
        currentTime={currentTime}
        onChange={(i, o) => {
          setInPoint(i);
          setOutPoint(o);
        }}
        onSeek={requestSeek}
        onScrubStart={onScrubStart}
        onScrubEnd={onScrubEnd}
      />
      <div className="flex justify-end gap-2 pt-1">
        <Button kind="ghost" size="sm" onClick={onClose} disabled={saving}>
          キャンセル
        </Button>
        <Button
          kind="primary"
          size="sm"
          onClick={() => onSubmit(inPoint, outPoint)}
          disabled={!canSubmit}
        >
          {isEdit ? <IconCheck size={13} /> : <IconPlus size={13} />}
          {saving
            ? isEdit
              ? "保存中..."
              : "追加中..."
            : isEdit
              ? "変更を保存"
              : "この区間を追加"}
        </Button>
      </div>
    </div>
  );
}
