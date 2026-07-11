// "exporting" はサーバ上の status ではなく、フロントが「processing かつ実行中
// ジョブなし＝書き出し中」と判別したときに表示用として渡す擬似ステータス。
type Status =
  | "uploaded"
  | "queued"
  | "processing"
  | "ready"
  | "exporting"
  | "completed"
  | "failed"
  | (string & {});

type Props = { status: Status; className?: string };

type Spec = { dot: string; bg: string; fg: string; border: string; label: string; pulse?: boolean };

const MAP: Record<string, Spec> = {
  uploaded:   { dot: "bg-accent", bg: "bg-accent-soft", fg: "text-accent-ink", border: "border-accent-ink/15", label: "アップロード済み" },
  queued:     { dot: "bg-warn",   bg: "bg-warn-soft",   fg: "text-warn-ink",   border: "border-warn-ink/15",   label: "処理待ち" },
  processing: { dot: "bg-warn",   bg: "bg-warn-soft",   fg: "text-warn-ink",   border: "border-warn-ink/15",   label: "処理中", pulse: true },
  ready:      { dot: "bg-accent", bg: "bg-accent-soft", fg: "text-accent-ink", border: "border-accent-ink/15", label: "編集可能" },
  exporting:  { dot: "bg-warn",   bg: "bg-warn-soft",   fg: "text-warn-ink",   border: "border-warn-ink/15",   label: "書き出し中", pulse: true },
  completed:  { dot: "bg-ok",     bg: "bg-ok-soft",     fg: "text-ok-ink",     border: "border-ok-ink/15",     label: "完了" },
  failed:     { dot: "bg-err",    bg: "bg-err-soft",    fg: "text-err-ink",    border: "border-err-ink/15",    label: "失敗" },
};

export default function StatusBadge({ status, className = "" }: Props) {
  const s: Spec =
    MAP[status] ??
    { dot: "bg-fg-3", bg: "bg-subtle", fg: "text-fg-2", border: "border-border", label: status };

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-[3px] text-[11.5px] font-medium ${s.bg} ${s.fg} ${s.border} ${className}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${s.dot} ${s.pulse ? "animate-pulseDot" : ""}`} />
      {s.label}
    </span>
  );
}
