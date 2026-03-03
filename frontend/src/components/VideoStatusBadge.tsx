type Props = {
    status: string;
}

const colorMap: Record<string, string> = {
  uploaded:   "bg-blue-100 text-blue-600",
  queued:     "bg-yellow-100 text-yellow-600",
  processing: "bg-yellow-100 text-yellow-600",
  completed:  "bg-green-100 text-green-600",
  failed:     "bg-red-100 text-red-600",
};

const labelMap: Record<string, string> = {
  uploaded:   "アップロード済み",
  queued:     "処理待ち",
  processing: "処理中...",
  completed:  "処理完了",
  failed:     "処理失敗",
};

export default function VideoStatusBadge({ status }: Props) {
  return (
    <span className={`rounded-full px-2 py-1 text-xs ${colorMap[status] ?? "bg-gray-100 text-gray-600"}`}>
      {labelMap[status] ?? status}
    </span>
  );
}