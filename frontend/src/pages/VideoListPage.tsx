import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import {
  listVideosVideosGet,
  type VideoResponse,
} from "../api/generated";
import { authHeaders } from "../lib/auth";
import AppShell from "../components/layout/AppShell";
import StatusBadge from "../components/ui/StatusBadge";
import Button from "../components/ui/Button";
import Stripes from "../components/ui/Stripes";
import EmptyState from "../components/ui/EmptyState";
import { IconFilm, IconPlus} from "../components/ui/Icons";

/**
 * Format seconds → mm:ss; "—" when missing.
 */
function fmtDuration(seconds: number | null | undefined) {
  if (seconds == null) return "—";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function fmtDateTime(iso: string) {
  const d = new Date(iso);
  const date = d.toLocaleDateString("ja-JP", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).replaceAll("/", "-");
  const time = d.toLocaleTimeString("ja-JP", {
    hour: "2-digit",
    minute: "2-digit",
  });
  return `${date} ${time}`;
}

export default function VideoListPage() {
  const navigate = useNavigate();

  const { data, isLoading } = useQuery({
    queryKey: ["videos"],
    queryFn: () => listVideosVideosGet({ headers: authHeaders() }),
  });

  const videos: VideoResponse[] = data?.status === 200 ? data.data : [];

  return (
    <AppShell>
      <div className="scroll-thin h-full overflow-auto">
        <div className="mx-auto max-w-[1040px] px-4 pt-6 pb-16 sm:px-8 sm:pt-7">
          {/* Page header */}
          <div className="mb-5 flex flex-wrap items-end justify-between gap-x-4 gap-y-3">
            <div>
              <div className="mb-1.5 font-mono text-[11px] uppercase tracking-[0.1em] text-fg-4">
                Library
              </div>
              <h1 className="m-0 text-[22px] font-semibold tracking-[-0.015em]">動画一覧</h1>
              <p className="mt-1.5 text-[13px] text-fg-3">
                {isLoading ? "読み込み中..." : `${videos.length}件`}
              </p>
            </div>
            <div className="flex gap-2">
              <Button kind="primary" size="sm" onClick={() => navigate("/videos/new")}>
                <IconPlus size={13} />
                動画を追加
              </Button>
            </div>
          </div>

          {/* Table header（モバイルでは行を2段組みにするため非表示） */}
          <div className="hidden h-8 grid-cols-[1fr_120px_120px_150px] items-center border-b border-border px-3.5 font-mono text-[10.5px] uppercase tracking-[0.08em] text-fg-4 sm:grid">
            <span>タイトル</span>
            <span className="text-right">再生時間</span>
            <span className="text-right">状態</span>
            <span className="text-right">アップロード</span>
          </div>

          {/* Rows */}
          {!isLoading && videos.length === 0 ? (
            <div className="mt-6">
              <EmptyState
                icon={<IconFilm size={18} />}
                title="まだ動画がありません"
                description="最初の動画をアップロードすると、プレーシーンが自動で抽出されます。"
                actions={
                  <Button kind="primary" size="sm" onClick={() => navigate("/videos/new")}>
                    <IconPlus size={13} />
                    動画を追加
                  </Button>
                }
              />
            </div>
          ) : (
            videos.map((v) => (
              <button
                type="button"
                key={v.id}
                onClick={() => navigate(`/videos/${v.id}`)}
                className="grid w-full cursor-pointer grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-2 border-b border-border px-3.5 py-3 text-left hover:bg-subtle sm:grid-cols-[1fr_120px_120px_150px] sm:gap-0"
              >
                {/* モバイル: 1段目=タイトル+状態バッジ / 2段目=再生時間+日時（order-* で並べ替え） */}
                <div className="flex min-w-0 items-center gap-3">
                  <div className="h-6 w-9 flex-none overflow-hidden rounded">
                    <Stripes />
                  </div>
                  <div className="min-w-0">
                    <div className="truncate text-[13.5px] font-medium">{v.title}</div>
                    <div className="font-mono text-[10.5px] text-fg-4">MP4</div>
                  </div>
                </div>
                <div className="order-2 font-mono text-[12px] text-fg-2 sm:order-none sm:text-right">
                  {fmtDuration(v.duration)}
                </div>
                <div className="order-1 flex justify-end sm:order-none">
                  <StatusBadge status={v.status} />
                </div>
                <div className="order-3 text-right text-[12px] text-fg-3 sm:order-none">
                  {fmtDateTime(v.created_at)}
                </div>
              </button>
            ))
          )}
        </div>
      </div>
    </AppShell>
  );
}
