import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router-dom";
import {
  deleteVideoVideosVideoIdDelete,
  getOutputVideoVideosVideoIdOutputGet,
  getVideoVideosVideoIdGet,
  listClipsByVideoVideosVideoIdClipsGet,
  listJobsByVideoVideosVideoIdJobsGet,
  retryJobJobsJobIdRetryPost,
  type ClipResponse,
  type VideoResponse,
} from "../api/generated";
import { authHeaders } from "../lib/auth";
import AppShell from "../components/layout/AppShell";
import StatusBadge from "../components/ui/StatusBadge";
import Button from "../components/ui/Button";
import DropdownMenu from "../components/ui/DropdownMenu";
import Stripes from "../components/ui/Stripes";
import EmptyState from "../components/ui/EmptyState";
import {
  IconChevR,
  IconClock,
  IconDownload,
  IconFilm,
  IconMore,
  IconPlay,
  IconRefresh,
  IconTrash,
} from "../components/ui/Icons";

function fmt(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}
function fmtDuration(s: number | null | undefined) {
  return s == null ? "—" : fmt(s);
}
function sumPlay(clips: ClipResponse[]) {
  return clips.reduce((a, c) => a + (c.end_time - c.start_time), 0);
}

export default function VideoDetailPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { id } = useParams<{ id: string }>();

  /* ─── queries ─── */

  const { data: videoRes, isLoading } = useQuery({
    queryKey: ["video", id],
    queryFn: () => getVideoVideosVideoIdGet(id!, { headers: authHeaders() }),
    enabled: !!id,
    refetchInterval: (q) => {
      const v = q.state.data?.status === 200 ? q.state.data.data : null;
      return v?.status === "queued" || v?.status === "processing" ? 3000 : false;
    },
  });
  const video: VideoResponse | null =
    videoRes?.status === 200 ? videoRes.data : null;

  const { data: jobsRes } = useQuery({
    queryKey: ["jobs", id],
    queryFn: () => listJobsByVideoVideosVideoIdJobsGet(id!, { headers: authHeaders() }),
    enabled: !!id,
    refetchInterval: (q) => {
      const jobs = q.state.data?.status === 200 ? q.state.data.data : [];
      const running = jobs.some((j) => j.status === "queued" || j.status === "processing");
      return running ? 3000 : false;
    },
  });
  const jobs = jobsRes?.status === 200 ? jobsRes.data : [];
  const failedJob = jobs.find((j) => j.status === "failed");

  const { data: clipsRes } = useQuery({
    queryKey: ["clips", id],
    queryFn: () => listClipsByVideoVideosVideoIdClipsGet(id!, { headers: authHeaders() }),
    enabled: !!id,
    refetchInterval: (q) => {
      const clips = q.state.data?.status === 200 ? q.state.data.data : [];
      const running = jobs.some((j) => j.status === "queued" || j.status === "processing");
      return running && clips.length === 0 ? 3000 : false;
    },
  });
  const clips: ClipResponse[] = clipsRes?.status === 200 ? clipsRes.data : [];

  // 連結済み動画は presigned URL を認証付きエンドポイントから取得し、
  // <video src> / ダウンロード href に直接セットする（バイト本体はR2から直接配信）。
  const { data: outputRes } = useQuery({
    queryKey: ["output", id],
    queryFn: () => getOutputVideoVideosVideoIdOutputGet(id!, { headers: authHeaders() }),
    enabled: !!id && video?.status === "completed",
  });
  const outputUrl =
    outputRes?.status === 200 ? (outputRes.data as { url: string }).url : null;

  /* ─── mutations ─── */

  const deleteMutation = useMutation({
    mutationFn: () => deleteVideoVideosVideoIdDelete(id!, { headers: authHeaders() }),
    onSuccess: () => navigate("/videos"),
    onError: () => alert("動画の削除に失敗しました"),
  });
  const retryMutation = useMutation({
    mutationFn: (jobId: string) =>
      retryJobJobsJobIdRetryPost(jobId, { headers: authHeaders() }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs", id] });
      queryClient.invalidateQueries({ queryKey: ["video", id] });
    },
    onError: () => alert("再実行に失敗しました"),
  });

  const onDelete = () => {
    if (window.confirm("本当にこの動画を削除しますか？")) deleteMutation.mutate();
  };
  const onRetry = () => {
    if (failedJob) retryMutation.mutate(failedJob.id);
  };

  /* ─── render ─── */

  if (isLoading || !video) {
    return (
      <AppShell>
        <div className="grid h-full place-items-center text-fg-3">読み込み中...</div>
      </AppShell>
    );
  }

  const isProcessing = video.status === "processing" || video.status === "queued";
  const isFailed = video.status === "failed";
  const isCompleted = video.status === "completed";

  return (
    <AppShell>
      <div className="scroll-thin h-full overflow-auto">
        <div className="mx-auto max-w-[1180px] px-8 pt-6 pb-16">
          {/* Breadcrumb */}
          <div className="mb-2.5 flex items-center gap-1.5 text-[12px] text-fg-3">
            <button
              onClick={() => navigate("/videos")}
              className="cursor-pointer bg-transparent p-0 text-fg-3 hover:text-fg"
            >
              動画一覧
            </button>
            <IconChevR size={12} className="text-fg-4" />
            <span className="text-fg-2">{video.title}</span>
          </div>

          {/* Header */}
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <h1 className="m-0 truncate text-[22px] font-semibold tracking-[-0.015em]">
                {video.title}
              </h1>
              <div className="mt-2 flex flex-wrap items-center gap-3.5 text-[12.5px] text-fg-3">
                <StatusBadge status={video.status} />
                <span className="flex items-center gap-1.5">
                  <IconClock size={12} className="text-fg-4" />
                  <span className="font-mono">{fmtDuration(video.duration)}</span>
                </span>
                {clips.length > 0 && (
                  <span className="flex items-center gap-1.5">
                    <IconFilm size={12} className="text-fg-4" />
                    <span className="font-mono">{clips.length} シーン</span>
                  </span>
                )}
                <span className="text-fg-4">
                  {new Date(video.created_at).toLocaleDateString("ja-JP")}
                </span>
              </div>
            </div>

            <div className="flex flex-none gap-1.5">
              {isCompleted && outputUrl && (
                <a
                  href={outputUrl}
                  download
                  className="no-underline"
                >
                  <Button kind="secondary" size="sm">
                    <IconDownload size={13} />
                    書き出し
                  </Button>
                </a>
              )}
              <DropdownMenu
                items={[
                  {
                    label: "削除",
                    icon: <IconTrash size={13} />,
                    onClick: onDelete,
                    variant: "danger",
                  },
                ]}
                align="right"
              >
                <Button kind="ghost" size="sm" aria-label="その他">
                  <IconMore size={14} />
                </Button>
              </DropdownMenu>
            </div>
          </div>

          {/* Content */}
          <div className="mt-5 grid grid-cols-1 items-start gap-6 md:grid-cols-[minmax(0,1fr)_320px]">
            <div>
              <PlayerBlock
                isCompleted={isCompleted}
                isProcessing={isProcessing}
                isFailed={isFailed}
                outputUrl={outputUrl}
              />

              <div className="mt-6 mb-3 flex items-baseline justify-between">
                <h2 className="m-0 text-[14px] font-semibold tracking-[-0.01em]">シーン一覧</h2>
              </div>

              {isProcessing && <ClipsSkeleton />}
              {isFailed && (
                <EmptyState
                  icon={<IconRefresh size={18} />}
                  title="処理に失敗しました"
                  description="もう一度お試しください。同じエラーが続く場合は動画を別のものに差し替えてください。"
                  actions={
                    <>
                      <Button
                        kind="secondary"
                        size="sm"
                        onClick={onRetry}
                        disabled={!failedJob || retryMutation.isPending}
                      >
                        <IconRefresh size={13} />
                        {retryMutation.isPending ? "再実行中..." : "処理を再実行"}
                      </Button>
                      <Button kind="ghost" size="sm" onClick={onDelete}>
                        <IconTrash size={13} />
                        動画を削除
                      </Button>
                    </>
                  }
                />
              )}
              {isCompleted && clips.length === 0 && (
                <EmptyState
                  icon={<IconFilm size={18} />}
                  title="シーンが検出されませんでした"
                  description={
                    <>
                      卓球台 / 選手が映っていない、もしくはプレー区間が短すぎる可能性があります。
                      <br />
                      別の動画でもう一度お試しください。
                    </>
                  }
                  actions={
                    <>
                      {failedJob && (
                        <Button
                          kind="secondary"
                          size="sm"
                          onClick={onRetry}
                          disabled={retryMutation.isPending}
                        >
                          処理を再実行
                        </Button>
                      )}
                      <Button kind="ghost" size="sm" onClick={onDelete}>
                        動画を削除
                      </Button>
                    </>
                  }
                />
              )}
              {isCompleted && clips.length > 0 && <ClipsGrid clips={clips} />}
            </div>

            {/* Right rail */}
            <aside className="rounded-[10px] border border-border bg-surface p-4">
              <div className="font-mono text-[10.5px] uppercase tracking-[0.1em] text-fg-4">
                {isProcessing ? "ステータス" : "サマリー"}
              </div>

              {isProcessing ? (
                <>
                  <div className="mt-3 mb-2 flex items-center gap-2.5">
                    <span className="h-2 w-2 rounded-full bg-warn text-warn animate-pulseDot" />
                    <span className="text-[13px] font-medium">処理中...</span>
                  </div>
                  <div className="mb-3.5 text-[12px] leading-[1.6] text-fg-3">
                    プレーシーンを抽出しています。完了するとシーンが表示されます。
                  </div>
                  <div className="mb-2 h-1 overflow-hidden rounded-full bg-subtle-2">
                    <div className="h-full w-2/5 rounded-full bg-warn animate-shimmer" />
                  </div>
                </>
              ) : (
                <>
                  <Stat label="検出シーン" value={`${clips.length}`} unit="件" />
                  <Stat label="プレー時間" value={fmt(sumPlay(clips))} mono />
                </>
              )}
            </aside>
          </div>
        </div>
      </div>
    </AppShell>
  );
}

/* ─── sub-components ─── */

function PlayerBlock({
  isCompleted,
  isProcessing,
  isFailed,
  outputUrl,
}: {
  isCompleted: boolean;
  isProcessing: boolean;
  isFailed: boolean;
  outputUrl: string | null;
}) {
  if (isCompleted && outputUrl) {
    return (
      <video
        controls
        className="aspect-video w-full rounded-[10px] bg-[#0e0f12]"
        src={outputUrl}
      />
    );
  }
  return (
    <div className="relative aspect-video w-full overflow-hidden rounded-[10px] bg-[#0e0f12]">
      <Stripes className="absolute inset-0 opacity-70" />
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(180deg, rgba(14,15,18,0) 40%, rgba(14,15,18,0.55) 100%)",
        }}
      />
      <div className="absolute inset-0 grid place-items-center text-white">
        {isProcessing ? (
          <div className="text-center">
            <div className="mx-auto h-14 w-14 rounded-full border-2 border-white/20 border-t-white animate-spin360" />
            <div className="mt-3.5 text-[14px] font-medium">処理中</div>
            <div className="mt-1 font-mono text-[11px] text-white/65">
              プレーシーンを抽出しています…
            </div>
          </div>
        ) : isFailed ? (
          <div className="text-center">
            <div className="text-[14px] font-medium">処理に失敗しました</div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function ClipsGrid({ clips }: { clips: ClipResponse[] }) {
  return (
    <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-2 lg:grid-cols-3">
      {clips.map((c, idx) => (
        <ClipCard key={c.id} clip={c} index={idx + 1} />
      ))}
    </div>
  );
}

function ClipCard({ clip, index }: { clip: ClipResponse; index: number }) {
  const length = Math.round(clip.end_time - clip.start_time);
  return (
    <div className="flex flex-col overflow-hidden rounded-[10px] border border-border bg-surface">
      <div className="relative aspect-video">
        <Stripes className="absolute inset-0" />
        <div className="absolute left-2 top-2 rounded bg-white/92 px-1.5 py-0.5 font-mono text-[10.5px] font-medium text-fg">
          #{String(index).padStart(2, "0")}
        </div>
        <div className="absolute bottom-2 right-2 rounded bg-[#14161a]/[0.78] px-1.5 py-0.5 font-mono text-[10.5px] text-white">
          {length}s
        </div>
      </div>
      <div className="flex items-center justify-between px-3 py-2.5">
        <span className="font-mono text-[12px] text-fg-2">
          {fmt(clip.start_time)} – {fmt(clip.end_time)}
        </span>
        <IconPlay size={13} className="text-fg-4" />
      </div>
    </div>
  );
}

function ClipsSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: 6 }).map((_, i) => (
        <div
          key={i}
          className="overflow-hidden rounded-[10px] border border-dashed border-border bg-surface opacity-75"
        >
          <div className="aspect-video animate-shimmer" />
          <div className="px-3 py-2.5">
            <div className="h-2.5 w-2/3 rounded bg-subtle-2" />
          </div>
        </div>
      ))}
    </div>
  );
}

function Stat({
  label,
  value,
  unit,
  mono,
}: {
  label: string;
  value: string;
  unit?: string;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between border-b border-dashed border-border py-2">
      <span className="text-[12px] text-fg-3">{label}</span>
      <span className={`text-[13.5px] font-medium ${mono ? "font-mono" : ""}`}>
        {value}
        {unit && <span className="ml-1 text-[11px] text-fg-3">{unit}</span>}
      </span>
    </div>
  );
}
