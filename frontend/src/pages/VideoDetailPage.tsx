import { useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router-dom";
import {
  deleteVideoVideosVideoIdDelete,
  exportVideoVideosVideoIdExportPost,
  getOutputVideoVideosVideoIdOutputGet,
  getSourceVideoVideosVideoIdSourceGet,
  getVideoVideosVideoIdGet,
  listClipsByVideoVideosVideoIdClipsGet,
  listJobsByVideoVideosVideoIdJobsGet,
  replaceClipsByVideoVideosVideoIdClipsPut,
  retryJobJobsJobIdRetryPost,
  type ClipResponse,
  type VideoResponse,
} from "../api/generated";
import { authHeaders } from "../lib/auth";
import { formatRetention } from "../lib/retention";
import AppShell from "../components/layout/AppShell";
import StatusBadge from "../components/ui/StatusBadge";
import Button from "../components/ui/Button";
import DropdownMenu from "../components/ui/DropdownMenu";
import Stripes from "../components/ui/Stripes";
import Thumbnail from "../components/ui/Thumbnail";
import EmptyState from "../components/ui/EmptyState";
import ClipEditModal from "../components/video/ClipEditModal";
import ClipPreviewPlayer, {
  type ClipPreviewHandle,
} from "../components/video/ClipPreviewPlayer";
import {
  IconChevR,
  IconClock,
  IconDownload,
  IconFilm,
  IconMore,
  IconPencil,
  IconPlay,
  IconPlus,
  IconRefresh,
  IconTrash,
} from "../components/ui/Icons";
import {
  normalizeClips,
  type ClipRange,
} from "../lib/clips";

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

type PlayerMode = "output" | "source" | "analyzing" | "exporting" | "failed" | "idle";

/** 切り抜きモーダルの状態。add = 新規追加、edit = 既存切り抜きの範囲編集 */
type ClipModalState = { mode: "add" } | { mode: "edit"; clip: ClipResponse } | null;

export default function VideoDetailPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { id } = useParams<{ id: string }>();
  const [clipModal, setClipModal] = useState<ClipModalState>(null);
  const previewRef = useRef<ClipPreviewHandle>(null);

  /* ─── queries ─── */

  const { data: videoRes, isLoading } = useQuery({
    queryKey: ["video", id],
    queryFn: () => getVideoVideosVideoIdGet(id!, { headers: authHeaders() }),
    enabled: !!id,
    refetchInterval: (q) => {
      const v = q.state.data?.status === 200 ? q.state.data.data : null;
      // 解析中（queued/processing）と書き出し中（processing）のあいだポーリング
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
  const jobRunning = jobs.some((j) => j.status === "queued" || j.status === "processing");

  const { data: clipsRes } = useQuery({
    queryKey: ["clips", id],
    queryFn: () => listClipsByVideoVideosVideoIdClipsGet(id!, { headers: authHeaders() }),
    enabled: !!id,
    refetchInterval: (q) => {
      const clips = q.state.data?.status === 200 ? q.state.data.data : [];
      return jobRunning && clips.length === 0 ? 3000 : false;
    },
  });
  const clips: ClipResponse[] = clipsRes?.status === 200 ? clipsRes.data : [];

  // 連結済み動画（output）と元動画（source）は presigned URL を認証付きエンドポイントから
  // 取得し、<video src> / ダウンロード href に直接セットする（バイト本体はR2から直接配信）。
  const { data: outputRes } = useQuery({
    queryKey: ["output", id],
    queryFn: () => getOutputVideoVideosVideoIdOutputGet(id!, { headers: authHeaders() }),
    enabled: !!id && video?.status === "completed",
  });
  const outputUrl =
    outputRes?.status === 200 ? (outputRes.data as { url: string }).url : null;
  // ダウンロードは Content-Disposition: attachment 付き URL を使う。
  // presigned URL はクロスオリジンで <a download> 属性が効かないため、
  // ヘッダ側で attachment を指定しないとモバイルで再生画面が開くだけになる。
  const outputDownloadUrl =
    outputRes?.status === 200 ? (outputRes.data.download_url ?? null) : null;

  const { data: sourceRes } = useQuery({
    queryKey: ["source", id],
    queryFn: () => getSourceVideoVideosVideoIdSourceGet(id!, { headers: authHeaders() }),
    enabled: !!id && (video?.status === "ready" || video?.status === "completed"),
  });
  const sourceUrl =
    sourceRes?.status === 200 ? (sourceRes.data as { url: string }).url : null;

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
  // 切り抜きの一括置換。出力動画は再生成しない（書き出しは別操作）。
  const replaceClipsMutation = useMutation({
    mutationFn: (items: ClipRange[]) => {
      const clamped = normalizeClips(
        items,
        video?.source_duration ?? null,
      );
      return replaceClipsByVideoVideosVideoIdClipsPut(
        id!,
        { clips: clamped },
        { headers: authHeaders() },
      );
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["clips", id] }),
    onError: () => alert("切り抜きの保存に失敗しました"),
  });
  const exportMutation = useMutation({
    mutationFn: () => exportVideoVideosVideoIdExportPost(id!, { headers: authHeaders() }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["video", id] }),
    onError: () => alert("書き出しの開始に失敗しました"),
  });

  /* ─── handlers ─── */

  const onDelete = () => {
    if (window.confirm("本当にこの動画を削除しますか？")) deleteMutation.mutate();
  };
  const onRetry = () => {
    if (failedJob) retryMutation.mutate(failedJob.id);
  };
  const onAddClip = (inP: number, outP: number) => {
    const items = [
      ...clips.map((c) => ({ start_time: c.start_time, end_time: c.end_time })),
      { start_time: inP, end_time: outP },
    ];
    replaceClipsMutation.mutate(items, { onSuccess: () => setClipModal(null) });
  };
  // 既存切り抜きの範囲変更。対象だけ差し替えた配列で一括置換する。
  const onEditClip = (clipId: string, inP: number, outP: number) => {
    const items = clips.map((c) =>
      c.id === clipId
        ? { start_time: inP, end_time: outP }
        : { start_time: c.start_time, end_time: c.end_time },
    );
    replaceClipsMutation.mutate(items, { onSuccess: () => setClipModal(null) });
  };
  const onDeleteClip = (clipId: string) => {
    const items = clips
      .filter((c) => c.id !== clipId)
      .map((c) => ({ start_time: c.start_time, end_time: c.end_time }));
    replaceClipsMutation.mutate(items);
  };
  const onExport = () => exportMutation.mutate();

  /* ─── render ─── */

  if (isLoading || !video) {
    return (
      <AppShell>
        <div className="grid h-full place-items-center text-fg-3">読み込み中...</div>
      </AppShell>
    );
  }

  const isFailed = video.status === "failed";
  const isCompleted = video.status === "completed";
  const isReady = video.status === "ready";
  const isAnalyzing = video.status === "queued" || (video.status === "processing" && jobRunning);
  const isExporting = video.status === "processing" && !jobRunning;
  const isEditable = isReady || isCompleted; // 切り抜き編集・書き出しが可能な状態
  const editLocked = replaceClipsMutation.isPending || exportMutation.isPending;
  const showClips = clips.length > 0 && (isEditable || isExporting);

  const playerMode: PlayerMode =
    isCompleted && outputUrl
      ? "output"
      : isExporting
        ? "exporting"
        : isAnalyzing
          ? "analyzing"
          : isFailed
            ? "failed"
            : isReady && sourceUrl
              ? "source"
              : "idle";
  // 擬似プレビュー（source モード）が出ているときだけシーン選択でジャンプ可能。
  const previewActive = playerMode === "source" && clips.length > 0;
  const retention = formatRetention(video.expires_at);

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
                {/* 書き出し中は「処理中」(processing) と区別して表示する */}
                <StatusBadge status={isExporting ? "exporting" : video.status} />
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
                {/* 保存期限。期限を過ぎた動画は自動削除されるので残り日数を明示する */}
                <span
                  className={`flex items-center gap-1.5 ${
                    retention.urgent ? "text-err-ink" : "text-fg-3"
                  }`}
                  title={`${retention.date} に自動削除されます`}
                >
                  <IconTrash size={12} className={retention.urgent ? "" : "text-fg-4"} />
                  <span>
                    保存期限 {retention.date}
                    <span className="ml-1.5 font-mono text-[11.5px]">
                      ({retention.label})
                    </span>
                  </span>
                </span>
              </div>
            </div>

            <div className="flex flex-none items-center gap-1.5">
              {isEditable && (
                <Button
                  kind="primary"
                  size="sm"
                  onClick={onExport}
                  disabled={clips.length === 0 || editLocked}
                >
                  <IconFilm size={13} />
                  {exportMutation.isPending
                    ? "書き出し中..."
                    : isCompleted
                      ? "再書き出し"
                      : "書き出し"}
                </Button>
              )}
              {isCompleted && outputDownloadUrl && (
                <a href={outputDownloadUrl} download className="no-underline">
                  <Button kind="secondary" size="sm">
                    <IconDownload size={13} />
                    ダウンロード
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
                mode={playerMode}
                outputUrl={outputUrl}
                sourceUrl={sourceUrl}
                thumbnailUrl={video.thumbnail_url}
                clips={clips}
                previewRef={previewRef}
              />

              <div className="mt-6 mb-3 flex items-center justify-between gap-2">
                <h2 className="m-0 text-[14px] font-semibold tracking-[-0.01em]">シーン一覧</h2>
                {isEditable && (
                  <Button
                    kind="secondary"
                    size="sm"
                    onClick={() => setClipModal({ mode: "add" })}
                    disabled={!sourceUrl || editLocked}
                  >
                    <IconPlus size={13} />
                    新規切り抜き
                  </Button>
                )}
              </div>

              {isAnalyzing && <ClipsSkeleton />}

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

              {isReady && clips.length === 0 && (
                <EmptyState
                  icon={<IconFilm size={18} />}
                  title="シーンがありません"
                  description="自動検出されたシーンはありません。「新規切り抜き」から区間を追加できます。"
                  actions={
                    <Button
                      kind="secondary"
                      size="sm"
                      onClick={() => setClipModal({ mode: "add" })}
                      disabled={!sourceUrl || editLocked}
                    >
                      <IconPlus size={13} />
                      新規切り抜き
                    </Button>
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
                    <Button kind="ghost" size="sm" onClick={onDelete}>
                      動画を削除
                    </Button>
                  }
                />
              )}

              {showClips && (
                <ClipsGrid
                  clips={clips}
                  onDelete={isEditable && !editLocked ? onDeleteClip : undefined}
                  onEdit={
                    isEditable && !editLocked && sourceUrl
                      ? (clip) => setClipModal({ mode: "edit", clip })
                      : undefined
                  }
                  onSelect={
                    previewActive ? (i) => previewRef.current?.jumpTo(i) : undefined
                  }
                />
              )}
            </div>

            {/* Right rail */}
            <aside className="rounded-[10px] border border-border bg-surface p-4">
              <div className="font-mono text-[10.5px] uppercase tracking-[0.1em] text-fg-4">
                {isAnalyzing || isExporting ? "ステータス" : "サマリー"}
              </div>

              {isAnalyzing ? (
                <BusyRail
                  title="解析中..."
                  desc="プレーシーンを抽出しています。完了するとシーンが表示され、編集できます。"
                />
              ) : isExporting ? (
                <BusyRail
                  title="書き出し中..."
                  desc="連結動画を生成しています。完了すると再生・ダウンロードできます。"
                />
              ) : (
                <>
                  <Stat label="検出シーン" value={`${clips.length}`} unit="件" />
                  <Stat label="プレー時間" value={fmt(sumPlay(clips))} mono />
                  {isReady && (
                    <div className="mt-3 text-[12px] leading-[1.6] text-fg-3">
                      解析が完了しました。区間を編集して「書き出し」すると連結動画が生成されます。
                    </div>
                  )}
                </>
              )}
            </aside>
          </div>
        </div>
      </div>

      <ClipEditModal
        open={clipModal != null}
        sourceUrl={sourceUrl}
        fallbackDuration={video.source_duration}
        saving={replaceClipsMutation.isPending}
        initialRange={
          clipModal?.mode === "edit"
            ? { start: clipModal.clip.start_time, end: clipModal.clip.end_time }
            : null
        }
        onSubmit={(inP, outP) => {
          if (clipModal?.mode === "edit") onEditClip(clipModal.clip.id, inP, outP);
          else onAddClip(inP, outP);
        }}
        onClose={() => setClipModal(null)}
      />
    </AppShell>
  );
}

/* ─── sub-components ─── */

function PlayerBlock({
  mode,
  outputUrl,
  sourceUrl,
  thumbnailUrl,
  clips,
  previewRef,
}: {
  mode: PlayerMode;
  outputUrl: string | null;
  sourceUrl: string | null;
  thumbnailUrl?: string | null;
  clips: ClipResponse[];
  previewRef: React.Ref<ClipPreviewHandle>;
}) {
  if (mode === "output" && outputUrl) {
    return (
      <video
        controls
        className="aspect-video w-full rounded-[10px] bg-[#0e0f12]"
        src={outputUrl}
      />
    );
  }
  if (mode === "source" && sourceUrl) {
    // 切り抜きがあれば「プレー区間のみ」の擬似プレビュー、無ければ元動画をそのまま。
    return clips.length > 0 ? (
      <ClipPreviewPlayer key={sourceUrl} ref={previewRef} src={sourceUrl} clips={clips} />
    ) : (
      <video
        controls
        className="aspect-video w-full rounded-[10px] bg-[#0e0f12]"
        src={sourceUrl}
      />
    );
  }
  return (
    <div className="relative aspect-video w-full overflow-hidden rounded-[10px] bg-[#0e0f12]">
      {/* 解析中・書き出し中でもどの動画かが分かるよう、サムネイルを下敷きにする */}
      <div className="absolute inset-0 opacity-70">
        <Thumbnail src={thumbnailUrl} alt="" />
      </div>
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(180deg, rgba(14,15,18,0) 40%, rgba(14,15,18,0.55) 100%)",
        }}
      />
      <div className="absolute inset-0 grid place-items-center text-white">
        {mode === "analyzing" ? (
          <div className="text-center">
            <div className="mx-auto h-14 w-14 rounded-full border-2 border-white/20 border-t-white animate-spin360" />
            <div className="mt-3.5 text-[14px] font-medium">処理中</div>
            <div className="mt-1 font-mono text-[11px] text-white/65">
              プレーシーンを抽出しています…
            </div>
          </div>
        ) : mode === "exporting" ? (
          <div className="text-center">
            <div className="mx-auto h-14 w-14 rounded-full border-2 border-white/20 border-t-white animate-spin360" />
            <div className="mt-3.5 text-[14px] font-medium">書き出し中</div>
            <div className="mt-1 font-mono text-[11px] text-white/65">
              連結動画を生成しています…
            </div>
          </div>
        ) : mode === "failed" ? (
          <div className="text-center">
            <div className="text-[14px] font-medium">処理に失敗しました</div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function ClipsGrid({
  clips,
  onDelete,
  onEdit,
  onSelect,
}: {
  clips: ClipResponse[];
  onDelete?: (id: string) => void;
  onEdit?: (clip: ClipResponse) => void;
  onSelect?: (index: number) => void;
}) {
  return (
    <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-2 lg:grid-cols-3">
      {clips.map((c, idx) => (
        <ClipCard
          key={c.id}
          clip={c}
          index={idx + 1}
          onDelete={onDelete}
          onEdit={onEdit}
          onSelect={onSelect ? () => onSelect(idx) : undefined}
        />
      ))}
    </div>
  );
}

function ClipCard({
  clip,
  index,
  onDelete,
  onEdit,
  onSelect,
}: {
  clip: ClipResponse;
  index: number;
  onDelete?: (id: string) => void;
  onEdit?: (clip: ClipResponse) => void;
  onSelect?: () => void;
}) {
  const length = Math.round(clip.end_time - clip.start_time);
  return (
    <div className="group flex flex-col overflow-hidden rounded-[10px] border border-border bg-surface">
      <div
        onClick={onSelect}
        className={`relative aspect-video ${onSelect ? "cursor-pointer" : ""}`}
      >
        <Stripes className="absolute inset-0" />
        {/* シーン選択（プレビューへジャンプ）可能なときのホバー時の再生オーバーレイ */}
        {onSelect && (
          <div className="absolute inset-0 grid place-items-center bg-black/0 opacity-0 transition-opacity group-hover:bg-black/35 group-hover:opacity-100">
            <span className="grid h-10 w-10 place-items-center rounded-full bg-black/55 text-white">
              <IconPlay size={18} />
            </span>
          </div>
        )}
        <div className="absolute left-2 top-2 rounded bg-white/92 px-1.5 py-0.5 font-mono text-[10.5px] font-medium text-fg">
          #{String(index).padStart(2, "0")}
        </div>
        {onEdit && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onEdit(clip);
            }}
            aria-label="この切り抜きの区間を編集"
            className="absolute right-9 top-2 grid h-6 w-6 cursor-pointer place-items-center rounded-md border-none bg-[#14161a]/[0.78] text-white transition-colors hover:bg-accent"
          >
            <IconPencil size={13} />
          </button>
        )}
        {onDelete && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete(clip.id);
            }}
            aria-label="この切り抜きを削除"
            className="absolute right-2 top-2 grid h-6 w-6 cursor-pointer place-items-center rounded-md border-none bg-[#14161a]/[0.78] text-white transition-colors hover:bg-err"
          >
            <IconTrash size={13} />
          </button>
        )}
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

function BusyRail({ title, desc }: { title: string; desc: string }) {
  return (
    <>
      <div className="mt-3 mb-2 flex items-center gap-2.5">
        <span className="h-2 w-2 rounded-full bg-warn text-warn animate-pulseDot" />
        <span className="text-[13px] font-medium">{title}</span>
      </div>
      <div className="mb-3.5 text-[12px] leading-[1.6] text-fg-3">{desc}</div>
      <div className="mb-2 h-1 overflow-hidden rounded-full bg-subtle-2">
        <div className="h-full w-2/5 rounded-full bg-warn animate-shimmer" />
      </div>
    </>
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
