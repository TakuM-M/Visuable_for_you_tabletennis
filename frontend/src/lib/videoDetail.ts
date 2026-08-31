import type { VideoStatus } from "../api/generated";

export type PlayerMode =
  | "output"
  | "source"
  | "analyzing"
  | "exporting"
  | "failed"
  | "idle";

export type VideoDetailInput = {
  status: VideoStatus;
  jobRunning: boolean;
  outputUrl: string | null;
  sourceUrl: string | null;
  clipCount: number;
};

export type VideoDetailState = {
  isFailed: boolean;
  isCompleted: boolean;
  isReady: boolean;
  isAnalyzing: boolean;
  isExporting: boolean;
  isEditable: boolean;
  showClips: boolean;
  playerMode: PlayerMode;
  previewActive: boolean;
};

export function resolveVideoDetailState({
  status,
  jobRunning,
  outputUrl,
  sourceUrl,
  clipCount,
}: VideoDetailInput): VideoDetailState {
  const isFailed = status === "failed";
  const isCompleted = status === "completed";
  const isReady = status === "ready";
  const isAnalyzing = status === "queued" || (status === "processing" && jobRunning);
  const isExporting = status === "processing" && !jobRunning;
  const isEditable = isReady || isCompleted;

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

  return {
    isFailed,
    isCompleted,
    isReady,
    isAnalyzing,
    isExporting,
    isEditable,
    showClips: clipCount > 0 && (isEditable || isExporting),
    playerMode,
    previewActive: playerMode === "source" && clipCount > 0,
  };
}
