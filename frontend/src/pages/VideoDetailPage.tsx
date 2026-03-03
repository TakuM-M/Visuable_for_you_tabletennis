import { useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  getVideoVideosVideoIdGet,
  listJobsByVideoVideosVideoIdJobsGet,
  listClipsByVideoVideosVideoIdClipsGet,
  deleteVideoVideosVideoIdDelete,
} from "../api/generated";
import VideoStatusBadge from "../components/VideoStatusBadge";
import { authHeaders } from "../lib/auth";

export default function VideoDetailPage() {
  const navigate = useNavigate();
  const { id } = useParams();

  const { data, isLoading } = useQuery({
    queryKey: ["video", id],
    queryFn: () => getVideoVideosVideoIdGet(id!, { headers: authHeaders() }),
  });

  const deleteVideo = async () => {
    if (!window.confirm("本当にこの動画を削除しますか？")) return;
    try {
      await deleteVideoVideosVideoIdDelete(id!, { headers: authHeaders() });
      navigate("/");
    } catch (error) {
      alert("動画の削除に失敗しました");
    }
  }

  const { data: jobsData } = useQuery({
    queryKey: ["jobs", id],
    queryFn: () => listJobsByVideoVideosVideoIdJobsGet(id!, { headers: authHeaders() }),
    refetchInterval: (query) => {
      // jobがqueued/processingなら3秒ごとに再取得
      const jobs = query.state.data?.status === 200 ? query.state.data.data : [];
      const isProcessing = jobs.some(
        (j) => j.status === "queued" || j.status === "processing"
      );
      return isProcessing ? 3000 : false;
    },
  });

  const { data: clipsData } = useQuery({
    queryKey: ["clips", id],
    queryFn: () => listClipsByVideoVideosVideoIdClipsGet(id!, { headers: authHeaders() }),
    refetchInterval: (query) => {
      // クリップがまだ0件でjobが処理中なら3秒ごとに再取得
      const clips = query.state.data?.status === 200 ? query.state.data.data : [];
      const jobs = jobsData?.status === 200 ? jobsData.data : [];
      const isProcessing = jobs.some(
        (j) => j.status === "queued" || j.status === "processing"
      );
      return isProcessing && clips.length === 0 ? 3000 : false;
    },
  });

  const jobStatusColor = (status: string) => {
    switch (status) {
      case "processing": return "bg-yellow-100 text-yellow-600";
      case "completed":  return "bg-green-100 text-green-600";
      case "failed":     return "bg-red-100 text-red-600";
      default:           return "bg-gray-100 text-gray-600";
    }
  };

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  return (
    <div className="mx-auto max-w-2xl px-4 py-8">

      {/* 戻るボタン */}
      <button
        onClick={() => navigate("/")}
        className="mb-6 text-sm text-gray-500 hover:text-gray-700"
      >
        ← 一覧に戻る
      </button>

      <button onClick={deleteVideo} className="text-sm text-red-500 hover:text-red-700">
        動画を削除
      </button>

      {isLoading ? (
        <p>読み込み中...</p>
      ) : (
        <div className="space-y-6">

          {/* 動画情報カード */}
          {data?.status === 200 && (
            <div className="rounded-lg border bg-white p-6 shadow-sm">
              <div className="flex items-start justify-between">
                <h1 className="text-2xl font-bold text-gray-800">{data.data.title}</h1>
                <VideoStatusBadge status={data.data.status} />
              </div>
              <p className="mt-2 text-sm text-gray-400">
                {new Date(data.data.created_at).toLocaleDateString("ja-JP")}
              </p>
            </div>
          )}

          {/* ジョブ一覧 */}
          {jobsData?.status === 200 && (
            <div>
              <h2 className="mb-3 text-lg font-semibold text-gray-700">処理ジョブ</h2>
              {jobsData.data.length === 0 ? (
                <p className="text-sm text-gray-400">ジョブはまだありません</p>
              ) : (
                <ul className="space-y-2">
                  {jobsData.data.map((job) => (
                    <li key={job.id} className="rounded-lg border bg-white p-4 shadow-sm">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">ジョブ ID: {job.id.slice(0, 8)}...</span>
                        <span className={`rounded-full px-2 py-1 text-xs ${jobStatusColor(job.status)}`}>
                          {job.status === "queued" && "待機中"}
                          {job.status === "processing" && "処理中..."}
                          {job.status === "completed" && "完了"}
                          {job.status === "failed" && "失敗"}
                        </span>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {/* クリップ一覧 */}
          {clipsData?.status === 200 && (
            <div>
              <h2 className="mb-3 text-lg font-semibold text-gray-700">
                検出されたプレーシーン
              </h2>
              {clipsData.data.length === 0 ? (
                <p className="text-sm text-gray-400">
                  {jobsData?.status === 200 &&
                  jobsData.data.some((j) => j.status === "queued" || j.status === "processing")
                    ? "ML処理中です。しばらくお待ちください..."
                    : "プレーシーンは検出されませんでした"}
                </p>
              ) : (
                <ul className="space-y-2">
                  {clipsData.data.map((clip, index) => (
                    <li
                      key={clip.id}
                      className="rounded-lg border bg-white p-4 shadow-sm"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-gray-700">
                          シーン {index + 1}
                        </span>
                        <span className="text-sm text-gray-500">
                          {formatTime(clip.start_time)} 〜 {formatTime(clip.end_time)}
                        </span>
                      </div>
                      <p className="mt-1 text-xs text-gray-400">
                        長さ: {Math.round(clip.end_time - clip.start_time)}秒
                      </p>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

        </div>
      )}
    </div>
  );
}
